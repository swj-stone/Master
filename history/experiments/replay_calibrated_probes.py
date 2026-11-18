"""Replay frozen development probes while capturing supports and prediction arrays."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.document_vote_benchmark import file_hash


def replay(kind, output, reference, arrays_reference):
    if kind == "a":
        module = importlib.import_module("experiments.probe_cheap_calibrated_document")
        entry = "cheap_calibrated_document_top_from_training"
        methods = ("full", "old", "calibrated")
    else:
        module = importlib.import_module("experiments.probe_rescaled_calibrated_document")
        entry = "rescaled_calibrated_document_top_from_training"
        methods = ("full", "old", "a2", "b2")
    select, predict = getattr(module, entry), module.fit_predict
    captures, counters = [], []

    def capture_select(*args, **kwargs):
        cols, info = select(*args, **kwargs)
        captures.append({name: value.copy() for name, value in info.items() if isinstance(value, np.ndarray)})
        captures[-1]["selected_support"] = cols.copy()
        if "eta" in info:
            captures[-1]["eta"] = np.array(info["eta"])
        counters.append(0)
        return cols, info

    def capture_predict(X, y, Xt, cols, *args, **kwargs):
        prediction = predict(X, y, Xt, cols, *args, **kwargs)
        method = methods[counters[-1]]
        counters[-1] += 1
        captures[-1][method + "_support"] = np.array(cols, copy=True)
        captures[-1][method + "_prediction"] = np.array(prediction, copy=True)
        return prediction

    setattr(module, entry, capture_select)
    module.fit_predict = capture_predict
    module.probe(output)
    assert len(captures) == 4 and counters == [len(methods)] * 4
    datasets = pd.read_csv(output / "diagnostics.csv").dataset.tolist()
    assert datasets == ["sms", "amazon_cells", "imdb", "yelp"]
    assert json.loads((output / "protocol.json").read_text()) == json.loads((reference / "protocol.json").read_text())
    csv_count = 0
    for previous in reference.glob("*.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(output / previous.name), pd.read_csv(previous),
                                      check_exact=False, atol=1e-12, rtol=0)
        csv_count += 1
    array_count = 0
    for dataset, arrays in zip(datasets, captures):
        np.savez_compressed(output / (dataset + "-arrays.npz"), **arrays)
        if arrays_reference is not None:
            with np.load(arrays_reference / (dataset + "-arrays.npz")) as expected:
                assert set(arrays) == set(expected.files)
                for name, value in arrays.items():
                    np.testing.assert_array_equal(value, expected[name], err_msg=dataset + ":" + name)
                    array_count += 1
    audit = dict(kind=kind, archived_csv_equal_atol=1e-12, archived_csv_count=csv_count,
                 arrays_compared_exactly=array_count, original_array_archive_existed=False,
                 interpretation="Original development files contain scores/curves, not supports/predictions. "
                                "Their CSVs are checked; two new independent processes compare full arrays.",
                 wrapper_sha256=file_hash(Path(__file__)),
                 captured_array_sha256={file.name: file_hash(file) for file in sorted(output.glob("*.npz"))})
    (output / "replay-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("a", "a2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arrays-reference", type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        replay(args.kind, args.output, args.reference, args.arrays_reference)
