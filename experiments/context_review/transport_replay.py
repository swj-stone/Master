"""Reproduce the transport benchmark in ignored output and compare published predictions."""
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments.context_review import transport_audit, transport_mammographic, transport_predict
from experiments.context_review import transport_score, transport_split
from experiments.context_review.transport_data import DATA, EVIDENCE, ROOT, SPEC, fetch, read
from experiments.context_study.benchmark import store
from experiments.context_study.data import digest


def replay():
    new = ROOT / "results/context_review/transport-replay"
    evidence = new / "evidence"
    evidence.mkdir(exist_ok=True, parents=True)
    sources = {row["dataset"]: row for row in read(EVIDENCE / "sources.json")}
    DATA.mkdir(exist_ok=True, parents=True)
    for source in read(SPEC)["sources"]:
        downloaded = fetch(source)
        if downloaded["sha256"] != sources[source[0]]["sha256"]:
            raise ValueError(f"Archive differs from the published source: {source[0]}")
    shutil.copyfile(EVIDENCE / "sources.json", evidence / "sources.json")
    for module in (transport_split, transport_predict, transport_audit, transport_score):
        module.RUN = new
        module.EVIDENCE = evidence
    manifest = pd.read_csv(EVIDENCE / "fold-manifest.csv")
    layouts = [new / row.dataset / f"fold-{row.fold}/layout.json" for row in manifest.itertuples()]
    if not all(path.exists() for path in layouts):
        transport_split.prepare()
    transport_predict.predict()
    transport_score.score()
    transport_audit.audit()
    repeated = pd.read_csv(evidence / "fold-manifest.csv")
    identity = ["dataset", "fold"]
    first = manifest.set_index(identity).sort_index()
    again = repeated.set_index(identity).sort_index()
    pd.testing.assert_index_equal(first.index, again.index)
    # NPZ arrays contain only model probabilities and row IDs, independent of clock readings.
    pd.testing.assert_series_equal(first.predictions, again.predictions)
    folds = ["dataset", "fold", "arm"]
    old = pd.read_csv(EVIDENCE / "fold-results.csv").set_index(folds).sort_index()
    new_rows = pd.read_csv(evidence / "fold-results.csv").set_index(folds).sort_index()
    structural = ["anchor", "nominee", "selected", "passed", "breach", "masks", "total_fits"]
    pd.testing.assert_frame_equal(old[structural], new_rows[structural])
    np.testing.assert_allclose(old[["u","auc","accuracy"]], new_rows[["u","auc","accuracy"]],
                               atol=1e-12, rtol=0)
    transport_mammographic.NEW = new / "mammographic-four"
    transport_mammographic.RUN = new
    transport_mammographic.EVIDENCE = evidence
    transport_mammographic.predict()
    transport_mammographic.score()
    old_sensitivity = pd.read_csv(EVIDENCE / "birads-four-features.csv").set_index(folds).sort_index()
    repeated_sensitivity = pd.read_csv(evidence / "birads-four-features.csv").set_index(folds).sort_index()
    pd.testing.assert_frame_equal(old_sensitivity[structural], repeated_sensitivity[structural])
    old_hashes = read(EVIDENCE / "birads-sensitivity-lock.json")["predictions"]
    new_hashes = read(evidence / "birads-sensitivity-lock.json")["predictions"]
    assert old_hashes == new_hashes
    store(new / "verified.json", dict(folds=len(first), policy_folds=len(old),
          published_seal=digest(EVIDENCE / "prediction-lock.json"),
          replay_seal=digest(evidence / "prediction-lock.json"), probability_hashes_match=True,
          sensitivity_probability_hashes_match=True, sensitivity_policy_folds=len(old_sensitivity)))
    print(f"Reproduced all {len(old)} primary and {len(old_sensitivity)} sensitivity policy runs.")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        replay()
