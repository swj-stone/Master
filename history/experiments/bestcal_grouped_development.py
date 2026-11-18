"""Evaluate predeclared calibration argmax using the frozen c pilot arrays.

No timing claim is made: selection arrays are reused only for this diagnostic.
Any subsequent cost measurement must recompute the entire pilot calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiments import budget_minpi_deployments as deployment
from experiments.shared_grouped_development import panels
from experiments.stable_minpi_stress import ROOT, file_hash
from minppi_mofs.bestcal_grouped_minpi import best_calibrated_support


REFERENCE = ROOT/"results/minpi-study/shared-grouped-development-20261117c"


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for panel in panels():
        input_path = REFERENCE/f"{panel.name}-1-1129.npz"
        hashes[input_path.relative_to(ROOT).as_posix()] = file_hash(input_path)
        saved = {}
        with np.load(input_path) as prior:
            for method in ("shared", "representatives_auto", "F_auto"):
                prefix = method+"_info_"
                info = {field[len(prefix):]: prior[field] for field in prior.files if field.startswith(prefix)}
                cols, selected = best_calibrated_support(panel.X.shape[1], info)
                pred, decision = deployment.fit_predict(panel.name, panel.X, panel.y, panel.Xt,
                                                         cols, panel.task, panel.seed)
                score, _, _ = deployment.score_predictions(panel.name, panel.gold, pred,
                                                            panel.units, panel.task, decision)
                rows.append(dict(dataset=panel.name, method=method+"_bestcal", p=panel.X.shape[1],
                                 width=len(cols), score=score,
                                 calibration_score=selected["selected_calibration_score"],
                                 full_calibration_score=float(info["full_pilot_score"]),
                                 reduced=selected["reduced"]))
                saved[method+"_cols"], saved[method+"_pred"], saved[method+"_decision"] = cols, pred, decision
        np.savez_compressed(output/f"{panel.name}-1-1129.npz", **saved)
    files = [Path(__file__), ROOT/"src/minppi_mofs/bestcal_grouped_minpi.py"]
    (output/"protocol.json").write_text(json.dumps(dict(
        decision="Highest calibration BA among reduced candidates; exact tie shortest; still >=fullpilot-.02",
        stage="Only same two development cases; cached calibration is not timed",
        source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
        input_sha256=hashes), indent=2), encoding="utf8")
    pd.DataFrame(rows).to_csv(output/"cases.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
