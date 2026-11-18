"""Replay the released notebooks for arXiv v1 Figures 4, 7 and 8 in a scratch directory."""

from contextlib import chdir, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "author-code"
NOTEBOOKS = SOURCE / "notebooks"
BENCHMARK = "lowdim_algorithm_benchmark_data.zip"
NOTEBOOK_SHA256 = {
    "figure4_top10_three_panel": "0f30854532419240e84a0bde8dfa0841decc2a459e4bbbd2d3e2a813a512c101",
    "figure7_lowdim_algorithm_design": "f15db778a3693e3734d53a6e2cecec553c8b481c749f9f0d33b2818cb80550a9",
    "figure8_lowdim_algorithm_design_heatmap": "74c3c51af7414c2e685ddab3727b211663efc5f44023c53070d2edd440691a5e",
}


def synthetic_results():
    with ZipFile(SOURCE / "data" / BENCHMARK) as archive:
        def read(suffix):
            with archive.open("lowdim_exact_oracle_benchmark_" + suffix + ".csv") as stream:
                return pd.read_csv(stream)

        data, summary, groups, search = (read(s) for s in ("data", "summary", "heatmap", "local_search_data"))

    unit = ["condition", "rep", "seed"]
    reference = data[data.method == "Oracle subset-ridge (grid)"].set_index(unit).mse
    assert len(reference) == 30 and reference.index.is_unique
    assert not data.duplicated(unit + ["method"]).any()
    assert (reference.groupby(level="condition").size() == 5).all()
    data = data.join(reference.rename("reference"), on=unit, validate="many_to_one")
    np.testing.assert_allclose(data.oracle_mse, data.reference, rtol=0, atol=1e-12)
    data["ratio"] = data.mse / data.reference

    by_method = {}
    for row in summary.itertuples():
        ratio = data[data.method == row.method].ratio
        assert len(ratio) == 30
        hits = int(np.isclose(ratio, 1.0, atol=1e-10, rtol=0).sum())
        assert hits == row.attained_oracle
        np.testing.assert_allclose([ratio.median(), ratio.max()], [row.median_oracle_ratio, row.max_oracle_ratio])
        by_method[row.method] = {"oracle_hits": hits, "median_ratio": ratio.median(), "max_ratio": ratio.max()}

    heatmap = groups.set_index("condition")
    rebuilt = data.pivot_table(index="condition", columns="method", values="ratio", aggfunc="median")
    np.testing.assert_allclose(rebuilt.loc[heatmap.index, heatmap.columns], heatmap, atol=1e-12, rtol=0)
    search = search.join(reference.rename("reference"), on=unit, validate="one_to_one")
    assert len(search) == len(reference)
    changes = {}
    for method in ("weighted", "greedy"):
        gain = (search[method + "_raw_mse"] - search[method + "_refined_mse"]) / search.reference
        assert (gain >= -1e-10).all()
        np.testing.assert_allclose(gain, search[method + "_refinement_gain_rel_oracle"], rtol=0, atol=1e-10)
        changes[method] = {"improved_cases": int((gain > 1e-10).sum()), "max_normalized_gain": gain.max()}
    return {"benchmark_instances": len(reference), "method_summary": by_method,
            "heatmap_cells_verified": int(heatmap.size), "local_search": changes}


def replay_notebooks():
    # The fixed, reviewed upstream notebook cells write relative to their current directory.
    # Copying input files into an empty directory also prevents use of cached author outputs.
    # A short path avoids the legacy Windows limit on the longer archived data filenames.
    scratch_parent = HERE.parents[2] / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=scratch_parent) as scratch:
        root = Path(scratch)
        shutil.copytree(SOURCE / "data", root / "data")
        for archive_path in (root / "data").rglob("*.zip"):
            with ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                assert all((root / name).resolve().is_relative_to(root) for name in archive.namelist())
        work = root / "notebooks"
        work.mkdir()
        with chdir(work), threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
            for name, expected_hash in NOTEBOOK_SHA256.items():
                notebook = NOTEBOOKS / (name + ".ipynb")
                raw = notebook.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == expected_hash, f"Unreviewed notebook: {name}"
                scope = {"__name__": "__main__"}
                for cell in json.loads(raw)["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    if source.lstrip().startswith("from IPython.display import Image, display"):
                        continue  # The last cell only displays the PNG that the preceding cells saved.
                    exec(compile(source, notebook.name, "exec"), scope)
                plt.close("all")

            outputs = Path("outputs")
            stats_file = Path("top10_three_panel_outputs/summary.json")
            healthcare = json.loads((outputs / stats_file).read_text())
            original = json.loads((NOTEBOOKS / "outputs" / stats_file).read_text())
            assert original.keys() == healthcare.keys()
            for key, expected in original.items():
                np.testing.assert_allclose(healthcare[key], expected, rtol=0, atol=1e-8)
            by_method = pd.read_csv(outputs / "top10_three_panel_outputs/data/method_comparison_alpha1_tau05_k25.csv")
            charts = sorted(path.name for path in outputs.glob("*/figures/*.pdf"))
            assert charts == ["lowdim_algorithm_design.pdf", "lowdim_algorithm_design_heatmap.pdf", "top10_three_panel.pdf"]
            return {"computed_figures": charts, "healthcare_summary_verified": healthcare,
                    "healthcare_method_risks_alpha_1": dict(zip(by_method.method, by_method.mse))}


if __name__ == "__main__":
    result = {"source": "Released notebooks, figure numbers from arXiv:2606.18867v1",
              "figure_7_8_scope": "Replayed plotting and verified archived synthetic results, not a new fit.",
              "figure_4_scope": "Recomputed HCC risk, support search, ridge grid and support frontier from supplied inputs.",
              "synthetic": synthetic_results(), "replay": replay_notebooks()}
    (HERE / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

