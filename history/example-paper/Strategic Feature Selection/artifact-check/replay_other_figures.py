"""Run the remaining eleven released Python figure notebooks offline from the pinned snapshot."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir, redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas import read_csv, testing
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "author-code"
NOTEBOOKS = SOURCE / "notebooks"
HASHES = {
    1: "1a3aad78508ce2ac1165064317e47250be01c405ac15e24cc761eba3c4009cc8",
    2: "eb1fb46ac83d81f1a54c097f57aa0c0869031fb59a24a8d36ee4079f7bde50b5",
    3: "bc1ee3f346e03f8891902db86a805b2b191d02e3451caeae7801810afed26496",
    5: "c850fab33ebb6d015650b799edff0ac1ee95ef515b2d00a1ebbdd1bd0719dd41",
    6: "5eed9bf0bd654a7f9bd6621e4294e46bd9d3d673af96a261481627312cd55e15",
    9: "eb0f3de16f472887526eec4d35892140c9fd48ad25b810348c6efa0b43f267a7",
    10: "a972b6ffcad7500d0f4dbdd266689468c057a12d0020e88576c6072b921656f4",
    11: "b16fd50ea12204a00c6d47ae9456b5380afbbe403bc37933e146155ee8efdeb5",
    12: "ad80c93c9668b6c6bb42af066d32d4c04a198a99e7c55f52f7bb62d353168d99",
    13: "ccb194a589ec905f588b7d84c9cb6daf15190cb00669c360f11ddbc45a687efc",
    14: "a33bff1ed0a3fe60c6210989dc7bf0249af74833902af42c90cb1a555d6ad2b5",
}


def replay(number):
    notebook = next(NOTEBOOKS.glob(f"figure{number}_*.ipynb"))
    raw = notebook.read_bytes()
    if sha256(raw).hexdigest() != HASHES[number]:
        raise ValueError(f"Unreviewed notebook: {notebook.name}")
    scratch = HERE.parents[2] / "tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        shutil.copytree(SOURCE / "data", root / "data")
        for archive_path in (root / "data").rglob("*.zip"):
            with ZipFile(archive_path) as archive:
                assert archive.testzip() is None
                assert all((root / f).resolve().is_relative_to(root) for f in archive.namelist())
        work = root / "notebooks"
        work.mkdir()
        context = {"__name__": "__main__", "display": None, "Image": None}
        omitted = []
        with chdir(work), threadpool_limits(limits=1), redirect_stdout(io.StringIO()), plt.rc_context():
            for i, cell in enumerate(json.loads(raw)["cells"]):
                if cell["cell_type"] != "code":
                    continue
                adapted = []
                for line in "".join(cell["source"]).splitlines():
                    stripped = line.strip()
                    if stripped in ("%load_ext autoreload", "%autoreload 2") or stripped.startswith(
                        ("from IPython.display import ", "display(")
                    ):
                        omitted.append({"cell": i, "line": stripped})
                        adapted.append(line[: len(line) - len(line.lstrip())] + "pass")
                    else:
                        adapted.append(line)
                exec(compile("\n".join(adapted), notebook.name, "exec"), context)
            plt.close("all")
        charts = sorted(f.relative_to(work) for f in work.rglob("*.pdf"))
        assert charts
        for chart in charts:
            assert (work / chart).read_bytes().startswith(b"%PDF")
            assert (NOTEBOOKS / chart).is_file(), chart
            assert (work / chart).with_suffix(".png").is_file()
        compared = []
        for f in work.glob("outputs/**/*.csv"):
            rel = f.relative_to(work)
            if "_unzipped" in rel.parts or not (NOTEBOOKS / rel).is_file():
                continue
            # Both sides are CSVs from the same notebook; allow ordinary floating arithmetic variation.
            testing.assert_frame_equal(read_csv(f), read_csv(NOTEBOOKS / rel), rtol=1e-6, atol=1e-8)
            compared.append(rel.as_posix())
        return {"notebook": notebook.name, "pdfs_regenerated": [f.as_posix() for f in charts],
                "matched_upstream_csvs": sorted(compared), "notebook_ui_lines_skipped": omitted}


if __name__ == "__main__":
    # The largest grids run in separate working directories, each with one numerical thread.
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    outcomes = {}
    with ProcessPoolExecutor(max_workers=3) as workers:
        jobs = {workers.submit(replay, n): n for n in [2, 3, 6, 1, 5, 9, 10, 11, 12, 13, 14]}
        for task in as_completed(jobs):
            n = jobs[task]
            try:
                outcomes[n] = task.result()
                print(f"Figure {n}: regenerated {len(outcomes[n]['pdfs_regenerated'])} PDF(s)", flush=True)
            except Exception as error:
                outcomes[n] = {"error": f"{type(error).__name__}: {error}"}
                print(f"Figure {n}: {outcomes[n]['error']}", flush=True)
    report = {"version": "arXiv:2606.18867v1 figure numbers", "figures": dict(sorted(outcomes.items()))}
    (HERE / "other-figures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if any("error" in result for result in outcomes.values()):
        raise SystemExit(1)

