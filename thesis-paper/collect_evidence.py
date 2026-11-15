"""Freeze compact paper data from reconciled study/replication runs and the historical grid."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from zipfile import ZIP_DEFLATED, ZipFile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDIES = ROOT / "results/thesis_benchmark"
ORIGINS = {"primary": "context-vote-20261115", "controls": "context-vote-controls-20261115-final",
           "historical": "baselines-eight-preprocessed-five-seeds"}
WIDTH_STUDY = "context-vote-width-20261115"
SPEAKER_STUDY = "context-vote-isolet-official-20261115"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Also package the effective run records")
    parser.add_argument("--with-width", action="store_true", help="Include all nine original-width stress-test outcomes")
    parser.add_argument("--with-speakers", action="store_true", help="Include the four official ISOLET holdout arms")
    args = parser.parse_args()
    evidence = HERE / "evidence"
    sources = {name: STUDIES / folder for name,folder in ORIGINS.items()}
    receipts = []
    tables = {}
    files_for_zip = set()
    groups = ["primary", "controls"]
    if args.with_width:
        sources["width"] = STUDIES / WIDTH_STUDY
        groups.append("width")
    if args.with_speakers:
        sources["speakers"] = STUDIES / SPEAKER_STUDY
        groups.append("speakers")
    for group in groups:
        parent = sources[group]
        report = parent / ("per-seed.csv" if group in {"width", "speakers"} else "per-seed-effective.csv")
        with report.open(encoding="utf-8-sig",newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != {"primary":96,"controls":90,"width":9,"speakers":4}[group]:
            raise ValueError(f"Incomplete {group} collection")
        for row in rows:
            folder = (parent / row["run_folder"]).resolve()
            if not folder.is_relative_to(parent):
                raise ValueError(f"Invalid effective row: {row}")
            if row["status"] != "completed":
                if group != "width":
                    raise ValueError(f"Unfinished main comparison: {row}")
                row["features"] = ""
                files_for_zip.add(folder / "watchdog.json")
                continue
            result = json.loads((folder / "result.json").read_bytes())
            frozen = json.loads((folder / "frozen.json").read_bytes())
            timer = json.loads((folder / "watchdog.json").read_bytes())
            if (timer["result"] != result or timer["status"] != "completed"
                    or not 0 < timer["elapsed_s"] < 3600 or frozen["selected"] != result["selected"]
                    or len(frozen["selected"]) != int(row["k"])):
                raise ValueError(f"Inconsistent source files in {folder}")
            row["features"] = json.dumps(frozen["selected"], separators=(",",":"))
            if group == "primary":
                diagnostics = json.loads((folder / "estimator.json").read_bytes())
                row["witness_predictions"] = diagnostics["witness_predictions"]
                row["accepted_proposals"] = sum(t["accepted"] for t in diagnostics["trace"])
            for name in ["arm.json","estimator.json","frozen.json","result.json","test_access.json",
                         "worker-completion.json","predictions.npz","watchdog.json","fits.jsonl"]:
                path = folder / name
                if path.exists():
                    files_for_zip.add(path)
        tables[group] = rows
        for f in [report, parent/"effective.json", parent/"protocol.json", parent/"run.log"]:
            if f.is_file():
                files_for_zip.add(f)
                receipts.append({"path":f.relative_to(ROOT).as_posix(),"sha256":sha(f)})
    evidence.mkdir(exist_ok=True)
    for group,rows in tables.items():
        with (evidence / (group+".csv")).open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    historical = sources["historical"] / "per-seed.csv"
    copyfile(historical,evidence / "historical.csv")
    transfer = HERE / "transfer"
    secondary = [transfer / "factorial-design.md", transfer / "factorial.csv"] if args.with_width else []
    for path in [historical, sources["historical"]/"protocol.json", transfer/"protocol.json",
                 transfer/"per-seed.csv", transfer/"run.py", *secondary]:
        receipts.append({"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path)})
    for group in ["primary","controls"]:
        copyfile(sources[group]/"effective.json",evidence/(group+"-timing.json"))
    manifest = {"studies":{group: path.name for group, path in sources.items()},"sources":receipts}
    if args.archive:
        target = args.archive.resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError("The evidence archive must be inside the thesis checkout")
        with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=5) as bundle:
            for path in sorted(files_for_zip):
                bundle.write(path, path.relative_to(ROOT).as_posix())
        manifest["archive"] = {"path":target.relative_to(ROOT).as_posix(),"sha256":sha(target),
                               "members":len(files_for_zip)}
    (evidence/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
