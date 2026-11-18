"""Index individual historical evaluation configurations from their run manifests."""
import argparse
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("historical-panels.csv")
RUNS = (("original", "adaptive-69-20261117a", 69), ("extra", "adaptive-transfer-20261117a", 57))


def build_index():
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["panel", "dataset", "fold", "seed", "features", "methods_csv", "arrays"])
    for cohort, name, expected in RUNS:
        base = Path("results/minpi-study") / name
        details = base / "cases.csv"
        with (ROOT / details).open(encoding="utf-8", newline="") as source:
            rows = [r for r in csv.DictReader(source) if r["method"] == "full"]
        keys = {(r["dataset"], r["fold"], r["seed"]) for r in rows}
        if len(rows) != expected or len(keys) != expected:
            raise ValueError(f"{name} expected {expected} distinct full-model records")
        for row in sorted(rows, key=lambda r: (r["dataset"], int(r["fold"]), int(r["seed"]))):
            dataset, fold, seed = row["dataset"], row["fold"], row["seed"]
            arrays = base / f"{dataset}-{fold}-{seed}.npz"
            if not (ROOT / arrays).is_file():
                raise FileNotFoundError(arrays)
            writer.writerow((cohort, dataset, fold, seed, row["p"], details.as_posix(), arrays.as_posix()))
    return output.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the navigation CSV")
    args = parser.parse_args()
    index = build_index()
    if args.write:
        REGISTRY.write_bytes(index.encode("utf-8"))
    elif REGISTRY.read_text(encoding="utf-8") != index:
        raise SystemExit("Index differs from the archived configurations")
    print("Verified 69 original and 57 extra configurations and their local array paths.")
