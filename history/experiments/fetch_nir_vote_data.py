"""Fetch the seven frozen NIRBENCH partitions used in the minPI study."""
import csv
import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1] / "data/raw"
BASE = "https://raw.githubusercontent.com/DeepLightLab/NIRBENCH_DL_v0.1/main/"
MANIFEST_SHA = "32b7fe279f8743786460b21d75ee26d51fee9aa3759c3e0508996e4b50879cce"
TASKS = {1, 5, 13, 15, 19, 22, 30}


def download(url, target, expected):
    if target.is_file() and sha256(target.read_bytes()).hexdigest() == expected:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=90) as source, NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        digest = sha256()
        try:
            while block := source.read(1024 * 1024):
                digest.update(block)
                tmp.write(block)
        except BaseException:
            tmp.close()
            temporary.unlink(missing_ok=True)
            raise
    if digest.hexdigest() != expected:
        temporary.unlink()
        raise ValueError(f"Source SHA-256 mismatch: {target.name}")
    temporary.replace(target)
    print(target.name, target.parent.name, flush=True)


def main(extra_seasons=False):
    eligible = TASKS | ({16, 23, 24} if extra_seasons else set())
    manifest = ROOT / "nirbench-manifest.csv"
    download(BASE + "datasets/manifest.csv", manifest, MANIFEST_SHA)
    with manifest.open(encoding="utf8") as source:
        for row in csv.DictReader(source):
            name = row["dataset"]
            if int(name.split("-")[0]) in eligible:
                path = f"datasets/{name}/data_{row['split']}.csv"
                assert path == row["path"] and row["availability"] == "included"
                download(BASE + path, ROOT / "nirbench" / name / Path(path).name, row["sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-seasons", action="store_true")
    main(parser.parse_args().extra_seasons)
