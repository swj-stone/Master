"""Fetch the pinned PMLB tables and dataset-specific provenance into this checkout."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/context_study/prospective.json"
DATA = ROOT / "data/context_study/pmlb"
OUT = ROOT / "results/thesis_benchmark/prospective-pmlb-contexts-r1"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config():
    return json.loads(CONFIG.read_text(encoding="utf-8-sig"))


def download(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "MinPI-research"}), timeout=45) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(1+attempt)


def fetch(name):
    spec = config()
    base = f"https://raw.githubusercontent.com/EpistasisLab/pmlb/{spec['revision']}/datasets/{name}"
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename in (f"{name}.tsv.gz", "metadata.yaml"):
        dest = folder / filename
        url = f"{base}/{filename}"
        if filename.endswith(".gz"):
            pointer = dest.with_suffix(".lfs")
            if not pointer.exists():
                pointer.write_bytes(download(url))
            lines = pointer.read_text(encoding="ascii").splitlines()
            expected = lines[1].removeprefix("oid sha256:")
            size = int(lines[2].removeprefix("size "))
            url = url.replace("raw.githubusercontent.com/", "media.githubusercontent.com/media/")
        else:
            expected = None
        if not dest.exists() or expected is not None and digest(dest) != expected:
            content = download(url)
            if expected and (len(content)!=size or hashlib.sha256(content).hexdigest()!=expected):
                raise ValueError(f"PMLB LFS checksum mismatch for {name}")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(content)
            tmp.replace(dest)
        records[filename] = {"url": url, "sha256": digest(dest), "bytes": dest.stat().st_size}
    return name, records


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        sources = dict(pool.map(fetch, config()["datasets"]))
    text = json.dumps({"protocol_sha256": digest(CONFIG), "sources": sources}, indent=2)
    path = OUT / "sources.json"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Frozen acquisition manifest differs")
    path.write_text(text, encoding="utf-8")
    print(f"Verified {len(sources)} pinned tables and dataset metadata files")


if __name__ == "__main__":
    main()

