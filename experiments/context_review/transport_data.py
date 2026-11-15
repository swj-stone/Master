"""Download the locked UCI archives without scoring or splitting data."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen
from zipfile import ZipFile

from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/context_review/transport_external.json"
DATA = ROOT / "data/context_study/transport-uci"
RUN = ROOT / "results/context_review/transport-uci"
EVIDENCE = ROOT / "thesis-paper/evidence/transport"


def read(path):
    return json.loads(path.read_bytes())


def fetch(source):
    name, number, slug, _, _ = source
    target = DATA / f"{name}.zip"
    url = f"https://archive.ics.uci.edu/static/public/{number}/{slug}.zip"
    if not target.exists():
        req = Request(url, headers={"User-Agent": "Academic feature-selection replication"})
        for attempt in range(5):
            try:
                with urlopen(req, timeout=75) as response:
                    content = response.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                sleep(1 + attempt)
        with ZipFile(BytesIO(content)) as archive:
            assert archive.testzip() is None
        partial = target.with_suffix(".partial")
        partial.write_bytes(content)
        partial.replace(target)
    content = target.read_bytes()
    with ZipFile(BytesIO(content)) as archive:
        names = archive.namelist()
        print(name, names[:16], flush=True)
    return dict(dataset=name, uci_id=number, url=url, sha256=sha256(content).hexdigest(),
                bytes=len(content), members=names)


def acquire():
    DATA.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(fetch, read(SPEC)["sources"]))
    path = EVIDENCE / "sources.json"
    if path.exists() and read(path) != sources:
        raise ValueError("Published archive changed relative to source manifest")
    path.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.parse_args()
    acquire()
