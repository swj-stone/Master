"""Fetch the four registered source archives and verify their content hashes."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
MANIFEST = HERE / "sources.json"

def fetch(name):
    record = json.loads(MANIFEST.read_text())[name]
    url = record["url"]
    path = RAW / (name + ".zip")
    expected = record["sha256"]
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        req = Request(url, headers={"User-Agent": "academic-replication/1"})
        with urlopen(req, timeout=90) as src, NamedTemporaryFile(dir=RAW, delete=False) as output:
            staged = Path(output.name)
            while chunk := src.read(1 << 20):
                output.write(chunk)
        try:
            with ZipFile(staged) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Broken source ZIP for {name}")
            if sha256(staged.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded bytes differ from registered archive for {name}")
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Unrecognized archive for {name}")
    return {"url": url, "sha256": digest}

if __name__ == "__main__":
    names = list(json.loads(MANIFEST.read_text()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = dict(zip(names, pool.map(fetch, names)))
    print(json.dumps(records, indent=2))
