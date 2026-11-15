"""Download and hash-check the three original UCI challenge archives used for validation."""
import hashlib
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


SOURCES = {
    "arcene": (167, "c7a40e39edf23a9b744eb570592d6e917fd9a1976aaf20ad581e09afd47d5220"),
    "dexter": (168, "70400df292178d8d3627411a5631a4cb13fb6b19da4a200b9a206f25255cddcc"),
    "dorothea": (169, "e12e0401e0cf9a5eb7f5a43cf1a0b261be6663db6b441080e1cdd8368753732e"),
}
DIRECTORY = Path(__file__).resolve().parents[2] / "data/raw/nips2003"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, (dataset, expected) in SOURCES.items():
        path = DIRECTORY / f"{name}.zip"
        if not path.exists():
            temp = path.with_suffix(".part")
            url = f"https://archive.ics.uci.edu/static/public/{dataset}/{name}.zip"
            with urlopen(url, timeout=90) as src, temp.open("wb") as out:
                copyfileobj(src, out)
            if checksum(temp) != expected:
                raise ValueError(f"UCI archive changed for {name}")
            temp.replace(path)
        if checksum(path) != expected:
            raise ValueError(f"Existing data differ from the pinned archive for {name}")
        print(f"{name}: source checksum verified")


if __name__ == "__main__":
    main()
