"""Acquire only the eight declared UCI archives. Extract the named predictors and target."""
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from time import sleep
from zipfile import ZipFile
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.datasets import load_diabetes


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "minpi_compression"
SOURCES = {
    "glass": (42, "glass+identification"),
    "raisin": (850, "raisin"),
    "wifi": (422, "wireless+indoor+localization"),
    "leaves": (241, "one+hundred+plant+species+leaves+data+set"),
    "airfoil": (291, "airfoil+self+noise"),
    "yacht": (243, "yacht+hydrodynamics"),
    "slump": (182, "concrete+slump+test"),
    "energy": (242, "energy+efficiency"),
}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (number, slug) in SOURCES.items():
        path = DATA / f"{name}.zip"
        if not path.exists():
            url = f"https://archive.ics.uci.edu/static/public/{number}/{slug}.zip"
            for attempt in range(6):
                try:
                    with urlopen(Request(url, headers={"User-Agent": "Academic MinPI replication"}), timeout=45) as handle:
                        content = handle.read()
                    break
                except OSError:
                    if attempt == 5:
                        raise
                    sleep(1+attempt)
            with ZipFile(BytesIO(content)) as archive:
                assert archive.testzip() is None
            part = path.with_suffix(".part")
            part.write_bytes(content)
            part.replace(path)
        with ZipFile(path) as archive:
            print(name, sha256(path.read_bytes()).hexdigest(), archive.namelist()[:20], flush=True)


def load(name):
    """Return raw numeric predictors, targets, task and split-group identifiers."""
    task = "classification"
    if name == "diabetes":
        x, y = load_diabetes(return_X_y=True, scaled=False)
        task = "regression"
    elif name in ("ionosphere", "vehicle", "sonar"):
        frame = pd.read_csv(ROOT / f"data/context_study/pmlb/{name}/{name}.tsv.gz", sep="\t")
        y = frame.pop("target").to_numpy()
        x = frame.to_numpy(float)
    else:
        with ZipFile(DATA / f"{name}.zip") as z:
            if name == "glass":
                frame = pd.read_csv(z.open("glass.data"), header=None).iloc[:, 1:]
            elif name == "wifi":
                frame = pd.read_csv(z.open("wifi_localization.txt"), sep=r"\s+", header=None)
            elif name == "leaves":
                frame = pd.read_csv(z.open("100 leaves plant species/data_Mar_64.txt"), header=None)
                frame = frame.iloc[:, [*range(1, 65), 0]]
            elif name == "raisin":
                with ZipFile(BytesIO(z.read("Raisin_Dataset.zip"))) as nested:
                    arff = nested.read("Raisin_Dataset/Raisin_Dataset.arff").decode("utf-8-sig")
                frame = pd.DataFrame(loadarff(StringIO(arff))[0])
            elif name == "energy":
                # The numeric worksheet has eight predictors, heating and cooling loads.
                with ZipFile(BytesIO(z.read("ENB2012_data.xlsx"))) as book:
                    sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
                rows = [[c.findtext("{*}v") for c in row.findall("{*}c")[:10]]
                        for row in sheet.findall(".//{*}row")[1:]]
                frame = pd.DataFrame([row[:9] for row in rows if len(row) == 10 and all(row)], dtype=float)
            elif name == "slump":
                frame = pd.read_csv(z.open("slump_test.data")).iloc[:, [*range(1, 8), -1]]
            else:
                member = {"airfoil": "airfoil_self_noise.dat", "yacht": "yacht_hydrodynamics.data"}[name]
                frame = pd.read_csv(z.open(member), sep=r"\s+", header=None)
        task = "regression" if name in {"energy", "slump", "airfoil", "yacht"} else "classification"
        x, y = frame.iloc[:, :-1].to_numpy(float), frame.iloc[:, -1].to_numpy()
    if task == "classification":
        _, y = np.unique(y, return_inverse=True)
    groupcols = {"yacht": slice(0, 5), "airfoil": [1, 2], "energy": slice(0, 5)}.get(name, slice(None))
    _, groups = np.unique(x[:, groupcols], axis=0, return_inverse=True)
    return x, y, task, groups


if __name__ == "__main__":
    fetch()
