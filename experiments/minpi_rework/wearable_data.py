"""Five-second windows, source-declared device maps and subject-disjoint MHEALTH/DSADS folds."""
import argparse
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from shutil import copyfileobj
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"data/minpi_rework"
OUT = ROOT/"thesis-paper/evidence/minpi_rework/wearable-groups"
PINS = Path(__file__).with_name("wearable_sources.json")
SOURCES = {
    "mhealth": ("319/mhealth+dataset", ("chest","left_ankle","right_wrist"), 10, 12),
    "dsads": ("256/daily+and+sports+activities", ("torso","right_arm","left_arm","right_leg","left_leg"), 8, 19),
}


def digest(file):
    with Path(file).open("rb") as stream:
        h = sha256()
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def archive(name):
    return DATA/f"wearable-{name}.zip"


def fetch(name):
    file = archive(name)
    if not file.exists():
        tmp = file.with_suffix(".part")
        with urlopen("https://archive.ics.uci.edu/static/public/"+SOURCES[name][0]+".zip",timeout=120) as src:
            with tmp.open("wb") as out:
                copyfileobj(src,out,1<<20)
        tmp.replace(file)
    if digest(file)!=json.loads(PINS.read_bytes())[name]:
        raise ValueError(f"Archive SHA mismatch for {name}")


def _features(blocks):
    return np.concatenate([blocks.mean(axis=-2),blocks.std(axis=-2)],axis=-1).astype(np.float32)


def mhealth(zipfile):
    records = []
    for f in zipfile.namelist():
        match = re.search(r"mhealth_subject(\d+)\.log$",f,re.I)
        if not match:
            continue
        values = pd.read_csv(zipfile.open(f),sep=r"\s+",header=None).to_numpy(dtype=float)
        assert values.shape[1]==24 and np.isfinite(values).all()
        n = len(values)//250
        grouped = values[:n*250].reshape(n,250,24)
        y = grouped[:,0,-1].astype(int)
        valid = (y>0)&(grouped[:,:,-1]==y[:,None]).all(axis=1)
        records.append((_features(grouped[:,:,:-1]),np.where(valid,y-1,-1),np.full(n,int(match[1]),int)))
    unit = np.tile(np.repeat(range(3),[5,9,9]),2)
    return records,unit


def dsads(zipfile):
    records = []
    for f in sorted(zipfile.namelist()):
        match = re.search(r"(?:^|/)a(\d+)/p(\d+)/s(\d+)\.txt$",f,re.I)
        if not match:
            continue
        block = np.loadtxt(zipfile.open(f),delimiter=",")
        assert block.shape==(125,45) and np.isfinite(block).all()
        records.append((_features(block)[None,:],np.array([int(match[1])-1]),np.array([int(match[2])])))
    assert len(records)==19*8*60
    return records,np.tile(np.repeat(range(5),9),2)


def prepare(name):
    file = archive(name)
    pin = json.loads(PINS.read_bytes())[name]
    if digest(file)!=pin:
        raise ValueError("Unverified archive")
    d = digest(__file__)[:16]
    dataset = DATA/f"wearable-{name}-{pin[:12]}-{d}.npz"
    if not dataset.exists():
        with ZipFile(file) as z:
            nested = [f for f in z.namelist() if f.lower().endswith(".zip")]
            parser = mhealth if name=="mhealth" else dsads
            if nested:
                assert len(nested)==1
                with ZipFile(BytesIO(z.read(nested[0]))) as content:
                    records,unit = parser(content)
            else:
                records,unit = parser(z)
        x,y,subjects = (np.concatenate(parts) for parts in zip(*records))
        assert x.shape==(len(y),len(unit)) and len(subjects)==len(y)
        assert len(np.unique(subjects))==SOURCES[name][2]
        np.testing.assert_array_equal(np.unique(y[y>=0]),np.arange(SOURCES[name][3]))
        np.savez_compressed(dataset,x=x,y=y,subject=subjects,unit=unit)
    return dataset


@lru_cache(maxsize=2)
def read(name):
    with np.load(prepare(name),allow_pickle=False) as z:
        return tuple(z[k] for k in ("x","y","subject","unit"))


def fold(name,pos):
    x,truth,subject,unit = read(name)
    ids = np.random.default_rng(42).permutation(np.unique(subject))
    rot = np.roll(ids,-pos)
    person = dict(test=rot[:1],vote=rot[1:2],width=rot[2:4],fit=rot[4:])
    idx = {key:np.flatnonzero(np.isin(subject,people)&((truth>=0) if key!="test" else True))
           for key,people in person.items()}
    y = truth.copy()
    y[idx["test"]] = -1
    assert len(np.unique(np.concatenate(list(person.values()))))==len(ids)
    return x,y,unit,idx,{k:v.tolist() for k,v in person.items()}


if __name__=="__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--fetch", action="store_true")
    args = cli.parse_args()
    for key in SOURCES:
        if args.fetch:
            fetch(key)
        x,y,subject,unit = read(key)
        print(key,x.shape, int((y>=0).sum()),np.bincount(unit),flush=True)
