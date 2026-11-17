"""Pinned UCI inputs and source-specific group partitions for fixed witness transfer."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import shutil
import subprocess
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.minpi_compression.sources import DATA


SOURCES = {
    "magic": (159,"magic+gamma+telescope","252e0a78333c108d0ea54537b61154aa781e95a15675e65ddf9b76993617191d"),
    "musk1": (74,"musk+version+1","84fd967a312cfe1d48d6f5a2becdda68f122767c0d51a3246cf699ff92a66031"),
    "wall": (194,"wall+following+robot+navigation+data","7c1a3838be678585570fa7fa675217ee626b3aa277bb187dde627418b5e95ed2"),
    "estate": (477,"real+estate+valuation+data+set","aa437bdac3ca23200258a0a58d251c0af90651aeaf7c7a07aaa714f0f9f25e2c"),
    "hardware": (29,"computer+hardware","5043f19d06e6dbfd4b9640596c92eb26d6fdecabeefab98437fb07a27d748b70"),
    "fires": (162,"forest+fires","abd40d6142b5e30fea73ff7292f91dcdd8b64bf3b3f36e20f96c37f5d14f06fb"),
}


def fetch():
    DATA.mkdir(exist_ok=True, parents=True)
    for name,(number,slug,expected) in SOURCES.items():
        path = DATA / f"{name}.zip"
        if not path.exists():
            with urlopen(f"https://archive.ics.uci.edu/static/public/{number}/{slug}.zip", timeout=60) as remote:
                raw = remote.read()
            if sha256(raw).hexdigest()!=expected:
                raise ValueError(f"Archive changed for {name}")
            path.write_bytes(raw)
        x, _, domain, group = load(name)
        print(name, domain, "shape", x.shape, "groups",len(np.unique(group)),expected)


@lru_cache
def load(name):
    raw = (DATA/f"{name}.zip").read_bytes()
    assert sha256(raw).hexdigest()==SOURCES[name][2]
    group=None
    task="classification" if name in ("magic","musk1","wall") else "regression"
    with ZipFile(BytesIO(raw)) as z:
        if name == "estate":
            with ZipFile(BytesIO(z.read("Real estate valuation data set.xlsx"))) as book:
                sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
                values = [[c.findtext("{*}v") for c in r.findall("{*}c")] for r in sheet.findall(".//{*}row")[1:]]
            frame=pd.DataFrame(values,dtype=float)
            x,y=frame.iloc[:,1:7].to_numpy(float),frame.iloc[:,7].to_numpy(float)
            group=x[:,0]
        elif name == "musk1":
            gzip = shutil.which("gzip")
            if gzip is None:
                raise RuntimeError("gzip is required to decode UCI's original clean1.data.Z")
            plain=subprocess.run([gzip,"-dc"], input=z.read("clean1.data.Z"),capture_output=True,check=True).stdout
            frame=pd.read_csv(BytesIO(plain),header=None)
            assert frame.shape==(476,169)
            group=frame.iloc[:,0].to_numpy()
            x,y=frame.iloc[:,2:102].to_numpy(float),frame.iloc[:,-1].to_numpy(int)
        elif name == "hardware":
            frame=pd.read_csv(z.open("machine.data"),header=None)
            group=frame.iloc[:,0].to_numpy()
            x,y=frame.iloc[:,2:8].to_numpy(float),frame.iloc[:,8].to_numpy(float)
        elif name == "fires":
            frame=pd.read_csv(z.open("forestfires.csv"))
            y=np.log1p(frame.pop("area").to_numpy(float))
            for field,levels in [("month","jan feb mar apr may jun jul aug sep oct nov dec"),
                                 ("day","mon tue wed thu fri sat sun")]:
                names=levels.split()
                phase=frame.pop(field).map({s:j for j,s in enumerate(names)}).to_numpy()*2*np.pi/len(names)
                frame[field+"_sin"]=np.sin(phase)
                frame[field+"_cos"]=np.cos(phase)
            x=frame.to_numpy(float)
        else:
            member = "magic04.data" if name=="magic" else "sensor_readings_24.data"
            frame=pd.read_csv(z.open(member),header=None)
            x,y=frame.iloc[:,:-1].to_numpy(float),frame.iloc[:,-1].to_numpy()
    if task=="classification":
        _,y=np.unique(y,return_inverse=True)
    if group is None:
        _,group=np.unique(x,axis=0,return_inverse=True)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    return x,y,task,np.asarray(group)


def split(name,seed):
    x,y,task,group=load(name)
    ids,rep,inverse=np.unique(group,return_index=True,return_inverse=True)
    if name=="estate":
        assert len(ids)==12
        pools=dict(fit=range(5),vote=range(5,7),width=range(7,9),test=range(9,12))
    else:
        def partition(pool, proportion, salt):
            labels=y[rep[pool]] if task=="classification" else None
            return train_test_split(pool,test_size=proportion,random_state=seed+salt,stratify=labels)
        dev,test=partition(np.arange(len(ids)),.2,0)
        base,width=partition(dev,.2,101)
        fit,vote=partition(base,.25,201)
        pools=dict(fit=fit,vote=vote,width=width,test=test)
    idx={k:np.flatnonzero(np.isin(inverse,a)) for k,a in pools.items()}
    assert np.array_equal(np.sort(np.concatenate(list(idx.values()))),np.arange(len(x)))
    return x,y,task,idx


if __name__=="__main__":
    fetch()
