"""Read the four predeclared, SHA-pinned UCI archives and reproduce their group splits."""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from experiments.minpi_compression.benchmark import ROOT


MANIFEST=Path(__file__).with_name("wide_veto_sources.json")
SOURCES=json.loads(MANIFEST.read_bytes())
DATA=ROOT/"data/minpi_rework"


def fetch():
    for name,(code,slug,expected) in SOURCES.items():
        file=DATA/f"veto-{code}.zip"
        if not file.exists():
            with urlopen(f"https://archive.ics.uci.edu/static/public/{code}/{slug}.zip",timeout=90) as response:
                raw=response.read()
            if sha256(raw).hexdigest()!=expected:
                raise ValueError(f"Unexpected UCI archive for {name}")
            file.write_bytes(raw)
        if sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError(name)


def duplicate_groups(x):
    # All sources are native float32/int fingerprint measurements at acquisition precision.
    v=np.ascontiguousarray(x)
    return np.unique(v.view(np.dtype((np.void,v.itemsize*v.shape[1]))).ravel(),return_inverse=True)[1]


@lru_cache
def _data(name):
    code,_,digest=SOURCES[name]
    raw=(DATA/f"veto-{code}.zip").read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError(f"Changed UCI archive for {name}")
    with ZipFile(BytesIO(raw)) as z:
        if name=="adverts":
            data=pd.read_csv(z.open("ad.data"),header=None,na_values="?",skipinitialspace=True)
            y=data.pop(1558).eq("ad.").to_numpy(np.int32)
            x=data.to_numpy(np.float32)
            groups=duplicate_groups(x)
        elif name=="oraltox":
            data=pd.read_csv(z.open("qsar_oral_toxicity.csv"),sep=";",header=None)
            label=data.pop(1024)
            assert set(label)=={"negative","positive"}
            y=label.eq("positive").to_numpy(np.int32)
            x=data.to_numpy(np.uint8)
            groups=duplicate_groups(x)
            ids=np.random.default_rng(74155).permutation(np.unique(groups))[:4000]
            rows=np.isin(groups,ids)
            x,y,groups=x[rows],y[rows],groups[rows]
        elif name=="uji":
            train=pd.read_csv(z.open("UJIndoorLoc/trainingData.csv")).sort_values("TIMESTAMP",kind="stable")
            val=pd.read_csv(z.open("UJIndoorLoc/validationData.csv"))
            assert set(val.USERID)=={0} and 0 not in set(train.USERID)
            take=np.concatenate([ix[np.linspace(0,len(ix)-1,min(180,len(ix)),dtype=int)]
                                 for ix in train.groupby("USERID",sort=True).indices.values()])
            data=pd.concat((train.iloc[take],val),ignore_index=True)
            x=data.filter(regex=r"^WAP\d+$").to_numpy(np.float32)
            x[x==100]=-110
            y=data.LATITUDE.to_numpy(float)-4860000.
            groups=data.USERID.to_numpy(int)
            fingerprint=duplicate_groups(x)
            shared=pd.Series(groups).groupby(fingerprint).transform("nunique").to_numpy()>1
            keep=~shared | (groups==0)
            x,y,groups=x[keep],y[keep],groups[keep]
        else:
            book=ZipFile(BytesIO(z.read("Residential-Building-Data-Set.xlsx")))
            rows=ET.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//{*}row")[2:]
            values=[[float(c.findtext("{*}v")) for c in row.findall("{*}c") if c.find("{*}v") is not None] for row in rows]
            data=np.array([v for v in values if v])
            assert data.shape==(372,109)
            cols=[4,5,6,7,8,9,11]+[12+lag*19+f for lag in range(1,5) for f in range(19) if f!=10]
            x=data[:,cols]
            y=data[:,-2]
            groups=(data[:,2]*4+data[:,3]).astype(int)
    assert x.shape[1]=={"uji":520,"tehran":79,"oraltox":1024,"adverts":1558}[name]
    return x,y,groups


def partition(name,seed):
    x,y,g=_data(name)
    unique,first=np.unique(g,return_index=True)
    if name=="tehran":
        fit,vote,width,test=np.split(unique,[int(len(unique)*v) for v in (.48,.64,.80)])
    elif name=="uji":
        dev=np.random.default_rng(seed).permutation(unique[unique!=0])
        fit,vote,width=np.split(dev,[round(.6*len(dev)),round(.8*len(dev))])
        test=[0]
    else:
        def separate(keys,frac,salt):
            return train_test_split(keys,test_size=frac,random_state=seed+salt,stratify=y[first[keys]])
        dev,test=separate(np.arange(len(unique)),.2,0)
        remain,width=separate(dev,.2,35)
        fit,vote=separate(remain,.25,71)
        fit,vote,width,test=(unique[k] for k in (fit,vote,width,test))
    idx={key:np.flatnonzero(np.isin(g,ids)) for key,ids in
         dict(fit=fit,vote=vote,width=width,test=test).items()}
    gid=duplicate_groups(x)
    assert sum(len(np.unique(gid[ix])) for ix in idx.values())==len(np.unique(gid))
    task="classification" if name in ("adverts","oraltox") else "regression"
    if task=="classification":
        assert all(len(np.unique(y[ix]))==2 for ix in idx.values())
    return x,y,task,idx


def split(name,seed):
    x,y,task,idx=partition(name,seed)
    blind=y.copy()
    blind[idx["test"]]=-1 if task=="classification" else np.nan
    return x,blind,task,idx


if __name__=="__main__":
    import argparse
    cli=argparse.ArgumentParser()
    cli.add_argument("--fetch",action="store_true")
    if cli.parse_args().fetch:
        fetch()
    for name in SOURCES:
        for seed in range(42,47):
            x,_,_,idx=split(name,seed)
            print(name,seed,x.shape,[len(rows) for rows in idx.values()],flush=True)
