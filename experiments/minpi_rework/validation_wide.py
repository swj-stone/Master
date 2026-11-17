"""Blind five-source, 34--96-input extension for fixed-width exact search."""
import argparse
import json
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.minpi_compression.benchmark import metrics
from experiments.minpi_rework.validation_panel import DATA, SEEDS, _canonical_groups


SOURCES = json.loads(Path(__file__).with_name("validation_wide_sources.json").read_bytes())
CLASSIFICATION = {"derma", "arrhy", "robotlp"}


def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (slug, digest) in SOURCES.items():
        path=DATA/f"wide-{name}.zip"
        if path.exists():
            if sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError(f"Unexpected bytes at {path}")
            continue
        request = Request(f"https://archive.ics.uci.edu/static/public/{slug}.zip",
                          headers={"User-Agent":"MinPI research replication"})
        with urlopen(request, timeout=90) as response:
            raw=response.read()
        if sha256(raw).hexdigest()!=digest:
            raise ValueError(f"Changed UCI source: {name}")
        tmp=path.with_suffix(".partial")
        tmp.write_bytes(raw)
        tmp.replace(path)


def _unique_to_groups(x, ids):
    """Keep declared groups intact by discarding vectors spanning multiple groups."""
    code=x.groupby(list(x.columns), sort=False, dropna=False).ngroup()
    counts=pd.Series(ids,index=x.index).groupby(code).transform("nunique")
    return counts.eq(1).to_numpy()


@lru_cache
def _data(name):
    path=DATA/f"wide-{name}.zip"
    if sha256(path.read_bytes()).hexdigest()!=SOURCES[name][1]:
        raise ValueError(f"Archive changed: {name}")
    with ZipFile(path) as archive:
        ids=None
        if name=="derma":
            f=pd.read_csv(archive.open("dermatology.data"),header=None,na_values="?")
            x=f.iloc[:,:34]
            y=f.iloc[:,34].eq(1).to_numpy(int)
        elif name=="arrhy":
            f=pd.read_csv(archive.open("arrhythmia.data"),header=None,na_values="?")
            f=f.loc[f.iloc[:,-1].between(1,15)].reset_index(drop=True)
            cols=[159+10*lead+wave for lead in range(12) for wave in (0,1,2,3,6,7)]
            x=f.iloc[:,cols]
            y=f.iloc[:,-1].ne(1).to_numpy(int)
        elif name=="robotlp":
            tokens=archive.read("lp1.data").decode("ascii").split()
            records=np.asarray(tokens).reshape(-1,91)
            x=pd.DataFrame(records[:,1:].astype(float))
            _,y=np.unique(records[:,0],return_inverse=True)
        elif name=="ctloc":
            columns=[f"value{i}" for i in range(0,384,4)]
            f=pd.read_csv(archive.open("slice_localization_data.csv"),
                          usecols=["patientId",*columns,"reference"])
            rows=[]
            for _, indices in f.groupby("patientId",sort=False).indices.items():
                rows.extend(indices[np.linspace(0,len(indices)-1,num=min(48,len(indices)),dtype=int)])
            f=f.iloc[rows].reset_index(drop=True)
            x=f[columns]
            ids=f.patientId.to_numpy()
            y=f.reference.to_numpy(float)
        elif name=="news":
            f=pd.read_csv(archive.open("OnlineNewsPopularity/OnlineNewsPopularity.csv"))
            f.columns=f.columns.str.strip()
            if f.url.duplicated().any():
                raise ValueError("Unexpected repeated article URLs")
            f["hash"]=[sha256(s.encode()).hexdigest() for s in f.url]
            f=f.sort_values(["timedelta","hash"]).groupby("timedelta").head(6).reset_index(drop=True)
            x=f.drop(columns=["timedelta","shares","url","hash",
                              *[c for c in f if c.startswith(("kw_","self_reference_"))]])
            ids=-f.timedelta.to_numpy()
            y=np.log1p(f.shares.to_numpy(float))
        else:
            raise KeyError(name)
    if ids is None:
        groups=_canonical_groups(x,None)
    else:
        keep=_unique_to_groups(x,ids)
        x,y,groups=x.loc[keep],y[keep],ids[keep]
    if x.shape[1] != {"derma":34,"arrhy":72,"robotlp":90,"ctloc":96,"news":46}[name]:
        raise ValueError(f"Unexpected feature schema in {name}")
    return x.to_numpy(float), y, "classification" if name in CLASSIFICATION else "regression", groups


def _partition(name, seed):
    x,y,task,groups=_data(name)
    identifiers,first,inv=np.unique(groups,return_index=True,return_inverse=True)
    keys=np.arange(len(identifiers))
    if name=="news":
        fit,vote,width,test=np.split(keys,[int(len(keys)*q) for q in (.48,.64,.8)])
    else:
        def part(pool,fraction,salt):
            labels=y[first[pool]] if task=="classification" else None
            return train_test_split(pool,test_size=fraction,random_state=seed+salt,stratify=labels)
        dev,test=part(keys,.20,0)
        joint,width=part(dev,.20,101)
        fit,vote=part(joint,.25,201)
    idx={key:np.flatnonzero(np.isin(inv,g)) for key,g in
         dict(fit=fit,vote=vote,width=width,test=test).items()}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(idx.values()))),np.arange(len(y)))
    if any(len(ids)==0 for ids in idx.values()):
        raise ValueError("Empty partition")
    if task=="classification" and any(len(np.unique(y[ids]))!=len(np.unique(y)) for ids in idx.values()):
        raise ValueError("Missing class in a partition")
    return x,y,task,idx


def split(name, seed):
    x,y,task,idx=_partition(name,seed)
    blinded=y.copy()
    blinded[idx["test"]]=-1 if task=="classification" else np.nan
    return x.copy(),blinded,task,idx


def score(name, seed, forecast_path, frozen_sha256):
    path=Path(forecast_path)
    if sha256(path.read_bytes()).hexdigest()!=frozen_sha256:
        raise ValueError(f"Forecast digest mismatch for {name}/{seed}")
    _,y,task,idx=_partition(name,seed)
    scale=max(float(np.std(y[np.r_[idx["fit"],idx["vote"],idx["width"]]])),1e-12)
    with np.load(path,allow_pickle=False) as forecasts:
        return {arm:metrics(y[idx["test"]],forecasts[arm],task,scale) for arm in forecasts.files}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fetch",action="store_true")
    if cli.parse_args().fetch:
        fetch()
    for name in SOURCES:
        for seed in SEEDS:
            x,y,task,idx=split(name,seed)
            hidden=y[idx["test"]]
            assert (hidden==-1).all() if task=="classification" else np.isnan(hidden).all()
        print(name,x.shape,"groups",len(np.unique(_data(name)[3])),flush=True)
