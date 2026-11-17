"""Offline score-only replay from released predictions and aligned evaluation labels."""
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parent


def metrics(y, prediction):
    if prediction.ndim == 2:
        classes = [int(i) for i in np.unique(y) if np.any(y != i)]
        auc = np.mean([roc_auc_score(y == i, prediction[:, i]) for i in classes])
        return dict(auc=auc, accuracy=np.mean(y == prediction.argmax(axis=1)))
    return dict(r2=r2_score(y, prediction), mae=mean_absolute_error(y, prediction))


def read_job(archive, key, locks, forecast_field):
    receipt = archive.read(key + "/selection.json")
    assert sha256(receipt).hexdigest() == locks[key]
    rec = json.loads(receipt)
    forecasts = archive.read(key + "/forecasts.npz")
    pin = rec["files"]["forecasts.npz"] if forecast_field == "files" else rec[forecast_field]
    assert sha256(forecasts).hexdigest() == pin
    return rec, np.load(BytesIO(forecasts), allow_pickle=False)


def primary(labels):
    rows = []
    with ZipFile(ROOT/"seven.zip") as seven, ZipFile(ROOT/"wide.zip") as wide, ZipFile(ROOT/"reachable.zip") as reach:
        for panel, packed, expected in (("seven",seven,35), ("wide",wide,20)):
            keys = json.loads(packed.read("lock.json"))["receipts"]
            assert len(keys) == expected
            for key in keys:
                rec, forecasts = read_job(packed, key, keys, "files" if panel=="seven" else "forecast")
                outer = rec["idx" if panel=="seven" else "splits"]["test"]
                np.testing.assert_array_equal(outer, labels[panel+"/"+key+"/rows"])
                y = labels[panel+"/"+key+"/y"]
                if panel=="seven":
                    widths = {m:len(rec["records"][m]["support"]) for m in ("full","mdi_min","mdi_plus_min_min")}
                else:
                    digests = json.loads(reach.read("lock.json"))["receipts"]
                    _, canonical = read_job(reach, key, digests, "forecast")
                    base = json.loads(reach.read(key+"/selection.json"))
                    assert base["base_digest"]==keys[key] and base["outer"]==outer
                    widths = {}
                    with canonical:
                        for orig, arm in (("full","full"),("mdi","mdi_original"),("portfolio","portfolio_min_original")):
                            np.testing.assert_allclose(forecasts[orig],canonical[arm],atol=1e-13,rtol=0)
                            widths[orig] = len(base["records"][arm]["canonical"])
                with forecasts:
                    for arm,k in widths.items():
                        rows.append(dict(panel=panel,job=key,arm=arm,k=k,**metrics(y,forecasts[arm])))
    return rows


def medical(labels):
    rows = []
    with ZipFile(ROOT/"field-cost.zip") as z:
        for receipt in sorted(n for n in z.namelist() if n.endswith(".json")):
            key = receipt[:-5]
            rec = json.loads(z.read(receipt))
            np.testing.assert_array_equal(rec["idx"]["test"],labels["medical/"+key+"/rows"])
            y = labels["medical/"+key+"/y"]
            with np.load(BytesIO(z.read(key+".npz")),allow_pickle=False) as forecasts:
                for arm,item in rec["arms"].items():
                    rows.append(dict(panel="medical",job=key,arm=arm,k=int(item["mask"]).bit_count(),
                                     fee=item["fee"],**metrics(y,forecasts[arm])))
    return rows


def brier(y, probs):
    actual = np.eye(probs.shape[1])[y]
    return np.square(probs-actual).sum(axis=1).mean()


def devices(labels):
    rows = []
    with ZipFile(ROOT/"wearable.zip") as z, ZipFile(ROOT/"wearable-refit.zip") as refits:
        refit_lock = json.loads(refits.read("refit-lock.json"))["receipts"]
        for name,n,p in (("mhealth",10,3),("dsads",8,5)):
            keys = json.loads(z.read("lock-"+name+".json"))["receipts"]
            assert len(keys)==n
            for key in keys:
                item, file = read_job(z,key,keys,"forecast_sha256")
                ids,y = labels["wearable/"+key+"/rows"],labels["wearable/"+key+"/y"]
                assert sha256(ids.astype("<i8").tobytes()).hexdigest()==item["rows_sha256"]
                observed = y>=0
                y = y[observed].astype(int)
                selected,refit = read_job(refits,key,refit_lock,"forecast")
                assert selected["initial"]==keys[key]
                with file as forecasts, refit:
                    full = (1<<p)-1
                    ref = {t:brier(y,forecasts[f"frozen_{t}"][observed]) for t in range(full+1)}
                    original = [(arm,mask,f"refit_{mask}_" if arm.startswith("refit") else "frozen_",
                                 forecasts,"wearable") for arm,mask in item["arms"].items()]
                    bounded = [(arm,v["mask"],f"m{v['mask']}_",refit,"wearable_refit")
                               for arm,v in selected["arms"].items()]
                    for arm,mask,prefix,archive,panel in original+bounded:
                        losses = {s:brier(y,archive[prefix+str(s)][observed])
                                  for s in range(full+1) if s&mask==s}
                        worst = max(np.mean([losses[t&mask]-ref[t] for t in range(full+1) if t.bit_count()==s])
                                    *p/(s*ref[0]) for s in range(1,p+1))
                        rows.append(dict(panel=panel,job=key,arm=arm,k=mask.bit_count(),worst_risk=worst,
                                         **metrics(y,archive[prefix+str(mask)][observed])))
    return rows


def main():
    for name,pin in json.loads((ROOT/"SHA256SUMS.json").read_bytes()).items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==pin,name
    with np.load(ROOT/"held-labels.npz",allow_pickle=False) as labels:
        table = pd.DataFrame(primary(labels)+medical(labels)+devices(labels))
    keys = ["panel","job","arm"]
    expected = pd.read_csv(ROOT/"published.csv").set_index(keys).sort_index()
    result = table.set_index(keys).sort_index()[expected.columns]
    pd.testing.assert_frame_equal(result,expected,check_dtype=False,rtol=1e-11,atol=1e-12)
    result.to_csv(ROOT/"recomputed.csv")
    source = result.reset_index()
    source["dataset"] = source.job.str.rsplit("-",n=1).str[0]
    source.groupby(["panel","dataset","arm"]).mean(numeric_only=True).to_csv(ROOT/"recomputed-source-means.csv")
    print(f"Matched {len(result)} held score/width rows including the wearable refit comparison.")


if __name__=="__main__":
    main()
