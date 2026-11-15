"""Trace selected HAR engineered columns back to the sensors named by the supplier."""
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"results/context_review/external-har-locked-r1"


def sensors(name):
    lower = name.lower()
    acc = "acc" in lower or "gravity" in lower
    gyro = "gyro" in lower
    if not (acc or gyro):
        raise ValueError(f"Unrecognized signal name {name}")
    return acc, gyro


def main():
    raw = ROOT/"data/context_study/uci-har/uci-har-240.zip"
    with ZipFile(raw) as outer, ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as inner:
        feature_text = inner.read("UCI HAR Dataset/features.txt").decode("ascii")
    names = [row.split(maxsplit=1)[1] for row in feature_text.splitlines()]
    assert len(names)==561
    receipt = json.loads((OUT/"frozen.json").read_bytes())
    columns, arms = [], []
    for arm in receipt["arms"]:
        feat = json.loads((OUT/arm/"support.json").read_bytes())["selected"]
        rows = []
        for j in feat:
            acc, gyro = sensors(names[j])
            rows.append(dict(arm=arm, column=j, name=names[j], accelerometer=acc,
                             gyroscope=gyro, frequency_domain=names[j].startswith("f")))
        columns.extend(rows)
        arms.append(dict(arm=arm,k=len(feat),requires_accelerometer=any(r["accelerometer"] for r in rows),
                         requires_gyroscope=any(r["gyroscope"] for r in rows),
                         frequency_domain_columns=sum(r["frequency_domain"] for r in rows)))
    pd.DataFrame(columns).to_csv(OUT/"sensor-columns.csv",index=False)
    data = pd.DataFrame(arms)
    data.to_csv(OUT/"sensor-modalities.csv",index=False)
    print(data.to_string(index=False))


if __name__=="__main__":
    main()
