"""Reproduce the original wearable score from checksum-verified portable inputs."""
import argparse
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from experiments.minpi_rework import wearable_data as data,wearable_run as run


ARCHIVE = data.ROOT/"packages/minpi-wearable-forecasts.zip"
PREPARED = {"dsads":"c48a3434214e8adebc11695b3715fb201f6acf967027585c54d88ac91ab78d45",
            "mhealth":"3f6c28def172ed65af89e92470c9f62b6fdfe005caaa52c0bf24799ced56b7bf"}


def members():
    for name,(_,_,n,_) in data.SOURCES.items():
        yield f"lock-{name}.json", data.OUT/f"lock-{name}.json"
        for pos in range(n):
            for file in ("selection.json","forecasts.npz"):
                key = f"{name}-{pos}/{file}"
                yield key,run.RUN/key


def pack():
    from experiments.minpi_rework.wearable_score import lock_all
    lock_all()
    with ZipFile(ARCHIVE,"w") as z:
        for name,file in sorted(members()):
            member = ZipInfo(name,date_time=(1980,1,1,0,0,0))
            member.compress_type = ZIP_DEFLATED
            z.writestr(member,file.read_bytes())
    pin = data.digest(ARCHIVE)
    (data.OUT/"archive-sha256.txt").write_text(f"{pin}  {ARCHIVE.name}\n",encoding="ascii")


@contextmanager
def inputs(prepared=None,archive=None,output=None):
    """Temporarily override the raw reader / receipt root without altering registered modules."""
    old_read,old_run,old_out = data.read,run.RUN,data.OUT
    if prepared:
        @lru_cache(maxsize=2)
        def load(name):
            file = Path(prepared)/f"wearable-{name}-prepared.npz"
            if data.digest(file)!=PREPARED[name]:
                raise ValueError(f"Prepared wearable SHA mismatch for {name}")
            with np.load(file,allow_pickle=False) as z:
                return tuple(z[k] for k in ("x","y","subject","unit"))
    try:
        with TemporaryDirectory() as tmp:
            if archive:
                if data.digest(archive)!=(old_out/"archive-sha256.txt").read_text().split()[0]:
                    raise ValueError("Forecast ZIP does not match the sealed archive")
                expected = dict(members())
                with ZipFile(archive) as z:
                    if len(z.namelist())!=len(expected) or set(z.namelist())!=set(expected):
                        raise ValueError("Unexpected forecast members")
                    for name,local in expected.items():
                        raw = z.read(name)
                        if "/" not in name:
                            if raw!=local.read_bytes():
                                raise ValueError("Archive lock differs from committed lock")
                            continue
                        dest = Path(tmp)/name
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        dest.write_bytes(raw)
                run.RUN = Path(tmp)
            if prepared:
                data.read = load
            if output:
                output = Path(output)
                output.mkdir(parents=True,exist_ok=True)
                for name in data.SOURCES:
                    part = f"lock-{name}.json"
                    if (output/part).resolve()!=(old_out/part).resolve():
                        (output/part).write_bytes((old_out/part).read_bytes())
                data.OUT = output
            yield
    finally:
        data.read,run.RUN,data.OUT = old_read,old_run,old_out


def cli_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--prepared",type=Path,help="Directory with checksum-pinned wearable-{name}-prepared.npz files")
    cli.add_argument("--archive",type=Path,help="Sealed forecast ZIP")
    cli.add_argument("--output",type=Path,help="Destination for regenerated CSVs")
    return cli.parse_args()


if __name__=="__main__":
    pack()
