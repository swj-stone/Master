"""Rescore sealed post-release device nominating methods from prepared numeric arrays."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

from experiments.minpi_rework import wearable_assets as assets,wearable_refit as exp


def validated(root,locks):
    for key,pin in locks.items():
        file = root/key/"selection.json"
        assert exp.data.digest(file)==pin
        other = root/key/"forecasts.npz"
        assert exp.data.digest(other)==json.loads(file.read_bytes())["forecast"]
        yield key+"/selection.json",file
        yield key+"/forecasts.npz",other


def pack():
    lock = exp.data.OUT/"refit-lock.json"
    members = [("refit-lock.json",lock),*validated(exp.RUN,json.loads(lock.read_bytes())["receipts"])]
    path = assets.ARCHIVE.with_name("minpi-wearable-refit-forecasts.zip")
    with ZipFile(path,"w") as z:
        for name,file in sorted(members):
            info = ZipInfo(name,date_time=(1980,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            z.writestr(info,file.read_bytes())
    (exp.data.OUT/"refit-archive-sha256.txt").write_text(f"{exp.data.digest(path)}  {path.name}\n",encoding="ascii")


@contextmanager
def refits(path):
    if not path:
        yield
        return
    pin = (exp.data.OUT/"refit-archive-sha256.txt").read_text().split()[0]
    assert exp.data.digest(path)==pin
    label = (exp.data.OUT/"refit-lock.json").read_bytes()
    locks = json.loads(label)["receipts"]
    old = exp.RUN
    try:
        with TemporaryDirectory() as tmp,ZipFile(path) as z:
            names = [key+"/"+f for key in locks for f in ("selection.json","forecasts.npz")]
            assert set(z.namelist())==set(names+["refit-lock.json"]) and len(z.namelist())==len(names)+1
            assert z.read("refit-lock.json")==label
            for name in names:
                dest = Path(tmp)/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(name))
            exp.RUN = Path(tmp)
            list(validated(exp.RUN,locks))
            yield
    finally:
        exp.RUN = old


if __name__=="__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--pack",action="store_true")
    cmd.add_argument("--prepared",type=Path)
    cmd.add_argument("--archive",type=Path,help="Original eighteen-fold wearable forecast ZIP")
    cmd.add_argument("--refit-archive",type=Path,help="Nominated refit forecast ZIP")
    cmd.add_argument("--output",type=Path)
    args = cmd.parse_args()
    if args.pack:
        pack()
    else:
        label = (exp.data.OUT/"refit-lock.json").read_bytes()
        with refits(args.refit_archive),assets.inputs(args.prepared,args.archive,args.output):
            if args.output:
                (args.output/"refit-lock.json").write_bytes(label)
            h = exp.score()
            h.to_csv(exp.data.OUT/"refit-held.csv",index=False)
            q = h.groupby(["dataset","method"]).agg(n=("k","size"), mean_devices=("k","mean"),
                fits=("fits","mean"),median_s=("runtime_s","median"),auc=("auc","mean"),
                acc=("acc","mean"),held_joint=("held_joint","sum"))
            q.to_csv(exp.data.OUT/"refit-summary.csv")
            print(q.to_string())
