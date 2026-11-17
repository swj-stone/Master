"""Package the prospective paper audit, including original forecasts and row identities."""
from hashlib import sha256
import os
from pathlib import Path
from shutil import copy2
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.anchored_vote import audit


def collect():
    root=audit.ROOT
    paths=set()
    for name in ("audit.py","select.py","run.py","protocol.md"):
        paths.add(root/"experiments/anchored_vote"/name)
    data=root/"experiments/minpi_replication/second_panel"
    paths.update(data/f for f in ("data.py","sources.json","protocol.md"))
    paths.update(audit.OUT/f for f in ("receipts.csv","seeds.csv","sources.csv"))
    paths.update([root/"src/minppi_mofs/split_vote.py",root/"experiments/split_vote/path.py",
                  root/"experiments/minpi_replication/direct_gain.py",root/"thesis-paper/formal/SplitVote.lean",
                  root/"thesis-paper/formal/AnchoredGuard.lean",root/"thesis-paper/formal/SplitVote.wl",
                  root/"thesis-paper/formal/split-vote-mathics.json"])
    for repeat in audit.REPEATS:
        direct=audit.RUN/f"prospective-r{repeat}"
        for s in audit.SEEDS:
            for n in audit.SOURCES:
                for m in audit.METHODS:
                    paths.update(direct/f"{n}-{s}"/m/f for f in ("receipt.json","forecast.npz"))
    return sorted(paths)


def main():
    audit.load_panel()
    dest=audit.ROOT/"packages/minpi-anchored-prospective.zip"
    temp=dest.with_suffix(".partial")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED,compresslevel=5) as bundle:
        checks=[]
        for path in collect():
            raw=path.read_bytes()
            name=path.relative_to(audit.ROOT).as_posix()
            bundle.writestr(name,raw)
            checks.append(sha256(raw).hexdigest()+"  "+name)
        bundle.writestr("SHA256SUMS","\n".join(checks)+"\n")
        bundle.writestr("README.txt","Extract into an empty folder. Use Python with NumPy, pandas, and scikit-learn.\n"
                        "From the extracted root download the pinned UCI sources by running:\n"
                        'python -c "from experiments.minpi_replication.second_panel.data import fetch; fetch()"\n'
                        "Then run: python -m experiments.anchored_vote.audit\n"
                        "This reconstructs every prospective table from saved forecasts without training.\n"
                        "To retrain the algorithms, use the full thesis checkout and its shared Python environment.\n")
    with ZipFile(temp) as check:
        if check.testzip() is not None:
            raise ValueError("Archive failed to verify")
        with TemporaryDirectory(prefix="anchored-audit-",dir=audit.ROOT/"tmp") as stage:
            check.extractall(stage)
            extracted=Path(stage)
            local_data=extracted/"data/minpi_replication_r3"
            local_data.mkdir(parents=True)
            for name in audit.SOURCES:
                copy2(audit.ROOT/"data/minpi_replication_r3"/(name+".zip"), local_data)
            env=os.environ.copy()
            env.pop("PYTHONPATH",None)
            subprocess.run([sys.executable,"-B","-m","experiments.anchored_vote.audit"],
                           cwd=extracted,env=env,check=True)
            for name in ("receipts.csv","seeds.csv","sources.csv"):
                if (extracted/audit.OUT.relative_to(audit.ROOT)/name).read_bytes()!=(audit.OUT/name).read_bytes():
                    raise ValueError(f"Extracted archive produced a different {name}")
    temp.replace(dest)
    print(dest,sha256(dest.read_bytes()).hexdigest(),dest.stat().st_size)


if __name__=="__main__":
    main()
