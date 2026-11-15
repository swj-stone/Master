"""Export and independently check sparse local-deletion certificates from frozen MinPI runs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PAPER = Path(__file__).resolve().parent
DEFAULT_RUN = PAPER.parent / "results/thesis_benchmark/context-vote-20261115"
DEFAULT_OUTPUT = PAPER / "evidence/certificates.json"
METRICS = {("accuracy","auc"), ("accuracy","ap","auc"), ("mae","r2")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(metric, candidate, reference):
    require(math.isfinite(candidate) and math.isfinite(reference), "Nonfinite score")
    loss = candidate - reference if metric == "mae" else reference - candidate
    return loss <= (.05 if metric == "ap" else .03)*abs(reference) + 1e-12


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def score_map(diag):
    return {frozenset(entry["mask"]):entry["scores"] for entry in diag["mask_scores"]}


def verify(cert, *, source=None):
    """Verify the empirical statement conditional on the supplied scores; source also checks provenance."""
    anchor = frozenset(cert["anchor"])
    final = frozenset(cert["selected"])
    metrics = tuple(cert["metrics"])
    require(0 < len(final) == len(cert["selected"]) and final <= anchor, "Invalid final support")
    require(len(anchor) == len(cert["anchor"]) and metrics in METRICS, "Invalid anchor or metric set")
    tails = [frozenset(e) for e in cert["extensions"]]
    require(bool(tails) and not tails[0] and len(tails)==len(set(tails))
            and all(not anchor & e for e in tails), "Invalid contexts")
    lookup = None
    if source is not None:
        require(digest(source/"estimator.json")==cert["estimator_sha256"], "Estimator hash mismatch")
        require(digest(source/"frozen.json")==cert["frozen_sha256"], "Frozen decision hash mismatch")
        diag=json.loads((source/"estimator.json").read_bytes())
        frozen=json.loads((source/"frozen.json").read_bytes())
        require(diag["arm"]=="min" and list(diag["anchor_order"])==cert["anchor"]
                and [frozenset(e) for e in diag["context_extensions"]]==tails
                and set(frozen["selected"])==final and frozen["dataset"]==cert["dataset"]
                and frozen["seed"]==cert["seed"], "Witness sources disagree")
        lookup=score_map(diag)

    def check_panels(entry, support, positive):
        ctx=entry["context"]
        require(type(ctx) is int and 0<=ctx<len(tails),"Unrecognized context")
        panels=entry["panels"]
        require(len(panels)==(3 if positive else 2),"Wrong majority witness size")
        indices=[p["panel"] for p in panels]
        require(len(set(indices))==len(indices) and all(type(p) is int and 0<=p<4 for p in indices),
                "Repeated or invalid panel")
        for panel in panels:
            scores=panel["scores"]
            require((set(scores)==set(metrics) if positive else len(scores)==1 and set(scores)<=set(metrics)),
                    "Incomplete metric witness")
            for metric, pair in scores.items():
                require(len(pair)==2,"Expected candidate and reference values")
                candidate,reference=pair
                require(within(metric,candidate,reference)==positive, "Claimed ballot contradicts scores")
                if lookup is not None:
                    e=tails[ctx]
                    i=panel["panel"]
                    require(candidate==lookup[support|e][i][metric] and
                            reference==lookup[anchor|e][i][metric], "Scores differ from the retained oracle")

    pos=cert["feasibility"]
    if final==anchor:
        require(pos==[],"Anchor feasibility follows by identity")
    else:
        require(sorted(p["context"] for p in pos)==list(range(len(tails))),"Missing acceptance context")
        for entry in pos:
            check_panels(entry,final,True)
    neg=cert["deletions"]
    expected=final if len(final)>1 else set()
    require(len(neg)==len(expected) and {p["remove"] for p in neg}==expected,"Incomplete terminal support check")
    for entry in neg:
        check_panels(entry, final-{entry["remove"]}, False)


def export(folder):
    diag=json.loads((folder/"estimator.json").read_bytes())
    frozen=json.loads((folder/"frozen.json").read_bytes())
    order=diag["anchor_order"]
    anchor=frozenset(order)
    selected=frozen["selected"]
    final=frozenset(selected)
    tails=[frozenset(t) for t in diag["context_extensions"]]
    metrics=sorted(diag["baseline"])
    lookup=score_map(diag)
    def evidence(s, idx, positive):
        tail=tails[idx]
        if (s|tail) not in lookup or (anchor|tail) not in lookup:
            return []
        panels=[]
        pairs=zip(lookup[s|tail],lookup[anchor|tail],strict=True)
        for i,(a,b) in enumerate(pairs):
            failed=[m for m in metrics if not within(m,a[m],b[m])]
            if positive == (not failed):
                keys=metrics if positive else failed[:1]
                panels.append({"panel":i,"scores":{m:[a[m],b[m]] for m in keys}})
        return panels[:3 if positive else 2]
    pos=[] if final==anchor else [
        {"context":i,"panels":evidence(final,i,True)} for i in range(len(tails))]
    neg=[]
    if len(final)>1:
        for column in selected:
            for i in range(len(tails)):
                panels=evidence(final-{column},i,False)
                if len(panels)==2:
                    neg.append({"remove":column,"context":i,"panels":panels})
                    break
    cert=dict(dataset=frozen["dataset"],seed=frozen["seed"],anchor=order,selected=selected,metrics=metrics,
              extensions=[sorted(t) for t in tails],feasibility=pos,deletions=neg,
              estimator_sha256=digest(folder/"estimator.json"),frozen_sha256=digest(folder/"frozen.json"))
    verify(cert,source=folder)
    return cert


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--file",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--sources",action="store_true",help="Also check hashes and scores against the archived run")
    args=parser.parse_args()
    if args.export:
        with (args.run/"per-seed-effective.csv").open(encoding="utf-8-sig", newline="") as f:
            rows=[r for r in csv.DictReader(f) if r["arm"]=="min"]
        certs=[]
        for row in rows:
            cert=export(args.run/row["run_folder"])
            cert["source"]=row["run_folder"]
            certs.append(cert)
        require(len(certs)==30 and len({(c["dataset"],c["seed"]) for c in certs})==30,"Incomplete campaign")
        args.file.parent.mkdir(exist_ok=True,parents=True)
        args.file.write_text(json.dumps(certs,separators=(",",":"))+"\n",encoding="utf-8")
    certificates=json.loads(args.file.read_bytes())
    for c in certificates:
        verify(c,source=args.run/c["source"] if args.sources else None)
    print(f"Verified {len(certificates)} final supports and {sum(len(c['deletions']) for c in certificates)} singleton vetoes")


if __name__=="__main__":
    main()
