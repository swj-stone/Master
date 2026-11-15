"""Independent verification from prediction arrays. Run using thesis-python.ps1."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, product
import hashlib
import json
from math import comb
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/context_review"
PRIMARY = ROOT / "results/thesis_benchmark/prospective-pmlb-contexts-r1"
DATA = ROOT / "data/context_study"
EVIDENCE = ROOT / "thesis-paper/evidence"


def read(path):
    return json.loads(path.read_bytes())


def check_digest(path, reference):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference


def scores(labels, probs):
    classes = probs.shape[1]
    assert set(labels) == set(range(classes))
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=2e-7)
    auc = []
    for target in [1] if classes == 2 else range(classes):
        positive = labels == target
        n = positive.sum()
        wins = rankdata(probs[:, target])[positive].sum() - n*(n+1)/2
        auc.append(wins / (n * (len(labels)-n)))
    return np.array([np.mean(auc), np.mean(labels == np.argmax(probs, axis=1))])


def passes(q, tolerance=.03):
    return np.all(q >= q[0] - tolerance * abs(q[0]) - 1e-12, axis=1)


def check_equal(result, reference):
    np.testing.assert_allclose(result, reference, rtol=0, atol=2e-14)


def fit(x, y, train, xval, cols, seed, rounds):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    xt = imp.fit_transform(x[train][:, cols])
    xv = imp.transform(xval[:, cols])
    model = XGBClassifier(n_estimators=rounds, max_depth=4, learning_rate=.1,
                          tree_method="hist", device="cpu", n_jobs=1, random_state=seed)
    return model.fit(xt, y[train]).predict_proba(xv)


def finite_stump():
    # Direct class-pair comparison. No assumption about which unmasked evaluation rows
    # take the D=1 split, or about an individual row's R under a hidden-D mask.
    tables = assignments = 0
    for n in range(2, 11):
        for npos in range(1, n):
            p = Fraction(npos, n)
            observed = defaultdict(set)
            for rr in product((0, 1), repeat=n):
                ones = sum(rr)
                # Enumerate the actual ranks/ties among positive and negative rows.
                pair = sum(2*(a>b) + (a==b) for a in rr[:npos] for b in rr[npos:])
                a = Fraction(pair, 2 * npos * (n-npos))
                c = Fraction(sum(rr[:npos])+n-npos-sum(rr[npos:]), n)
                observed[ones].add((a,c))
                assignments += 1
            for ones, points in observed.items():
                q = Fraction(ones, n)
                expected = set()
                for overlap in range(max(0, npos+ones-n), min(npos, ones)+1):
                    u = Fraction(overlap, n)
                    expected.add((Fraction(1,2)+(u-p*q)/(2*p*(1-p)), 1-p-q+2*u))
                assert points == expected
                tables += len(points)
    print("Finite stump", assignments, "assignments;", tables, "feasible tables")


def group_collisions():
    ledger = defaultdict(lambda: defaultdict(list))
    for archive in sorted((RUN/"rule-sensitivity").glob("*/fold-*/votes.npz")):
        task, fold = archive.parent.parent.name, archive.parent.name
        donor = RUN/"refit-target"/task/fold
        plan = read(donor/"selection.json")
        selection = read(PRIMARY/task/fold/"selection.json")
        rawlabels = pd.read_csv(DATA/"pmlb"/task/(task+".tsv.gz"), sep="\t", usecols=["target"])
        yy = rawlabels.target.map(dict(zip(selection["class_order"], range(len(selection["class_order"]))))).to_numpy()
        yy = yy[selection["test"]]
        failures = {}
        for target, path in [("xgb", donor/"predictions.npz"), ("logistic", archive.parent/"logistic-outer.npz")]:
            with np.load(path) as refits:
                np.testing.assert_array_equal(refits["row_id"], selection["test"])
                np.testing.assert_array_equal(refits["y"], yy)
                failures[target] = ~passes(np.array([scores(yy, refits[key]) for key in
                    ["anchor", *[f"s{i}" for i in range(len(plan["supports"]))]]]))[1:]
        with np.load(archive) as frozen:
            assert set(frozen["metrics"]) == {"auc","accuracy"}
            lookup = {frozenset(np.where(mask)[0]): values for mask, values in
                      zip(frozen["xgb_masks"], frozen["xgb_scores"], strict=True)}
        sep = list(map(frozenset, plan["contexts"]["separating"]))
        ext = sorted(set(frozenset(c) for contexts in plan["contexts"].values() for c in contexts),
                     key=lambda c: (len(c), sorted(c)))
        ref = np.array([lookup[frozenset(plan["anchor"])|c][:4] for c in sep])
        for i, cols in enumerate(plan["supports"]):
            member = frozenset(cols)
            reduced = np.array([lookup[member|c][:4] for c in sep])
            votes = np.all(reduced >= ref - .03*np.abs(ref) - 1e-12, axis=-1)
            assert np.array_equal(votes, plan["ballots"][i]["separating"])
            other = np.array([lookup[member|c] for c in ext])
            keys = {"width": (), "worst approvals": min(np.sum(votes,axis=1)),
                    "separating ballots": tuple(votes.flat), "separating metrics": tuple(reduced.flat),
                    "all recorded metrics": tuple(other.flat)}
            for nd in (12, 9, 6, 4, 3, 2):
                keys[f"rounded_{nd}"] = tuple(np.round(reduced,nd).flat)
            cohorts = ["union", *[key for key, members in plan["cohorts"].items() if cols in members]]
            for target, fail in failures.items():
                for cohort in cohorts:
                    for summary, signature in keys.items():
                        ledger[(cohort,target,summary)][(task,fold,len(cols),signature)].append(fail[i])
    claimed = pd.read_csv(ROOT/"thesis-paper/audit/refit-observation-collisions.csv").set_index(
        ["cohort","target","summary"])
    for key, buckets in ledger.items():
        mixed = {k:g for k,g in buckets.items() if any(g) and not all(g)}
        errors = sum(min(sum(g),len(g)-sum(g)) for g in mixed.values())
        weighted = sum(min(2*sum(g),len(g)-sum(g)) for g in mixed.values())
        result = [sum(map(len,buckets.values())), len(buckets), len(mixed), sum(map(len,mixed.values())),
                  len({k[0] for k in mixed}), errors, weighted]
        if key in claimed.index:
            np.testing.assert_array_equal(claimed.loc[key], result)
        elif key[:2] == ("union","xgb"):
            print("XGB",key[-1],"mixed groups, proposals, errors",result[2],result[3],errors)
    print("Collision rows confirmed",len(claimed))


def candidate_pool(order, width, source_id, fold):
    anchor = sorted(order)
    prefix = tuple(sorted(order[:width]))
    if comb(len(anchor), width) <= 8:
        pool = set(combinations(anchor, width))
    else:
        pool = {prefix}
        rng = np.random.default_rng([81027,1000+source_id,fold,width])
        while len(pool)<8:
            pool.add(tuple(sorted(rng.choice(anchor, size=width, replace=False))))
    return sorted(pool,key=lambda cols:(sum(order.index(c) for c in cols),cols))


def new_panel():
    seal = read(EVIDENCE/"transport/low-fidelity-lock.json")
    protocol = RUN/"low-fidelity/protocol.json"
    for name, digest in read(protocol).items():
        check_digest(ROOT/name,digest)
    result = []
    counts, bridge_counts, ties_8 = Counter(), Counter(), []
    support_rows = pd.read_csv(EVIDENCE/"low_fidelity/supports.csv").set_index(["dataset","fold","idx"])
    published = pd.read_csv(EVIDENCE/"low_fidelity/folds.csv").set_index(["dataset","fold","method"])
    for key, digest in sorted(seal.items()):
        small = RUN/"low-fidelity"/key
        original = RUN/"transport-uci"/key
        check_digest(small/"sealed.json",digest)
        rec = read(small/"sealed.json")
        check_digest(small/"predictions.npz", rec["predictions"])
        check_digest(protocol,rec["design"])
        check_digest(original/"layout.json", rec["layout"])
        saved = read(original/"selection.json")
        layout = read(original/"layout.json")
        check_digest(original/"inputs.npz", layout["inputs_sha256"])
        check_digest(original/"targets.npz", layout["targets_sha256"])
        check_digest(original/"predictions.npz", saved["predictions"])
        cols = [tuple(c) for c in rec["supports"]]
        task, fold = rec["dataset"], rec["fold"]
        order = saved["choices"]["gain"]["order"][:len(cols[0])]
        k = len(order)
        assert tuple(sorted(order)) == cols[0] and not rec["rare"]
        expected = [] if k==1 else candidate_pool(order,(k+1)//2,layout["id"],fold)
        assert cols[1:] == expected == [tuple(c["columns"]) for c in saved["pool"]]
        with np.load(original/"inputs.npz") as data, np.load(original/"targets.npz") as targets:
            x,y,future,train,prop,guard = [data[k] for k in ("x","y","x_outer","train","proposal","guard")]
            np.testing.assert_array_equal(np.sort(np.r_[train,prop,guard]), np.arange(len(x)))
            test_y = targets["y"]
            np.testing.assert_array_equal(targets["test_row"],data["test_row"])
        with np.load(small/"predictions.npz") as probs, np.load(original/"predictions.npz") as prim:
            gm = np.array([scores(y[guard],probs[f"g{i}"]) for i in range(len(cols))])
            om = np.array([scores(test_y,probs[f"o{i}"]) for i in range(len(cols))])
            check_equal(gm, [[s["auc"],s["accuracy"]] for s in rec["guard_metrics"]])
            if len(cols)>1:
                sm = np.array([scores(y[prop],fit(x,y,train,x[prop],c,20421+fold,8)) for c in cols])
                check_equal(sm,[[s["auc"],s["accuracy"]] for s in rec["proposal_scores_8"]])
                # First Gain-ordered maximum from normalized slack vs the small-model anchor.
                rel = (sm[1:]-sm[0]+.03*abs(sm[0])+1e-12)/np.maximum(.03*abs(sm[0]),1e-12)
                margins = rel.min(axis=1)
                nominee = 1+np.argmax(margins)
                if np.sum(abs(margins-margins.max()) < 1e-10) > 1:
                    ties_8.append((task,fold,int(np.sum(abs(margins-margins.max()) < 1e-10))))
                assert rec["fits_8"] == len(cols)
            else:
                nominee = 0
                assert rec["proposal_scores_8"] == [] and rec["fits_8"] == rec["seconds_8"] == 0
            assert nominee == rec["low_fidelity"]
            g, o = passes(gm), passes(om)
            counts.update({"singletons":int(k==1),"small_fits":rec["fits_8"],"folds":1,
                           "candidates":len(cols)-1,"outer_available":int(any(o[1:])),
                           "joint_available":int(any((g&o)[1:]))})
            bridge_scores = []
            for i,c in enumerate(cols):
                if i:
                    counts[f"support(g={int(g[i])},o={int(o[i])})"]+=1
                sr = support_rows.loc[(task,fold,i)]
                assert sr.columns == " ".join(map(str,c)) and (sr.guard_pass,sr.outer_pass)==(g[i],o[i])
                check_equal(om[i], [sr.auc,sr.accuracy])
                combined = np.concatenate((x[guard],future))
                prob = fit(x,y,np.sort(np.r_[train,prop]),combined,c,20421+fold,100)
                np.testing.assert_array_equal(prob[:len(guard)],probs[f"g{i}"])
                bridge_scores.append(scores(test_y,prob[len(guard):]))
                final_probs = fit(x,y,np.arange(len(x)),future,c,20421+fold,100)
                np.testing.assert_array_equal(final_probs,probs[f"o{i}"])
                counts["full_models_refit"]+=2
            b = passes(np.array(bridge_scores))
            for i in range(1,len(cols)):
                triple = int(g[i]),int(b[i]),int(o[i])
                bridge_counts[triple]+=1
            methods = {"small":int(nominee), **{m:cols.index(tuple(s["nominee"])) for m,s in saved["choices"].items()}}
            for method,i in methods.items():
                if method not in ("small","gain") and expected:
                    archived = [item[method] for item in saved["pool"]]
                    assert i == 1+np.argmax(archived)
                if method=="gain":
                    assert i==int(k>1)
                selected = i if g[i] else 0
                if method!="small":
                    np.testing.assert_array_equal(prim[method],probs[f"o{selected}"])
                d = (k-(k+1)//2)/k
                typ = ("singleton" if k==1 else "no_outer_pass" if not o[1:].any() else
                       "guard_excludes_all" if not (g&o)[1:].any() else
                       "realized" if selected and o[selected] else "missed_nomination")
                row = published.loc[(task,fold,method)]
                assert (row.nominee,row.selected,row.disposition) == (i,selected,typ)
                val = [d, d*any(o[1:]), d*any((o&g)[1:]), d*bool(selected and o[selected]),
                       not o[selected], bool(selected),*om[selected],rec["fits_8"] if method=="small" else 0]
                fields = ["nominal","outer_ceiling","joint_ceiling","u","breach","accepted",
                          "auc","accuracy","small_fits"]
                check_equal(val,row[fields].astype(float).to_numpy())
                result.append(dict(zip(["task","method",*fields],[task,method,*val])))
    calculated = pd.DataFrame(result).groupby(["method","task"])[fields].mean().groupby("method").mean()
    expected = pd.read_csv(EVIDENCE/"low_fidelity/summary.csv").set_index("method")
    check_equal(calculated,expected.loc[calculated.index,fields].to_numpy())
    print("New panel",dict(counts))
    print("UCI same-model outer bridge (guard, bridge, final)", dict(sorted(bridge_counts.items())))
    print("Eight-round top ties to 1e-10",ties_8)
    print(calculated[["u","breach","joint_ceiling","small_fits"]].to_string())
    contrasts = pd.read_csv(EVIDENCE/"low_fidelity/contrasts.csv").set_index(["reference","field"])
    by_task_method = pd.DataFrame(result).groupby(["method","task"])[fields].mean()
    for method in ("gain","vote","slack","point"):
        delta = by_task_method.loc["small"] - by_task_method.loc[method]
        draws=np.random.default_rng(33119).integers(len(delta),size=(30000,len(delta)))
        for field in ("u","breach","auc","accuracy"):
            x = delta[field].to_numpy()
            expected = [x.mean(),*np.percentile(x[draws].mean(axis=1),[2.5,97.5])]
            check_equal(contrasts.loc[(method,field),["delta","lo95","hi95"]].to_numpy(),expected)


def historical():
    folds = pd.read_csv(EVIDENCE/"opportunity/folds.csv").set_index(["dataset","fold","tolerance","method"])
    summaries = pd.read_csv(EVIDENCE/"opportunity/summary.csv").set_index(["tolerance","method","metric"])
    support_rows = pd.read_csv(EVIDENCE/"opportunity/supports.csv").set_index(["dataset","fold","idx"])
    counts, result = Counter(), []
    for path in sorted((RUN/"refit-opportunity").glob("*/fold-*/complete.json")):
        task,fold=path.parent.parent.name,int(path.parent.name.split("-")[1])
        rec=read(path)
        check_digest(path.parent/"probabilities.npz",rec["predictions"])
        source = RUN/"budgeted-guard"/task/path.parent.name
        plan = read(source/"shortlists.json")
        check_digest(source/"shortlists.json",rec["source"])
        order, cols = plan["order"],rec["supports"]
        k=len(cols[0])
        assert sorted(order[:k])==cols[0] and cols[1:]==plan["pools"].get(str((k+1)//2),[[]])[0]
        if task=="optdigits":
            with ZipFile(DATA/"uci-optdigits/uci80.zip") as archive:
                train_frame = pd.read_csv(archive.open("optdigits.tra"),header=None)
                future_frame = pd.read_csv(archive.open("optdigits.tes"),header=None)
                yy=train_frame.pop(64).to_numpy()
                future_y=future_frame.pop(64).to_numpy()
        else:
            prev = read(PRIMARY/task/path.parent.name/"selection.json")
            raw = pd.read_csv(DATA/"pmlb"/task/(task+".tsv.gz"),sep="\t")
            yy = raw.pop("target").map(dict(zip(prev["class_order"],range(len(prev["class_order"]))))).to_numpy()
            future_y = yy[prev["test"]]
        with np.load(path.parent/"probabilities.npz") as pred:
            np.testing.assert_array_equal(yy[plan["guard"]],pred["guard_labels"])
            np.testing.assert_array_equal(future_y,pred["outer_labels"])
            gm = np.array([scores(yy[plan["guard"]],pred[f"g{i}"]) for i in range(len(cols))])
            om = np.array([scores(future_y,pred[f"o{i}"]) for i in range(len(cols))])
            if (task,fold) in {("ionosphere",0),("new_thyroid",0)}:
                x=raw.to_numpy(dtype=np.float32)
                for i,c in enumerate(cols):
                    for tag, train, ev in [("g",sorted(plan["train"]+plan["proposal"]),plan["guard"]),
                                          ("o",plan["dev"],prev["test"])]:
                        np.testing.assert_array_equal(fit(x,yy,train,x[ev],c,plan["seed"],100),pred[f"{tag}{i}"])
                        counts["full_models_refit"]+=1
        g = passes(gm)
        for i,c in enumerate(cols):
            row=support_rows.loc[(task,fold,i)]
            assert row.columns==" ".join(map(str,c)) and row.guard_pass==g[i]
            check_equal(row[["guard_auc","guard_accuracy","outer_auc","outer_accuracy"]].astype(float),
                        np.r_[gm[i],om[i]])
        for tolerance in (.01,.03,.05):
            o = passes(om,tolerance)
            d=(k-(k+1)//2)/k
            outer=any(o[1:])
            both=any((g&o)[1:])
            if tolerance==.03:
                counts.update({f"{task=='optdigits'}_folds":1,f"{task=='optdigits'}_candidates":len(cols)-1,
                               f"{task=='optdigits'}_singletons":int(k==1),f"{task=='optdigits'}_outer":int(outer),
                               f"{task=='optdigits'}_joint":int(both)})
                if task!="optdigits":
                    for gi,oi in zip(g[1:],o[1:]):
                        counts[f"support(g={int(gi)},o={int(oi)})"]+=1
            for method in ("fraction_gain","vote_gain"):
                nom=plan["shortlists"]["one_"+method]
                i=cols.index(nom[0]) if nom else 0
                assert i==rec["nominees"][method]
                success=bool(i and g[i] and o[i])
                val = [d,d*outer,d*both,d*(not outer),d*(outer and not both),d*(both and not success),d*success]
                fields=["nominal","outer_ceiling","joint_ceiling","no_outer_pass","guard_excludes_all",
                        "missed_nomination","realized"]
                row=folds.loc[(task,fold,tolerance,method)]
                check_equal(val,row[fields].astype(float))
                assert row.nominee==i and row.candidates==len(cols)-1
                if task!="optdigits":
                    result.append(dict(zip(["tol","method","task",*fields],[tolerance,method,task,*val])))
    frame = pd.DataFrame(result).groupby(["tol","method","task"])[fields].mean()
    for (tolerance,method),rows in frame.groupby(level=["tol","method"]):
        for field in fields:
            x=rows[field].to_numpy()
            draws=np.random.default_rng(33119).integers(len(x),size=(30000,len(x)))
            expected = [x.mean(),*np.percentile(x[draws].mean(axis=1),[2.5,97.5])]
            check_equal(summaries.loc[(tolerance,method,field),["mean","lo95","hi95"]],expected)
    print("Historical panel (False=PMLB, True=Optdigits)",dict(counts))


if __name__=="__main__":
    with threadpool_limits(limits=1):
        finite_stump()
        new_panel()
        historical()
        group_collisions()
