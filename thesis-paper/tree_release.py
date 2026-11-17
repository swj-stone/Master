"""Build the MinPI tree manuscript and its offline replication archives."""
import argparse
import ast
import csv
import hashlib
from importlib.util import resolve_name
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import NamedTuple
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PACKAGES = ROOT / "packages"
EXPERIMENT = ROOT / "experiments/minpi_rework"
EVIDENCE = PAPER / "evidence/minpi_rework"


class Cohort(NamedTuple):
    archive: str
    group: str
    count: int
    receipt: str = "selection.json"
    forecast: str = "forecasts.npz"
    archive_hash: str = "forecast-sha256.txt"
    artifacts: str = "forecast"


COHORTS = (
    Cohort("minpi-quota-tree-forecasts.zip","quota-tree-confirmation",65),
    Cohort("minpi-scale-five-forecasts.zip","quota-scale-confirmation",25,"complete.json","all-forecasts.npz"),
    Cohort("minpi-predictive-seven-forecasts.zip","predictive-confirmation",35,artifacts="files"),
    Cohort("minpi-predictive-mc-seven-forecasts.zip","predictive-mc-confirmation",35,
           archive_hash="archive-sha256.txt",artifacts="hashes"),
    Cohort("minpi-wide-veto-forecasts.zip","wide-veto",20,archive_hash="archive-sha256.txt"),
    Cohort("minpi-reachable-forecasts.zip","predictive-reachable",55,archive_hash="archive-sha256.txt"),
)
TABLES = {
    "quota-tree-confirmation": ("scores.csv","rank-control-scores.csv","source-means.csv","source-contrasts.csv"),
    "quota-scale-confirmation": ("scores.csv","source-means.csv","source-contrasts.csv"),
    "stratified-tree": ("contrasts.csv","source-results.csv","subject-scores.csv"),
    "predictive-confirmation": ("scores.csv","paper-contrasts.csv","source-means.csv","independent-mask-audit.csv"),
    "predictive-mc-confirmation": ("repeats.csv","jobs.csv","contrasts.csv","released-scores.csv","released-contrasts.csv"),
    "predictive-prepared": ("source-means.csv","paper-contrasts.csv","refit-comparison.csv"),
    "predictive-lattice": ("scores.csv",),
    "observed-availability": ("scores.csv","paired.csv"),
    "wide-veto": ("timings.csv","scores.csv","source-scores.csv","paired-jobs.csv","contrasts.csv","native-audit.csv"),
    "predictive-refit": ("scores.csv","source-means.csv","contrasts.csv"),
    "predictive-polynomial": ("paired.csv","contrasts.csv","sources.csv","pipeline-paired.csv",
                              "production-paired.csv","mdi-paired.csv"),
    "predictive-reachable": ("scores.csv","source-means.csv","contrasts.csv","stability.csv"),
    "predictive-width": ("contrasts.csv",),
    "wide-veto-ablation": ("scores.csv","source-means.csv","contrasts.csv","width-ratios.csv"),
    "predictive-universe": ("scores.csv","source-means.csv","fixed-summary.csv","reselection.csv"),
    "field-cost": ("scores.csv","means.csv"),
    "wearable-groups": ("held.csv","acceptance-subsets.csv","clocks.csv","summary.csv","refit-held.csv","refit-summary.csv"),
    "predictive-veto-timing": (
        "contrasts.csv","source-contrasts.csv","split-medians.csv",
        "native-contrasts.csv","native-source-contrasts.csv","native-split-medians.csv",
    ),
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def checked(command, cwd, capture=False, **kwargs):
    try:
        return subprocess.run(command,cwd=cwd,check=True,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.STDOUT if capture else None,**kwargs)
    except subprocess.CalledProcessError as exc:
        if capture:
            print(exc.stdout.decode(errors="replace")[-3500:],file=sys.stderr)
        raise


def archive(path, members, manifest_name="SHA256SUMS.json"):
    """Sorted member names, fixed ZIP metadata and a manifest of the exact contents."""
    temporary = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True,exist_ok=True)
    hashes = {}
    try:
        with ZipFile(temporary,"w",compression=ZIP_DEFLATED,compresslevel=4) as z:
            for name, source in sorted(members.items()):
                info = ZipInfo(name,date_time=(1980,1,1,0,0,0))
                info.compress_type = ZIP_DEFLATED
                h = hashlib.sha256()
                with z.open(info,"w") as dest, source.open("rb") as src:
                    while chunk := src.read(1024**2):
                        h.update(chunk)
                        dest.write(chunk)
                hashes[name] = h.hexdigest()
            z.writestr(ZipInfo(manifest_name),json.dumps(hashes,sort_keys=True,indent=2)+"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{path.name}: {len(hashes)} files, {path.stat().st_size/1024**2:.1f} MiB, SHA256 {digest(path)}",flush=True)


def tex_files():
    """Follow braced inputs, the paper's plain table-input alias and local styles."""
    # Keep the same archive entry point when packaging the older forest paper.
    forest = PAPER/"minpi-forest.tex"
    pending = [forest if forest.is_file() else PAPER/"main.tex"]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix!=".tex":
            continue
        body = re.sub(r"(?<!\\)%[^\n]*","",path.read_text(encoding="utf8"))
        inputs = re.findall(r"\\(?:input|include|tableinput)\s*\{([^}]+)\}",body)
        inputs += re.findall(r"\\tableinput\s+([\w./-]+\.tex)",body)
        for value in inputs:
            child = path.parent/value
            pending.append(child if child.suffix else child.with_suffix(".tex"))
        for value in re.findall(r"\\bibliography\{([^}]+)\}",body):
            pending.extend(PAPER/(name.strip()+".bib") for name in value.split(","))
        for value in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}",body):
            for name in value.split(","):
                style = PAPER/(name.strip()+".sty")
                if style.exists():
                    seen.add(style)
        for value in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",body):
            image = PAPER/value
            if not image.suffix:
                image = next(p for p in (image.with_suffix(s) for s in (".pdf",".png",".jpg")) if p.is_file())
            seen.add(image)
    return seen


def compile_paper(folder):
    env = {**os.environ,"SOURCE_DATE_EPOCH":"1794873600","FORCE_SOURCE_DATE":"1"}
    tex = ["pdflatex","-interaction=nonstopmode","-halt-on-error","-no-shell-escape","main.tex"]
    for command in (tex,["bibtex","main"],tex,tex):
        checked(command,folder,capture=True,env=env)
    log = (folder/"main.log").read_text(errors="replace")
    if re.search(r"undefined (references|citations)|Citation .* undefined|Reference .* undefined",log,re.I):
        raise RuntimeError("Unresolved manuscript references or citations")


def raw_sources():
    base = ROOT/"data/minpi_rework"
    read = lambda name: json.loads((EXPERIMENT/name).read_bytes())
    pins = {v[0]:v[2] for v in read("validation_sources.json").values()}
    pins.update({f"wide-{k}.zip":v[1] for k,v in read("validation_wide_sources.json").items()})
    for manifest in ("validation_exchange_sources.json","validation_predictive_sources.json"):
        pins.update({v["filename"]:v["sha256"] for v in read(manifest).values()})
    pins.update({f"veto-{v[0]}.zip":v[2] for v in read("wide_veto_sources.json").values()})
    pins["field-cost-45.zip"] = json.loads((EVIDENCE/"field-cost/lock.json").read_bytes())["source"]
    if len(pins)!=30:
        raise ValueError("Expected 30 public raw inputs in the core package")
    for name, pin in pins.items():
        path = base/name
        if digest(path)!=pin:
            raise ValueError(f"Provider digest mismatch: {name}")
        yield path


def local_modules(paths):
    """Include local transitive imports without copying unrelated active studies."""
    more = list(paths)
    seen = set()
    while more:
        path = more.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_bytes(),filename=str(path))):
            if isinstance(node,ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    base = ROOT/"src" if path.is_relative_to(ROOT/"src") else ROOT
                    package = ".".join(path.relative_to(base).parts[:-1])
                    mod = resolve_name("."*node.level+mod,package)
                names = [mod,*(mod+"."+a.name for a in node.names)]
            else:
                continue
            for name in names:
                if name.startswith("minppi_mofs"):
                    base = ROOT/"src"
                elif name.startswith("experiments."):
                    base = ROOT
                else:
                    continue
                stem = base/Path(*name.split("."))
                file = stem/"__init__.py" if stem.is_dir() else stem.with_suffix(".py")
                if file.is_file():
                    more.append(file)
    return seen


def release_files():
    files = set(raw_sources())
    for cohort in COHORTS:
        path = PACKAGES/cohort.archive
        expected = (EVIDENCE/cohort.group/cohort.archive_hash).read_text().split()[0]
        if digest(path)!=expected:
            raise ValueError(f"Sealed forecast archive mismatch: {path.name}")
        files.add(path)
    roots = []
    for pattern in ("quota_compress_tree*.py","quota_compress_scale*.py","tree_*py","stratified_tree*.py",
                    "span_*.py","scalar_tree*.py","compress_model*.py","validation*.py",
                    "predictive*.py","wide_veto*.py","field_cost*.py","wearable*.py",
                    "observed_availability.py","prepared_forest_comparison.py",
                    "quota_coalition_rf_check.py","availability_checks.py"):
        roots.extend(EXPERIMENT.glob(pattern))
    roots.append(EXPERIMENT/"quota_compress_rescue_check.py")
    roots.append(ROOT/"src/minppi_mofs/__init__.py")
    modules = local_modules(roots)
    files.update(modules)
    files.update(p.with_suffix(".md") for p in modules if p.with_suffix(".md").exists())
    files.update(EXPERIMENT.glob("validation*.json"))
    files.update(EXPERIMENT.glob("predictive*.md"))
    files.update((EXPERIMENT/"wearable_protocol.md",EXPERIMENT/"wearable_sources.json",EXPERIMENT/"field_cost_protocol.md"))
    files.update(EXPERIMENT.glob("wearable*.md"))
    device_archive = PACKAGES/"minpi-wearable-forecasts.zip"
    if digest(device_archive)!=(EVIDENCE/"wearable-groups/archive-sha256.txt").read_text().split()[0]:
        raise ValueError("Unexpected wearable forecast package")
    files.add(device_archive)
    refit_archive = PACKAGES/"minpi-wearable-refit-forecasts.zip"
    if digest(refit_archive)!=(EVIDENCE/"wearable-groups/refit-archive-sha256.txt").read_text().split()[0]:
        raise ValueError("Unexpected bounded wearable refit forecast package")
    files.add(refit_archive)
    files.update((PAPER/"build_reviewer.py",PAPER/"reviewer_scores.py",PAPER/"reviewer-scores.md",
                  PAPER/"reviewer-requirements.txt"))
    files.update((EXPERIMENT/"wide_veto_sources.json",EXPERIMENT/"wide_veto_protocol.md"))
    files.update((EXPERIMENT/"README.md",EXPERIMENT/"span_findings.md",EXPERIMENT/"stratified_tree_protocol.md",
                  EXPERIMENT/"validation_protocol.md",ROOT/"experiments/minpi_compression/protocol.json",
                  ROOT/"experiments/minpi_compression/legacy_gain_protocol.md",
                  ROOT/"docs/thesis/archives/piv1/protocol.json",ROOT/"pyproject.toml",Path(__file__).resolve(),
                  PAPER/"tree-replication.md",PAPER/"tree-requirements.txt"))
    files.add(PAPER/"evidence/minpi_compression/legacy-gain/novel-input-rows.npz")
    files.update((PAPER/"minpi-tree-rows.tex",PAPER/"minpi-predictive-rows.tex"))
    files.add(EVIDENCE/"availability/two-sample-five.csv")
    for folder in ("quota-tree","quota-tree-confirmation","quota-scale-confirmation","stratified-tree","exchange-validation",
                   "predictive-confirmation","predictive-mc-confirmation","predictive-prepared","predictive-validation",
                   "predictive-lattice","predictive-veto-timing","observed-availability","wide-veto",
                   "predictive-refit","predictive-polynomial","predictive-reachable","predictive-width",
                   "predictive-universe","wide-veto-ablation","field-cost","wearable-groups"):
        files.update(p for p in (EVIDENCE/folder).rglob("*") if p.is_file())
    files.update(EVIDENCE/group/name for group,names in TABLES.items() for name in names)
    for folder in ("tree-ranking-controls","quota-tree"):
        files.update(p for p in (ROOT/"results/minpi_rework"/folder).rglob("*")
                     if p.suffix in {".npz",".json"})
    wide_controls = json.loads((EVIDENCE/"wide-veto-ablation/lock.json").read_bytes())
    if len(wide_controls)!=20:
        raise ValueError("Incomplete wide ranking control lock")
    for key,pin in wide_controls.items():
        path = ROOT/"results/minpi_rework/wide-veto-ablation"/(key+".json")
        if digest(path)!=pin:
            raise ValueError(f"Wide ranking control receipt mismatch: {key}")
        files.add(path)
    for pattern in ("Tree*.lean","Tree*.wl","tree-*-receipt.json","lean-toolchain"):
        files.update((PAPER/"formal").glob(pattern))
    return files


def prepared_files():
    files = set()
    tasks = set()
    for manifest in (ROOT/"results/icomp2026_formal").glob("*/data_manifest.csv"):
        files.add(manifest)
        with manifest.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                if row["dataset_id"] not in {"abalone","credit_card_fraud","gisette","isolet","madelon","spambase"}:
                    continue
                tasks.add((row["dataset_id"],row["seed"]))
                files.update(manifest.parent/row[key] for key in ("train_relpath","val_relpath","test_relpath"))
    if len(tasks)!=30:
        raise ValueError(f"Missing prepared splits, found {len(tasks)}/30")
    # The historical tree/guard experiments use these previously frozen receivers.
    files.update(p for p in (ROOT/"results/minpi_compression/legacy-gain/xgb-receiver").rglob("*")
                 if p.suffix in {".json",".npz"})
    prepared_receipts = json.loads((EVIDENCE/"predictive-prepared/receipts.json").read_bytes())
    if len(prepared_receipts)!=30:
        raise ValueError("Expected 30 completed original-six predictive forest fits")
    for key,pin in prepared_receipts.items():
        folder = ROOT/"results/minpi_rework/predictive-prepared"/key
        rec,forecasts = folder/"selection.json",folder/"forecasts.npz"
        if digest(rec)!=pin or digest(forecasts)!=json.loads(rec.read_bytes())["forecast"]:
            raise ValueError(f"Prepared predictive run differs: {key}")
        files.update((rec,forecasts))
    return files


def wearable_windows():
    from experiments.minpi_rework import wearable_data, wearable_assets

    files = {}
    for name,pin in wearable_assets.PREPARED.items():
        key = f"data/minpi_rework/wearable-{name}-prepared.npz"
        file = ROOT/key if (ROOT/key).exists() else wearable_data.prepare(name)
        if digest(file)!=pin:
            raise ValueError(f"Changed portable wearable window file: {name}")
        files[key] = file
    return files


def build(with_prepared, with_wearable_raw=False):
    (ROOT/"tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="tree-manuscript-") as tmp:
        folder = Path(tmp)
        manuscript = {("main.tex" if p.name == "minpi-forest.tex" else p.relative_to(PAPER).as_posix()):p
                      for p in tex_files()}
        for name, p in manuscript.items():
            target = folder/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        compile_paper(folder)
        pdf = PAPER/"minpi-tree.pdf"
        shutil.copyfile(folder/"main.pdf",pdf)
        build_folder = PAPER/"build-tree"
        build_folder.mkdir(exist_ok=True)
        for ext in ("pdf","log","bbl","blg","aux","out"):
            item = folder/("main."+ext)
            if item.exists():
                shutil.copyfile(item,build_folder/item.name)
        # The live manuscript can be edited while the clean copy is compiling.
        staged = {name:folder/name for name in manuscript}
        archive(PACKAGES/"minpi-tree-source.zip",{**staged,"main.pdf":folder/"main.pdf"})
        members = {p.relative_to(ROOT).as_posix():p for p in release_files()}
        members.update(wearable_windows())
        members.update({"thesis-paper/"+name:source for name,source in staged.items()})
        guide = PAPER/"tree-replication.md"
        entry = folder/"README.md"
        guide_text = guide.read_text(encoding="utf8").replace("](../experiments/","](experiments/")
        guide_text = guide_text.replace("](evidence/","](thesis-paper/evidence/")
        entry.write_text(guide_text.replace("](reviewer-scores.md)","](thesis-paper/reviewer-scores.md)"),encoding="utf8")
        members.update({"README.md":entry,"thesis-paper/minpi-tree.pdf":folder/"main.pdf"})
        archive(PACKAGES/"minpi-tree-replication.zip",members)
    if with_prepared:
        archive(PACKAGES/"minpi-tree-prepared-six.zip",{p.relative_to(ROOT).as_posix():p for p in prepared_files()},
                "PREPARED-SHA256SUMS.json")
    if with_wearable_raw:
        pins = json.loads((EXPERIMENT/"wearable_sources.json").read_bytes())
        originals = {f"data/minpi_rework/wearable-{name}.zip":ROOT/f"data/minpi_rework/wearable-{name}.zip" for name in pins}
        if any(digest(originals[f"data/minpi_rework/wearable-{name}.zip"])!=pin for name,pin in pins.items()):
            raise ValueError("Wearable source digest mismatch")
        archive(PACKAGES/"minpi-wearable-raw.zip",originals,"WEARABLE-RAW-SHA256SUMS.json")


def verify_manifest(name="SHA256SUMS.json"):
    hashes = json.loads((ROOT/name).read_bytes())
    for name, expected in hashes.items():
        if digest(ROOT/name)!=expected:
            raise ValueError(f"Release digest mismatch: {name}")
    print(f"Verified {len(hashes)} released file hashes",flush=True)


def restore():
    for cohort in COHORTS:
        sealed = PACKAGES/cohort.archive
        lock_file = EVIDENCE/cohort.group/"lock.json"
        locks = json.loads(lock_file.read_bytes())["receipts"]
        if len(locks)!=cohort.count or digest(sealed)!=(lock_file.parent/cohort.archive_hash).read_text().split()[0]:
            raise ValueError(f"Forecast archive mismatch: {cohort.group}")
        with ZipFile(sealed) as z:
            if z.read("lock.json")!=lock_file.read_bytes():
                raise ValueError(f"Sealed receipt lock differs: {cohort.group}")
            for key,expected in locks.items():
                d = ROOT/"results/minpi_rework"/cohort.group/key
                d.mkdir(parents=True,exist_ok=True)
                rec = z.read(f"{key}/{cohort.receipt}")
                fields = json.loads(rec)[cohort.artifacts]
                if cohort.artifacts=="forecast":
                    fields = {cohort.forecast:fields}
                elif set(fields)!={"forecasts.npz","increments.npz","incidence.npz"}:
                    raise ValueError(f"Incomplete mask game in {key}")
                for name,pin in {cohort.receipt:expected,**fields}.items():
                    raw = z.read(f"{key}/{name}")
                    if hashlib.sha256(raw).hexdigest()!=pin:
                        raise ValueError(f"Corrupt sealed file: {key}/{name}")
                    dest = d/name
                    if dest.exists() and dest.read_bytes()!=raw:
                        raise ValueError(f"Existing run differs from archive: {dest}")
                    if not dest.exists():
                        dest.write_bytes(raw)
        print(f"Verified and restored {cohort.count} locked {cohort.group} runs",flush=True)
    with ZipFile(PACKAGES/"minpi-wearable-forecasts.zip") as z:
        expected_names = set()
        for source,n in (("mhealth",10),("dsads",8)):
            lock_name = f"lock-{source}.json"
            expected_names.add(lock_name)
            raw_lock = z.read(lock_name)
            if raw_lock!=(EVIDENCE/"wearable-groups"/lock_name).read_bytes():
                raise ValueError("Wearable support lock differs")
            locks = json.loads(raw_lock)["receipts"]
            if set(locks)!={f"{source}-{i}" for i in range(n)}:
                raise ValueError("Incomplete wearable participant folds")
            for key,pin in locks.items():
                rec = json.loads(z.read(f"{key}/selection.json"))
                for filename,checksum in (("selection.json",pin),("forecasts.npz",rec["forecast_sha256"])):
                    member = f"{key}/{filename}"
                    expected_names.add(member)
                    raw = z.read(member)
                    dest = ROOT/"results/minpi_rework/wearable-groups"/member
                    if hashlib.sha256(raw).hexdigest()!=checksum or (dest.exists() and dest.read_bytes()!=raw):
                        raise ValueError(f"Mismatch in sealed {member}")
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():
                        dest.write_bytes(raw)
        if set(z.namelist()) != expected_names:
            raise ValueError("Unexpected wearable forecast ZIP members")
    print("Verified and restored 18 held-participant forecast sets",flush=True)


def replay_mc_costs():
    """Recover setup and search clocks from the signed measurement rows without timing the verifier."""
    import pandas as pd
    from experiments.minpi_rework.predictive_mc_report import summary

    folder = EVIDENCE/"predictive-mc-confirmation"
    keys = json.loads((folder/"lock.json").read_bytes())["receipts"]
    runs = ROOT/"results/minpi_rework/predictive-mc-confirmation"
    rows = [row for key in keys for row in json.loads((runs/key/"selection.json").read_bytes())["rows"]]
    repeat = pd.DataFrame(rows)
    repeat.to_csv(folder/"repeats.csv",index=False)
    jobs = repeat.groupby(["dataset","seed","draw","method"],as_index=False).median(numeric_only=True)
    jobs.to_csv(folder/"jobs.csv",index=False)
    summary(jobs[~jobs.method.str.startswith("direct")]).to_csv(folder/"contrasts.csv",index=False)


def verify(check_tex, prepared):
    import pandas as pd
    from pandas.testing import assert_frame_equal

    verify_manifest()
    if prepared:
        verify_manifest("PREPARED-SHA256SUMS.json")
    list(raw_sources())
    restore()
    for family in ("", "controls", "guard"):
        suffix = (family+"-" if family else "") + "42-123-456-789-1024"
        lock = json.loads((EVIDENCE/"quota-tree"/f"lock-{suffix}.json").read_bytes())
        if len(lock)!=30:
            raise ValueError("Incomplete historical six-source study")
        for key, expected_hash in lock.items():
            folder = ROOT/"results/minpi_rework/quota-tree"/family/key
            receipt = folder/"selection.json"
            if digest(receipt)!=expected_hash or digest(folder/"predictions.npz")!=json.loads(receipt.read_bytes())["forecast_hash"]:
                raise ValueError(f"Historical original-six receipt mismatch: {family}/{key}")
    print("Verified 90 historical original-six arm-family receipt/prediction pairs",flush=True)
    expected = {EVIDENCE/group/name:pd.read_csv(EVIDENCE/group/name) for group,names in TABLES.items() for name in names}
    tables_before = {p:p.read_text(encoding="utf8") for p in
                     (PAPER/"minpi-predictive-rows.tex",PAPER/"minpi-tree-rows.tex")}
    env = {**os.environ,"PYTHONPATH":os.pathsep.join(map(str,(ROOT/"src",ROOT))),
           "PYTHONDONTWRITEBYTECODE":"1","NUMBA_CACHE_DIR":str(ROOT/"tmp/numba")}
    checked([sys.executable,"-B","-c",
             "import minppi_mofs as m; from pathlib import Path; "
             "assert Path(m.__file__).resolve().is_relative_to(Path.cwd()/'src')"],ROOT,env=env)
    jobs = [("quota_compress_tree_confirmation","score"),("tree_rank_controls","score"),
            ("quota_compress_scale_confirmation","score"),("quota_compress_tree_report",),
            ("quota_compress_scale_confirmation_report",),("tree_paper_results",)]
    snapshots = {p:p.read_bytes() for p in [*expected,*tables_before]}
    if prepared:
        seeds = ("--seeds","42","123","456","789","1024")
        jobs.extend([("quota_compress_tree_run","score",*seeds),
                     ("quota_compress_tree_run","score","--controls",*seeds),
                     ("quota_compress_tree_guard","score",*seeds),("predictive_prepared",)])
        pred_dir = EVIDENCE/"predictive-prepared"
        for filename in ("scores.csv","receipts.json"):
            p = pred_dir/filename
            snapshots[p] = p.read_bytes()
        expected[pred_dir/"scores.csv"] = pd.read_csv(pred_dir/"scores.csv")
        locked = json.loads(snapshots[pred_dir/"receipts.json"])
        jobs_root = ROOT/"results/minpi_rework/predictive-prepared"
        if len(locked)!=30 or any(digest(jobs_root/key/"selection.json")!=pin for key,pin in locked.items()):
            raise ValueError("Original-six supervised receipts incomplete or stale")
        for short in ("","controls-","guard-"):
            key = short+"42-123-456-789-1024"
            for pattern in (f"lock-{key}.json",f"receipts-{key}.zip",f"historical-{key}.csv"):
                p = EVIDENCE/"quota-tree"/pattern
                snapshots[p] = p.read_bytes()
            expected[p] = pd.read_csv(p)
    jobs.extend([("predictive_confirmation","score"),("predictive_mc_score",),("predictive_paper",),
                 ("predictive_mask_audit","packages/minpi-predictive-seven-forecasts.zip",
                  "thesis-paper/evidence/minpi_rework/predictive-confirmation"),
                 ("predictive_lattice","--check"),("predictive_lattice",),
                 ("predictive_veto_report",),("predictive_veto_report","--native"),
                 ("predictive_rf_deploy_check","--sealed"),("observed_availability","score"),
                 ("wide_veto_score",),("wide_veto_verify",),
                 ("predictive_refit","check"),("predictive_refit","score"),("predictive_refit_report",),
                 ("predictive_polynomial_check",),("predictive_polynomial_report",),("predictive_reachable_score",),
                 ("prepared_forest_comparison",),("predictive_width_timing","--summarize"),
                 ("wide_veto_ablation","score"),("predictive_universe","score"),("predictive_universe_report",),
                 ("predictive_stability",),("predictive_group_check",),("field_cost","score"),
                 ("wearable_score","--prepared","data/minpi_rework"),
                 ("wearable_refit_offline","--prepared","data/minpi_rework","--refit-archive",
                  "packages/minpi-wearable-refit-forecasts.zip")])
    try:
        checked([sys.executable,"-B",str(Path(__file__).resolve()),"replay-mc-costs"],ROOT,env=env,capture=True)
        for series in ("pipeline","production","mdi"):
            checked([sys.executable,"-B","-c",
                     f"from experiments.minpi_rework.predictive_polynomial_pipeline import summarize; summarize('{series}')"],
                    ROOT,env=env,capture=True)
        for name,*arguments in jobs:
            print("Replaying",name,*arguments,flush=True)
            checked([sys.executable,"-B","-m","experiments.minpi_rework."+name,*arguments],ROOT,env=env,capture=True)
        for path, frame in expected.items():
            actual = pd.read_csv(path)
            if path==EVIDENCE/"predictive-lattice/scores.csv":
                # The lattice is refit and timed anew. Check every support and score.
                if not actual["lattice_s"].gt(0).all():
                    raise ValueError("Incomplete fresh lattice clocks")
                actual,frame = actual.drop(columns="lattice_s"),frame.drop(columns="lattice_s")
            assert_frame_equal(actual,frame,check_dtype=False,rtol=1e-12,atol=1e-12,
                               obj=str(path.relative_to(ROOT)))
        for path,table in tables_before.items():
            if path.read_text(encoding="utf8")!=table:
                raise ValueError(f"Regenerated LaTeX source table changed: {path.name}")
    finally:
        # Restore exact archive bytes (including line endings and ZIP metadata).
        for path,raw in snapshots.items():
            if path.read_bytes()!=raw:
                path.write_bytes(raw)
    if check_tex:
        compile_paper(PAPER)
    print(f"Recomputed {len(expected)} archived CSV tables and {len(tables_before)} typeset numerical tables.",flush=True)


def verify_printed_scores(pdf_text, evidence):
    """Check the displayed precision of both primary tables after PDF layout."""
    import pandas as pd

    lines = [s.strip().replace("−","-") for s in pdf_text.decode("utf8").splitlines()]
    primary = pd.read_csv(evidence/"predictive-confirmation/source-means.csv").set_index(["dataset","method"])
    reach = pd.read_csv(evidence/"predictive-reachable/source-means.csv").set_index(["dataset","arm"])
    labels = {"votes":"House voting","churn":"Iranian churn","liver":"Liver patients","german":"German credit",
              "seoul":"Seoul bicycles","beijing":"Beijing PM2.5","sgemm":"SGEMM runtime",
              "adverts":"Advertisements","oraltox":"Oral toxicity","tehran":"Tehran housing","uji":"UJI latitude"}
    for name,label in labels.items():
        if name in primary.index.get_level_values("dataset"):
            f,mdi,v = (primary.loc[(name,arm)] for arm in ("full","mdi_min","mdi_plus_min_min"))
            expected = [f.p,mdi.k,v.k]
            first_score = f
        else:
            f,mdi,v = (reach.loc[(name,arm)] for arm in ("full","mdi_original","portfolio_min_original"))
            expected = [f.p,f.reachable,mdi.reachable,v.reachable]
            first_score = mdi
        metrics = ("auc","accuracy") if pd.notna(v.auc) else ("r2","mae")
        expected.extend(row[m] for row in (first_score,v) for m in metrics)
        row = next(line[len(label):].strip() for line in lines
                   if line.startswith(label+" ") and re.match(r"\s+\d",line[len(label):]))
        tokens = row.replace("/"," ").split()
        if len(tokens)!=len(expected):
            raise ValueError(f"Incomplete printed score table row: {label}")
        for token,value in zip(tokens,expected):
            decimals = len(token.partition(".")[2])
            if abs(float(token)-value)>0.5*10**(-decimals)+1e-10:
                raise ValueError(f"Printed score differs for {label}: {token} versus {value}")
    print(f"Verified widths and scores in all {len(labels)} PDF table rows.",flush=True)


def verify_archives(prepared,wearable_raw=False):
    """Reproduce the published scores using only files inside fresh extracted copies."""
    (ROOT/"tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(dir=ROOT/"tmp",prefix="tree-release-test-") as tmp:
        dest = Path(tmp)/"replication"
        with ZipFile(PACKAGES/"minpi-tree-replication.zip") as z:
            z.extractall(dest)
        manuscript = Path(tmp)/"source"
        with ZipFile(PACKAGES/"minpi-tree-source.zip") as z:
            z.extractall(manuscript)
        if prepared:
            with ZipFile(PACKAGES/"minpi-tree-prepared-six.zip") as z:
                z.extractall(dest)
        if wearable_raw:
            with ZipFile(PACKAGES/"minpi-wearable-raw.zip") as z:
                z.extractall(dest)
        env = {**os.environ,"PYTHONPATH":os.pathsep.join(map(str,(dest/"src",dest))),
               "PYTHONDONTWRITEBYTECODE":"1","NUMBA_CACHE_DIR":str(dest/"tmp/numba")}
        args = [sys.executable,"-B","thesis-paper/tree_release.py","verify","--tex"]
        if wearable_raw:
            checked([sys.executable,"-B",str(dest/"thesis-paper/tree_release.py"),"verify-wearable-raw"],dest,env=env)
        checked(args+(["--prepared"] if prepared else []),dest,env=env)
        for name,h in json.loads((manuscript/"SHA256SUMS.json").read_bytes()).items():
            if digest(manuscript/name)!=h:
                raise ValueError(f"Standalone source archive differs at {name}")
        compile_paper(manuscript)
        included = dest/"thesis-paper/minpi-tree.pdf"
        # Different directories can affect PDF trailer IDs. Compare complete page text.
        contents = [checked(["pdftotext","-layout",str(pdf),"-"],dest,capture=True).stdout
                    for pdf in (included,manuscript/"main.pdf",dest/"thesis-paper/main.pdf")]
        if not contents[0].strip() or len(set(contents))!=1:
            raise ValueError("Independently compiled documents differ in page text")
        verify_printed_scores(contents[0],dest/"thesis-paper/evidence/minpi_rework")
        print("Standalone and replication manuscripts reproduce the complete included PDF text.",flush=True)


def verify_wearable_raw():
    import numpy as np
    from experiments.minpi_rework import wearable_data

    verify_manifest("WEARABLE-RAW-SHA256SUMS.json")
    for name in wearable_data.SOURCES:
        rebuilt = wearable_data.prepare(name)
        with np.load(rebuilt,allow_pickle=False) as one, np.load(wearable_data.DATA/f"wearable-{name}-prepared.npz",allow_pickle=False) as two:
            for key in ("x","y","subject","unit"):
                np.testing.assert_array_equal(one[key],two[key])
    print("Both original wearable archives reproduce every released window, label, subject, and device assignment.",flush=True)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action",required=True)
    make = sub.add_parser("build")
    make.add_argument("--with-prepared",action="store_true")
    make.add_argument("--with-wearable-raw",action="store_true")
    replay = sub.add_parser("verify")
    replay.add_argument("--tex",action="store_true")
    replay.add_argument("--prepared",action="store_true")
    audit = sub.add_parser("verify-archives")
    audit.add_argument("--prepared",action="store_true")
    audit.add_argument("--wearable-raw",action="store_true")
    sub.add_parser("verify-wearable-raw")
    sub.add_parser("replay-mc-costs",help="Rebuild MC work summaries from locked timing receipts")
    args = cli.parse_args()
    if args.action=="build":
        build(args.with_prepared,args.with_wearable_raw)
    elif args.action=="verify-archives":
        verify_archives(args.prepared,args.wearable_raw)
    elif args.action=="verify-wearable-raw":
        verify_wearable_raw()
    elif args.action=="replay-mc-costs":
        replay_mc_costs()
    else:
        verify(args.tex,args.prepared)
