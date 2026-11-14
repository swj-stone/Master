"""Export the latest audited formal-study CSVs. No models are run by this tool."""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from experiments.report_formal_literature import DEFAULT, KEYS, PREDICTIONS, PROJECT, SCORES, require


def _csv(path):
    text_fields = {"dataset","mode","method","status","feature","sampling_policy"}
    ints = {"seed","width","width_min","width_max","k","deleted","fits","fits_started","n_planned",
            "n_completed","source_position","selected_position", *PREDICTIONS}
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return [{k:(None if not v else v if k in text_fields else int(v) if k in ints else float(v))
                 for k,v in row.items()} for row in csv.DictReader(stream)]


def _key(row, keys=KEYS):
    return tuple(row[k] for k in keys)


def _column(index):
    label = ""
    while index:
        index, digit = divmod(index - 1, 26)
        label = chr(ord("A") + digit) + label
    return label


def _verify(filename, data):
    ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    # Export order can differ from protocol order; resolve each row by its identity.
    ds={"Abalone":"abalone","Spambase":"spambase","Madelon":"madelon","Credit Card Fraud":"credit_card_fraud",
        "Gisette":"gisette","ISOLET":"isolet"}
    modes={"共享参照":"shared","文献估计量":"native","统一冻结效用":"frozen"}
    methods={"全列模型":"full","minPI":"minpi","紧凑 Gain":"gain","快速 PFI":"pfi","MinShap":"minshap",
             "排列 Shapley":"shapley","SamplingSHAP":"sampling_shap","采样 Banzhaf":"banzhaf","ShapG":"shapg"}
    fields=("k","deleted","fits","wall_s",*SCORES)
    predictions = PREDICTIONS if "sample_prediction_rows" in data["rows"][0] else ()
    details = ("width", *fields, *["full_"+k for k in SCORES], "selector_s", *predictions)
    if predictions:
        details += ("sampling_policy",)
    rows={_key(r):r for r in data["rows"]}
    groups={_key(r,KEYS[1:]):r for r in data["groups"]}
    features={(_key(r),r["selected_position"]):(r["source_position"],r["feature"]) for r in data["features"]}
    seen=[set(),set(),set()]
    def compare(actual,expected):
        if isinstance(expected,(int,float)):
            require(isinstance(actual,(int,float)) and math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-11),
                    f"Workbook value mismatch: {actual} vs {expected}")
        else:
            require(actual==expected, f"Workbook value mismatch: {actual} vs {expected}")

    with ZipFile(filename) as z:
        require(z.testzip() is None, "Invalid XLSX package")
        shared=ET.fromstring(z.read("xl/sharedStrings.xml"))
        strings=["".join(t.itertext()) for t in shared.findall("m:si",ns)]
        w=ET.fromstring(z.read("xl/workbook.xml"))
        names=[s.get("name") for s in w.findall("m:sheets/m:sheet",ns)]
        require(names==["跨 seed 汇总","逐 seed 明细","保留特征"],"Unexpected sheets")
        for i,name in enumerate(names):
            sheet=ET.fromstring(z.read(f"xl/worksheets/sheet{i+1}.xml"))
            pane=sheet.find("m:sheetViews/m:sheetView/m:pane",ns)
            require(pane is not None and pane.get("topLeftCell")=="C5",f"Missing fixed headings: {name}")
            for record in sheet.findall(".//m:sheetData/m:row",ns):
                idx=int(record.get("r"))
                if idx<5:
                    continue
                values={}
                for c in record.findall("m:c",ns):
                    typ=c.get("t")
                    require(typ!="e",f"Excel error at {name}:{c.get('r')}")
                    text=c.findtext("m:v",None,ns)
                    if text is not None:
                        col=c.get("r").rstrip("0123456789")
                        values[col]=strings[int(text)] if typ=="s" else text if typ=="str" else float(text)
                if not values:
                    continue
                group=(ds[values["A"]],modes[values["B"]],methods[values["C"]])
                if i==0:
                    reference=groups[group]
                    identity=group
                    compare(values["D"],reference["n_completed"])
                    compare(values["E"],reference["n_planned"])
                    for j,f in enumerate((*fields, *predictions)):
                        for delta,suffix in ((0,"mean"),(1,"sd")):
                            compare(values.get(_column(7+2*j+delta)),reference.get(f+"_"+suffix))
                else:
                    route=(int(values["D"]),*group)
                    if i==1:
                        reference=rows[route]
                        identity=route
                        for j,f in enumerate(details):
                            compare(values.get(_column(6+j)),reference.get(f))
                    else:
                        identity=(route,int(values["E"]))
                        compare(values["F"],features[identity][0])
                        compare(values["G"],features[identity][1])
                require(identity not in seen[i],f"Duplicate Excel row: {name} {identity}")
                seen[i].add(identity)
        require(seen==[set(groups),set(rows),set(features)],"Incomplete Excel export")


def build(root=DEFAULT):
    root=Path(root).resolve()
    data={name:_csv(root/file) for name,file in (("rows","per-seed.csv"),("groups","five-seed-summary.csv"),
                                                 ("features","retained-features.csv"))}
    with (root/"protocol.json").open(encoding="utf-8") as f:
        expected={_key(j) for j in json.load(f)["jobs"]}
    require(expected=={_key(r) for r in data["rows"]} and len(expected)==len(data["rows"]),"Stale run list")
    done={_key(r):r for r in data["rows"] if r["status"]=="completed"}
    counts=Counter(_key(f) for f in data["features"])
    require(all(counts[k]==r["k"] for k,r in done.items()) and not counts.keys()-done.keys(),
            "Feature list and results describe different snapshots. Refresh the report first.")
    runtime=Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    node=runtime/"bin"/("node.exe" if os.name=="nt" else "node")
    require(node.is_file() and (runtime/"node_modules/@oai/artifact-tool").is_dir(),"Bundled spreadsheet runtime unavailable")
    staging=PROJECT/"tmp";staging.mkdir(exist_ok=True)
    # The existing workspace runtime link also avoids creating a Windows junction per export.
    def bundled_link(p):
        resolved=p.resolve()
        return ((resolved==runtime/"node_modules" or
                 any(part.startswith("OpenAI.CodexPrimaryRuntime.") for part in resolved.parts))
                and (p/"@oai/artifact-tool").is_dir())
    links = [p for p in staging.glob("*/node_modules") if bundled_link(p)]
    base=links[0].parent if links else staging
    seeds = sorted({row["seed"] for row in data["rows"]})
    if len(seeds) == 1:
        target = root / f"正式实验-seed{seeds[0]}-方法对比.xlsx"
    elif len(seeds) == 5:
        target = root / "正式实验五个seed汇总.xlsx"
    else:
        target = root / f"正式实验{len(seeds)}个seed汇总.xlsx"
    with TemporaryDirectory(prefix="formal-xlsx-",dir=base) as tmpname:
        tmp=Path(tmpname)
        if not links and os.name=="nt":
            subprocess.run(["powershell","-NoProfile","-NonInteractive","-Command",
                            "New-Item -ItemType Junction -Path node_modules -Target $env:FORMAL_EXCEL_DEPS | Out-Null"],
                           cwd=tmp,env={**os.environ,"FORMAL_EXCEL_DEPS":str(runtime/"node_modules")},check=True)
        elif not links:
            (tmp/"node_modules").symlink_to(runtime/"node_modules",target_is_directory=True)
        script=Path(__file__).with_name("formal_literature_workbook.mjs")
        shutil.copyfile(script,tmp/script.name)
        datafile=tmp/"snapshot.json"
        datafile.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),"utf-8")
        working=tmp/target.name
        subprocess.run([str(node),str(tmp/script.name),str(datafile),str(working)],check=True,cwd=tmp)
        _verify(working,data)
        preview=staging/"formal-literature-excel-preview";preview.mkdir(exist_ok=True)
        for png in tmp.glob("*.png"):
            shutil.copy2(png,preview/png.name)
        os.replace(working,target)
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(build(args.run))
