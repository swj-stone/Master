"""Build the submission source and optionally the local replication supplement."""
import argparse
import ast
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import runpy
from shutil import copyfile
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PACKAGES = ROOT / "packages"
ARTIFACTS = PAPER / "evidence/revision/artifacts.json"
REASSESSMENT = PACKAGES/"thesis-refit-reassessment-evidence.zip"
OPERATING = PACKAGES / "thesis-operating-evidence.zip"
TRANSPORT = PACKAGES / "thesis-transport-evidence.zip"
ROBUSTNESS = PACKAGES / "thesis-robustness-evidence.zip"
DECISION = PACKAGES / "thesis-decision-evidence.zip"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def paper_source():
    inputs = {PAPER / "main.tex"}
    pending = list(inputs)
    while pending:
        tex = pending.pop().read_text(encoding="utf8")
        for command, value in re.findall(
                r"\\(input|include|bibliography|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex):
            for token in value.split(","):
                path = PAPER / token.strip()
                if command == "bibliography":
                    path = path.with_suffix(".bib")
                elif command in ("include", "input") and not path.suffix:
                    path = path.with_suffix(".tex")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in inputs:
                    inputs.add(path)
                    if path.suffix == ".tex":
                        pending.append(path)
    result = {p.relative_to(PAPER).as_posix():p for p in inputs}
    for tex in inputs:
        if tex.suffix == ".tex":
            for packages in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf8")):
                for name in packages.split(","):
                    local = PAPER / (name.strip() + ".sty")
                    if local.is_file():
                        result[local.name] = local
    result["main.bbl"] = PAPER / "build/main.bbl"
    return result


def write_archive(destination, members):
    tmp = destination.with_suffix(".partial.zip")
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=3) as pack:
        for member, path in sorted(members.items()):
            if isinstance(path, bytes):
                pack.writestr(member, path)
            else:
                pack.write(path, member)
    with ZipFile(tmp) as pack:
        invalid = pack.testzip()
        if invalid or set(pack.namelist()) != set(members):
            raise ValueError(f"Archive verification failed at {invalid}")
    tmp.replace(destination)


def verify_source(archive, pdf):
    scratch = ROOT/"tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="minpi-source-", dir=scratch) as work:
        with ZipFile(archive) as source:
            source.extractall(work)
        subprocess.run(["latexmk","-pdf","-silent","-interaction=nonstopmode","-halt-on-error",
                        "main.tex"],cwd=work,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        documents = [pdf, Path(work)/"main.pdf"]
        texts = [subprocess.check_output(["pdftotext","-layout",str(doc),"-"]) for doc in documents]
        if texts[0] != texts[1]:
            raise ValueError("Rebuilt submission and delivered PDF text differ")
    print("Standalone source rebuilt with identical PDF text and page breaks.")


def local_module(name):
    for directory in (ROOT, ROOT/"src"):
        prefix = directory.joinpath(*name.split("."))
        for path in (prefix.with_suffix(".py"), prefix / "__init__.py"):
            if path.is_file():
                return path
    return None


def include_imports(files):
    pending = [p for p in files if p.suffix == ".py"]
    while pending:
        path = pending.pop()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        base = ROOT/"src" if path.is_relative_to(ROOT/"src") else ROOT
        package = path.relative_to(base).parts[:-1]
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else ()
                module = ".".join((*prefix, node.module) if node.module else prefix)
                if module:
                    names.append(module)
                    names.extend(f"{module}.{a.name}" for a in node.names if a.name != "*")
        for name in names:
            parts = name.split(".")
            if parts[0] not in {"minppi_mofs", "experiments"}:
                continue
            # Importing a child module also executes each existing package initializer.
            for n in range(1, len(parts) + 1):
                new = local_module(".".join(parts[:n]))
                if new and new not in files:
                    files.add(new)
                    pending.append(new)


RUNS = ROOT / "results/context_review"
EVIDENCE = PAPER / "evidence"


def read(path):
    return json.loads(path.read_bytes())


def guard_targets(folder):
    """Export only the labels needed to score the archived guard predictions."""
    import numpy as np

    layout = read(folder / "layout.json")
    inputs = folder / "inputs.npz"
    if digest(inputs) != layout["inputs_sha256"]:
        raise ValueError(f"Inputs differ from the sealed layout: {inputs}")
    with np.load(inputs) as arrays, BytesIO() as data:
        np.savez_compressed(data, row_id=layout["guard"], y=arrays["y"][arrays["guard"]])
        return data.getvalue()


def transport():
    files = {ROOT / name for name in ("README.md", "AGENTS.md", "thesis-python.ps1", "pyproject.toml",
                                       "requirements.txt", "src/minppi_mofs/__init__.py",
                                       "experiments/test/run_piv0_seed42.py", "tests/test_matched_stopping.py",
                                       "tests/test_comparison_design.py")}
    files.update({PAPER / "README.md", PAPER / "minpi.pdf", PAPER / "minpi-source.zip",
                  Path(__file__).resolve(), EVIDENCE / "transport-release.md", EVIDENCE / "robustness-release.md",
                  EVIDENCE / "decision-release.md", PAPER / "final-review/README.md",
                  PAPER / "final-review/response.md", EVIDENCE / "budget_comparison/timed-source.py",
                  PAPER / "identification-bounds.tex", PAPER / "audit/rounds/pmlb_baselines.py"})
    for pattern in ("transport*", "robustness*", "decision-*", "refit_observation*", "refit-observation*", "identification_bounds*"):
        files.update((PAPER / "audit").glob(pattern))
    files.update(p for p in (PAPER / "formal").iterdir() if p.suffix in {".lean", ".wl"}
                 or p.name == "lean-toolchain")
    files.update((ROOT / "experiments/context_review").glob("*.py"))
    files.update((ROOT / "experiments/context_review").glob("*.md"))
    files.update((ROOT / "configs/context_review").glob("*.json"))
    files.update((ROOT / "configs/context_study").glob("*.json"))
    reviews = list((PACKAGES / "transport-review-drafts").glob("*.pdf"))
    if not reviews:
        raise FileNotFoundError("Missing reviewed transport manuscripts")
    files.update(reviews)
    for name in ("transport", "wide", "low_fidelity", "refinement", "matched_signal", "opportunity",
                 "wide_slate", "chance", "margin_factorial", "receiver", "opportunities", "marginal",
                 "concordance", "ensemble_cells", "transcript_permutations", "transcript_information",
                 "gain_increment", "forest_cross", "filter_neighborhood", "neighborhood_replicates",
                 "transcript_swaps", "score_resolution", "ordinal_decomposition", "matched_stopping",
                 "budget_comparison", "class_precision", "within_tier", "joint_choice", "transfer_choice"):
        files.update(p for p in (EVIDENCE / name).glob("*") if p.suffix in {".csv", ".json", ".md", ".zip", ".npz"})
    campaigns = ["transport-uci", "transport-mammographic-four", "wide-uci", "matched-signal-r2",
                 "refit-opportunity", "low-fidelity", "wide-slate", "ballot-refinement-first", "ballot-refinement-wide"]
    campaigns += [p.name for p in RUNS.glob("margin-factorial*") if p.is_dir()]
    for name in campaigns:
        files.update(p for p in (RUNS / name).rglob("*") if p.suffix in {".csv", ".json", ".npz"}
                     and p.name != "inputs.npz")
    # Earlier receipts required for the historical cost and collision calculations.
    for campaign, names in [
        ("rule-sensitivity", ["votes.npz", "completed.json", "voting.json"]),
        ("refit-target", ["selection.json"]), ("budgeted-guard", ["shortlists.json"])]:
        for name in names:
            files.update((RUNS / campaign).glob(f"*/fold-*/{name}"))
    files.add(RUNS / "refit-guard/eligibility.csv")
    files = {p for p in files if not p.is_dir()}
    include_imports(files)
    members = {p.relative_to(ROOT).as_posix(): p for p in files}
    for name in ("transport-uci", "wide-uci"):
        for layout in (RUNS / name).glob("*/fold-*/layout.json"):
            key = layout.with_name("guard-targets.npz").relative_to(ROOT).as_posix()
            members[key] = guard_targets(layout.parent)
    return members


def verify_slate(panel, campaign, lock):
    """Check guard scores independently against released labels and probabilities."""
    import numpy as np
    from experiments.context_review.wide_audit import metrics

    for key, expected in lock.items():
        fold, base = RUNS / campaign / key, RUNS / panel / key
        receipt = fold / "sealed.json"
        state = read(receipt)
        if (digest(receipt) != expected or digest(fold / "predictions.npz") != state["predictions"]
                or digest(base / "layout.json") != state["layout"]
                or digest(fold.parent.parent / "protocol.json") != state["design"]):
            raise ValueError(f"Changed slate receipt: {fold}")
        with np.load(base / "guard-targets.npz") as guard, np.load(fold / "predictions.npz") as pred:
            np.testing.assert_array_equal(guard["row_id"], read(base / "layout.json")["guard"])
            for idx, reference in enumerate(state["guard_metrics"]):
                actual = metrics(guard["y"], pred[f"g{idx}"], state["rare"])
                assert actual.keys() == reference.keys()
                np.testing.assert_allclose([actual[k] for k in reference], list(reference.values()),
                                           atol=1e-12, rtol=0)


def verify_external_slates():
    for panel, primary, campaign, evidence, slate in [
        ("first", "transport-uci", "low-fidelity", "transport", "transport/low-fidelity-lock.json"),
        ("wide", "wide-uci", "wide-slate", "wide", "wide_slate/prediction-lock.json"),
    ]:
        lock = read(EVIDENCE / slate)
        primary_lock = EVIDENCE / evidence / "prediction-lock.json"
        expected = read(primary_lock)["selections" if panel == "first" else "folds"]
        assert set(lock) == set(expected)
        verify_slate(primary, campaign, lock)
        factors = read(EVIDENCE / "margin_factorial" / f"{panel}-lock.json")
        assert factors["code"] == digest(ROOT / "experiments/context_review/margin_factorial.py")
        assert factors["design"] == digest(ROOT / "configs/context_review/margin_factorial.json")
        assert factors["original"] == digest(primary_lock)
        refined = read(RUNS / f"ballot-refinement-{panel}/protocol.json")
        assert refined["implementation"] == digest(ROOT / "experiments/context_review/ballot_refinement.py")
        assert refined["design"] == digest(ROOT / "configs/context_review/ballot_refinement.json")
        assert refined["inputs"] == digest(RUNS / primary / "protocol.json")


def verify_wide_fold_scores():
    """Verify the committed probability archive and independently score every policy."""
    import numpy as np
    import pandas as pd
    from experiments.context_review import wide_audit

    wide = EVIDENCE / "wide"
    seal = read(wide / "prediction-lock.json")
    status = read(wide / "fold-status.json")
    assert set(seal["folds"]) == {key for key, entry in status.items() if entry["feasible"]}
    assert seal["planned"] == len(status) and seal["feasible"] == len(seal["folds"])
    assert digest(wide / "predictions.zip") == seal["package_sha256"]
    for name, expected in seal["dependencies"]["files"].items():
        assert digest(ROOT / name) == expected, name
    published = pd.read_csv(wide / "fold-results.csv").set_index(["dataset", "fold", "arm"])
    with ZipFile(wide / "predictions.zip") as models, ZipFile(wide / "outcomes.zip") as labels:
        assert hashlib.sha256(models.read("protocol.json")).hexdigest() == seal["protocol"]
        for key, files in seal["folds"].items():
            fold = RUNS / "wide-uci" / key
            for name, expected in files.items():
                assert digest(fold / name) == expected == hashlib.sha256(models.read(f"{key}/{name}")).hexdigest()
            layout = read(fold / "layout.json")
            assert digest(fold / "targets.npz") == layout["targets_sha256"]
            assert labels.read(f"{key}/targets.npz") == (fold / "targets.npz").read_bytes()
            info = read(fold / "selection.json")
            with np.load(fold / "targets.npz") as target, np.load(fold / "predictions.npz") as preds:
                np.testing.assert_array_equal(target["test_row"], preds["test_row"])
                anchor = wide_audit.metrics(target["y"], preds["anchor"], info["choices"]["gain"]["rare"])
                for arm, choice in info["choices"].items():
                    row = published.loc[layout["dataset"], layout["fold"], arm]
                    values = wide_audit.metrics(target["y"], preds[arm], choice["rare"])
                    np.testing.assert_allclose([row[k] for k in values], list(values.values()), rtol=0, atol=1e-12)
                    breach = any(values[k] + 1e-12 < v - abs(v) * (.05 if k == "ap" else .03)
                                 for k, v in anchor.items())
                    assert breach == row.breach
                    assert np.isclose((1 - len(choice["selected"]) / len(choice["anchor"])) * (not breach), row.u)


def historical_report():
    import numpy as np
    import pandas as pd
    from experiments.context_review import refit_opportunity, wide_audit

    eligible = pd.read_csv(RUNS / "refit-guard/eligibility.csv")
    jobs = [(r.dataset, r.fold) for r in eligible.itertuples() if r.eligible] + [("optdigits", 0)]
    records = []
    for task, number in jobs:
        fold = RUNS / "refit-opportunity" / task / f"fold-{number}"
        rec = read(fold / "complete.json")
        assert rec["predictions"] == digest(fold / "probabilities.npz")
        assert rec["source"] == digest(RUNS / "budgeted-guard" / task / fold.name / "shortlists.json")
        with np.load(fold / "probabilities.npz") as probs:
            for phase, scores, target in [("g", "guard_metrics", "guard_labels"),
                                         ("o", "outer_metrics", "outer_labels")]:
                for i, ref in enumerate(rec[scores]):
                    actual = wide_audit.metrics(probs[target], probs[f"{phase}{i}"], "ap" in ref)
                    np.testing.assert_allclose([actual[k] for k in ref], list(ref.values()), rtol=0, atol=1e-12)
        records.append((task, number, rec))
    refit_opportunity.report(records)


def replay_archived_tables():
    """Run in a temporary extraction that contains no predictor matrices or upstream datasets."""
    from functools import partial
    import pandas as pd
    from experiments.context_review import (
        ballot_refinement, candidate_concordance, chance_nomination, decision_opportunities, lottery_reference,
        low_fidelity_score, marginal_replay, matched_signal, receiver_transport, transport_mammographic,
        transport_score, ensemble_identification, filter_neighborhood_score, forest_replay, gain_increment_score,
        transcript_information, transcript_permutations, transcript_swaps, neighborhood_replicates_score,
        score_resolution_replay, score_resolution_check, ordinal_decomposition, matched_stopping,
        matched_stopping_audit,
        budget_comparison, class_precision, class_precision_check, joint_choice, joint_choice_check,
        plan_comparison, transfer_choice, transfer_choice_audit, within_tier, within_tier_audit,
    )

    def compare(path, frame):
        result = pd.read_csv(path, float_precision="round_trip")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), result, check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)

    verify_external_slates()
    verify_wide_fold_scores()
    matched = RUNS / "matched-signal-r2"
    for name, expected in read(EVIDENCE / "matched_signal/manifest.json")["outputs"].items():
        assert digest(matched / name) == expected, name
    predictions = pd.read_csv(matched / "predictions.csv", float_precision="round_trip")
    summary = matched_signal.analyze(predictions, read(matched_signal.DESIGN))
    for name, frame in summary.items():
        compare(matched / f"{name}.csv", frame)

    producers = [
        (transport_score.score, "transport", (
            "fold-results", "fold-manifest", "candidate-scores", "resolution", "task-results",
            "summary", "contrasts", "task-contrasts")),
        (transport_mammographic.score, "transport", ("birads-four-features", "birads-replaced-summary")),
        (low_fidelity_score.main, "low_fidelity", ("folds", "supports", "summary", "contrasts")),
        (historical_report, "opportunity", ("folds", "supports", "summary")),
        (receiver_transport.replay, "receiver", ("candidates", "folds", "tasks", "means", "contrasts")),
    ]
    for panel in ("first", "wide"):
        producers.append((partial(chance_nomination.report, panel), "chance",
                          [f"{panel}-{key}" for key in ("folds", "tasks", "means", "contrasts")]))
        producers.append((partial(ballot_refinement.score, panel), "refinement",
                          [f"{panel}-{key}" for key in ("folds", "means", "contrasts")]))
    producers.append((lottery_reference.main, "chance", ("finite-lottery",)))
    producers.append((decision_opportunities.main, "opportunities",
                      ("folds", "tasks", "means", "contrasts", "fold-envelopes", "envelopes")))
    producers.append((marginal_replay.replay, "marginal",
                      ("costs", "agreement", "folds", "tasks", "means", "contrasts")))
    producers.append((candidate_concordance.main, "concordance",
                      ("candidates", "folds", "tasks", "summary", "contrasts")))
    producers.extend((
        (ensemble_identification.main, "ensemble_cells", ("metric-sets", "summary")),
        (transcript_permutations.replay, "transcript_permutations", ("proposals", "summary")),
        (transcript_information.replay, "transcript_information", ("proposals", "summary")),
        (gain_increment_score.main, "gain_increment",
         ("folds", "tasks", "means", "paired-folds", "paired-tasks", "contrasts", "costs")),
        (forest_replay.replay, "forest_cross", ("policies", "tasks", "means", "contrasts", "rank-folds",
         "rank-tasks", "rank-means", "rank-contrasts", "score-collisions", "candidates", "costs")),
        (filter_neighborhood_score.replay, "filter_neighborhood",
         ("bounds", "candidates", "contrasts", "costs", "fold-bounds", "means", "policies", "task-bounds", "tasks", "ties")),
        (neighborhood_replicates_score.replay, "neighborhood_replicates",
         ("policies", "candidates", "overlap", "draw-tasks", "draw-means", "draw-contrasts", "tasks", "means", "contrasts")),
        (transcript_swaps.replay, "transcript_swaps", ("proposals", "tasks", "summary")),
        (score_resolution_replay.replay, "score_resolution", ("occupancy", "proposals", "tasks", "summary")),
        (ordinal_decomposition.replay, "ordinal_decomposition", ("folds", "task-contrasts", "summary")),
        (matched_stopping.replay, "matched_stopping", ("folds", "summary", "timing-summary")),
        (budget_comparison.replay, "budget_comparison", ("folds", "summary", "contrasts")),
        (class_precision.replay, "class_precision", ("mobility", "proposals", "tasks", "summary")),
        (joint_choice.replay, "joint_choice", ("pairs", "summary", "case-study", "attribution")),
        (within_tier.replay, "within_tier", ("policies", "opportunities", "fold-bounds", "tasks", "means",
                                             "contrasts", "envelopes", "ties-only")),
        (transfer_choice.replay, "transfer_choice", ("held", "summary")),
    ))

    # Remove the reference copies in the temporary extraction so each scorer must create its outputs.
    table_count = len(summary)
    for produce, directory, names in producers:
        paths = [EVIDENCE / directory / f"{name}.csv" for name in names]
        expected = {p: pd.read_csv(p, float_precision="round_trip") for p in paths}
        for path in paths:
            path.unlink()
        produce()
        for path, frame in expected.items():
            compare(path, frame)
        table_count += len(paths)
    resolution_audit = EVIDENCE / "score_resolution/audit.json"
    expected_checks = read(resolution_audit)
    resolution_audit.unlink()
    score_resolution_check.check()
    if read(resolution_audit) != expected_checks:
        raise ValueError("The independent finite-orbit verification changed")
    matched_stopping_audit.audit()
    within_tier_audit.verify()
    transfer_choice_audit.verify()
    for folder, verify in (("class_precision", class_precision_check.check), ("joint_choice", joint_choice_check.audit)):
        audit_path = EVIDENCE / folder / "audit.json"
        golden = read(audit_path)
        audit_path.unlink()
        verify()
        if golden != read(audit_path):
            raise ValueError(f"Independent verification disagreed in {folder}")
    design = plan_comparison.assess(plan_comparison.original_wide(), target_pp=5)
    if not design["target_ruled_out"] or abs(design["upper_pp"] - 100 / 36) > 1e-12:
        raise ValueError("The label-free design check disagrees with the original guarded supports")
    collisions = PAPER / "audit/refit-observation-collisions.csv"
    reference = pd.read_csv(collisions)
    collisions.unlink()
    runpy.run_path(str(PAPER / "audit/refit_observation_collisions.py"), run_name="__main__")
    compare(collisions, reference)
    table_count += 1
    for name, module in sys.modules.items():
        if name.split(".")[0] in {"experiments", "minppi_mofs"} and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise ValueError(f"Replay imported {name} from outside the extracted archive")
    print(f"Verified {table_count} archived tables, both masking learners, and the outer policy/fold scores.")


def verify_transport(archive):
    with TemporaryDirectory(prefix="minpi-transport-") as temp, ZipFile(archive) as pack:
        folder = Path(temp)
        pack.extractall(folder)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [folder / "src", folder])),
               "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_CACHE_DIR": str(folder / "cache"),
               "MPLCONFIGDIR": str(folder / "mpl")}
        command = [sys.executable, "-B", "-c", "import runpy; "
                   "runpy.run_path('thesis-paper/package_revision.py')['replay_archived_tables']()"]
        result = subprocess.run(command, cwd=folder, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout.decode(errors="replace"))
        print(result.stdout.decode(errors="replace").splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-evidence", "--robustness-evidence", "--transport-evidence", "--evidence",
                        action="store_true", help="Build the current supplement as thesis-decision-evidence.zip")
    parser.add_argument("--verify-transport", type=Path, metavar="ZIP",
                        help="Replay an existing transport evidence ZIP in an isolated extraction")
    parser.add_argument("--verify-source", action="store_true", help="Compile the ZIP alone and compare every page's text")
    args = parser.parse_args()
    if args.verify_transport:
        verify_transport(args.verify_transport)
        return
    pdf = PAPER/"minpi.pdf"
    archive = PAPER/"minpi-source.zip"
    copyfile(PAPER/"build/main.pdf", pdf)
    write_archive(archive, paper_source())
    if args.verify_source:
        verify_source(archive, pdf)
    if args.decision_evidence:
        release = DECISION
        PACKAGES.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix="transport-release-", dir=PACKAGES) as staging:
            candidate = Path(staging) / release.name
            write_archive(candidate, transport())
            verify_transport(candidate)
            candidate.replace(release)
    materials = [pdf, archive, PACKAGES/"thesis-pmlb-evidence-r1.zip",
                 PACKAGES/"thesis-context-revision-evidence.zip", REASSESSMENT, OPERATING, TRANSPORT, ROBUSTNESS, DECISION]
    entries = []
    for path in materials:
        if path.exists():
            item = dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                        sha256=digest(path))
            if path.suffix == ".zip":
                with ZipFile(path) as pack:
                    item["members"] = len(pack.namelist())
            entries.append(item)
    ARTIFACTS.write_text(json.dumps(entries, indent=2)+"\n", encoding="utf8")
    for item in entries:
        print(item["path"], item["sha256"])


if __name__ == "__main__":
    main()
