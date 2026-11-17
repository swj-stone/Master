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
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
PACKAGES = ROOT / "packages"
ARTIFACTS = PAPER / "evidence/revision/artifacts.json"
REASSESSMENT = PACKAGES/"thesis-refit-reassessment-evidence.zip"
OPERATING = PACKAGES / "thesis-operating-evidence.zip"
TRANSPORT = PACKAGES / "thesis-transport-evidence.zip"
ROBUSTNESS = PACKAGES / "thesis-robustness-evidence.zip"
DECISION = PACKAGES / "thesis-decision-evidence.zip"
SCREENING = PACKAGES / "thesis-minimum-vote-paper.zip"
COMPRESSION = PACKAGES / "thesis-minpi-compression.zip"
FROZEN_SCREENING = {
    PAPER / "supplement/minimum-vote-study.pdf":
        "1b9d50649a4c6724e0ba95612b60ec9b064f915de8dc7a09f50c13bc5597b20e",
    PAPER / "supplement/minimum-vote-study-source.zip":
        "d5985733b33e2bbb3f2947361486ca2a9bb471443000f64fd2e30bbe3626b4a9",
}
DECISION_SHA256 = "659b22b062baa1d6e69a78f59197100831b09c0b2a7384f8e23ed5407c4cfe41"
FROZEN_STUDY = {
    PAPER / "supplement/decision-study.pdf":
        "e128184e2e5e826b7e35a055be6e83e27a1e8a28511d2446882dc2076b321460",
    PAPER / "supplement/decision-study-source.zip":
        "2daeab06b5d36d8f982bdfd2de0c3b02da6c31eb8d58a8092fe7cb56ca448838",
}


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
    with TemporaryDirectory(prefix="pack-", dir=destination.parent) as staging:
        tmp = Path(staging) / destination.name
        with ZipFile(tmp, "w") as pack:
            for member, value in sorted(members.items()):
                # Fixed ZIP headers keep source archives stable across checkout times and platforms.
                info = ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                data = value if isinstance(value, bytes) else value.read_bytes()
                pack.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=3)
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


def compression_members(pdf, source):
    """Collect the measured outcomes, their original inputs and the local import closure."""
    for path, expected in {**FROZEN_STUDY, **FROZEN_SCREENING}.items():
        if digest(path) != expected:
            raise ValueError(f"Historical supplement differs from its release: {path}")
    suite = ROOT / "experiments/minpi_compression"
    evidence = EVIDENCE / "minpi_compression"
    files = {ROOT / p for p in (
        "pyproject.toml", "requirements.txt", "thesis-python.ps1",
        "configs/context_study/prospective.json", "configs/context_review/wide_panel.json",
        "thesis-paper/evidence/wide/sources.json", "experiments/minpi_compression/PMLB-LICENSE.txt",
        "docs/thesis/archives/piv1/protocol.json",
        "docs/thesis/archives/piv1/evidence.sha256", "docs/thesis/archives/piv1/final_results_by_seed.csv")}
    files.add(PAPER / "formal/lean-toolchain")
    files.update(p for p in suite.iterdir() if p.suffix in {".md", ".json", ".py"})
    files.update(p for p in evidence.rglob("*") if p.suffix in {".md", ".json", ".npz", ".csv", ".zip"})
    files.update(PAPER / "formal" / p for p in
                 ["SampledMarginal.lean", "SampledMarginal.wl", "Dictator.lean", "PartialUpper.lean"])
    files.update(PAPER / "audit" / p for p in ["minpi-compression-audit.md", "sampled-minpi-referee.md",
                                               "sampled-minpi-final.md", "minpi-area-chair.md"])
    files.update(ROOT / "tests" / p for p in
                 ["test_sampled_linear_minpi.py", "test_dictator_minpi.py", "test_minpi_chain_replay.py",
                  "test_minpi_greedy_rls.py"])
    files.update(ROOT / "data/minpi_compression" / (name + ".zip") for name in
                 ["glass", "raisin", "wifi", "leaves", "airfoil", "yacht", "slump", "energy",
                  "landsat", "biodeg", "aquatic", "power"])
    for name in ["ionosphere", "vehicle", "sonar"]:
        files.update(p for p in (ROOT / "data/context_study/pmlb" / name).iterdir() if p.is_file())
    layouts = sorted((RUNS / "wide-uci").glob("*/fold-*/layout.json"))
    if len(layouts) != 16:
        raise ValueError("Expected 16 wide-transfer folds")
    for layout in layouts:
        files.update([layout, layout.with_name("inputs.npz"), layout.with_name("targets.npz")])
    for phase, size in (("pilot", 20), ("confirmation", 40)):
        strict = ROOT / "results/minpi_compression/zero_tolerance" / phase
        lock = read(strict / "lock.json")
        if len(lock) != size or lock != read(evidence / "zero_tolerance" / phase / "lock.json"):
            raise ValueError(f"Unexpected sensitivity receipts for {phase}")
        files.add(strict / "lock.json")
        for key, checksum in lock.items():
            receipt, forecast = strict / key / "selection.json", strict / key / "predictions.npz"
            if digest(receipt) != checksum or digest(forecast) != read(receipt)["predictions"]:
                raise ValueError(f"Changed sensitivity forecasts for {key}")
            files.update([receipt, forecast])
    for panel in ["pilot", "confirmation", "wide-transfer", "chain-validation", "legacy-gain",
                  "legacy-gain/xgb-receiver"]:
        files.update(evidence / panel / p for p in ["forecasts.zip", "scores.csv"])
    include_imports(files)
    files.add(Path(__file__).resolve())
    members = {p.relative_to(ROOT).as_posix(): p for p in files}
    members["README.md"] = (suite / "release.md").read_text(encoding="utf8").replace("](../../", "](").encode()
    members["thesis-paper/minpi.pdf"] = pdf
    members["thesis-paper/minpi-source.zip"] = source
    piv1 = PACKAGES / "thesis-piv1-evidence.zip"
    if digest(piv1) != (ROOT / "docs/thesis/archives/piv1/evidence.sha256").read_text().split()[0]:
        raise ValueError("Prior predictions differ from the frozen PIv1 release")
    with ZipFile(piv1) as archive, BytesIO() as payload:
        names = sorted(n for n in archive.namelist() if "PIv1-all-seeds-" in n and
                       n.endswith(("/direct_gain/predictions.npz", "/PIv1/predictions.npz")))
        if len(names) != 60:
            raise ValueError("Expected the 60 original Gain and PIv1 forecast arrays")
        with ZipFile(payload, "w") as subset:
            for name in names:
                subset.writestr(ZipInfo(name), archive.read(name), compress_type=ZIP_DEFLATED)
        members["packages/piv1-original-forecasts.zip"] = payload.getvalue()
    manifest = {name: hashlib.sha256(p).hexdigest() if isinstance(p, bytes) else digest(p)
                for name, p in sorted(members.items())}
    members["release.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return members


def verify_compression(archive, rebuild_source=False):
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="minpi-replay-", dir=scratch) as staging, ZipFile(archive) as pack:
        dest = Path(staging)
        manifest = json.loads(pack.read("release.json"))
        if sorted(pack.namelist()) != sorted([*manifest, "release.json"]):
            raise ValueError("Unexpected or missing replication archive member")
        for name, expected in manifest.items():
            if not (dest / name).resolve().is_relative_to(dest):
                raise ValueError(f"Unsafe archive path: {name}")
            with pack.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Incorrect digest for {name}")
        pack.extractall(dest)
        if rebuild_source:
            verify_source(dest / "thesis-paper/minpi-source.zip", dest / "thesis-paper/minpi.pdf")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [dest / "src", dest])),
               "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_CACHE_DIR": str(dest / "cache"),
               "MPLCONFIGDIR": str(dest / "mpl")}
        subprocess.run([sys.executable, "-B", "-m", "experiments.minpi_compression.replay_release"],
                       cwd=dest, env=env, check=True)


def verify_screening(archive, predictions):
    """Reconstruct the new result tables from the pinned forest release and measured clocks."""
    scratch = ROOT / "tmp"
    scratch.mkdir(exist_ok=True)
    with ZipFile(archive) as short, ZipFile(predictions) as previous:
        manifest = json.loads(short.read("release.json"))
        if manifest["prediction_archive"]["sha256"] != DECISION_SHA256 or digest(predictions) != DECISION_SHA256:
            raise ValueError("Expected the original R13 prediction archive")
        entries = manifest["members_sha256"]
        if set(short.namelist()) != set(entries) | {"release.json"}:
            raise ValueError("Release contents differ from the manifest")
        for name, expected in entries.items():
            if hashlib.sha256(short.read(name)).hexdigest() != expected:
                raise ValueError(f"Changed short-release member: {name}")
        with TemporaryDirectory(prefix="vote-", dir=scratch) as temp:
            folder = Path(temp)
            previous.extractall(folder)
            short.extractall(folder)
            results = folder / "thesis-paper/evidence/vote_certificates"
            expected = {p: p.read_text(encoding="utf8") for p in (
                results / "folds.csv", results / "summary.csv", results / "timing-summary.csv")}
            for path in expected:
                path.unlink()
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [folder / "src", folder])),
                   "PYTHONDONTWRITEBYTECODE": "1", "NUMBA_CACHE_DIR": str(folder / "cache"),
                   "MPLCONFIGDIR": str(folder / "mpl")}
            for args in (["experiments.context_review.vote_certificates"],
                         ["experiments.context_review.vote_certificate_timing", "verify"]):
                proc = subprocess.run([sys.executable, "-B", "-m", *args], cwd=folder, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if proc.returncode:
                    raise RuntimeError(proc.stdout.decode(errors="replace"))
            for path, text in expected.items():
                if path.read_text(encoding="utf8") != text:
                    raise ValueError(f"Archived and recomputed {path.name} differ")
    print("Recomputed all three certificate and timing tables from the two releases.")


def record_artifacts(paths):
    # Keep the recorded digests of historical releases even in a checkout without local data archives.
    ledger = {row["path"]: row for row in read(ARTIFACTS)}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        item = dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
        if path.suffix == ".zip":
            with ZipFile(path) as pack:
                item["members"] = len(pack.namelist())
        ledger[name] = item
        print(name, item["sha256"])
    with TemporaryDirectory(dir=ARTIFACTS.parent) as staging:
        updated = Path(staging) / ARTIFACTS.name
        updated.write_text(json.dumps(list(ledger.values()), indent=2) + "\n", encoding="utf8")
        updated.replace(ARTIFACTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--screening-release", "--evidence", action="store_true",
                      help="Verify the archived R14 minimum-vote paper in read-only mode")
    mode.add_argument("--compression-release", action="store_true",
                      help="Publish the current feature-level MinPI paper and offline replication package")
    mode.add_argument("--verify-compression", type=Path, metavar="ZIP",
                      help="Replay the archived MinPI outcomes in a clean temporary extraction")
    for flag, path in (("--decision-evidence", DECISION), ("--robustness-evidence", ROBUSTNESS),
                       ("--transport-evidence", TRANSPORT)):
        mode.add_argument(flag, dest="frozen", action="store_const", const=path,
                          help=f"Verify the frozen {path.name} using its archived replay code")
    mode.add_argument("--verify-transport", type=Path, metavar="ZIP",
                      help="Replay an existing research evidence ZIP in an isolated extraction")
    mode.add_argument("--verify-screening", type=Path, metavar="ZIP",
                      help="Replay the short paper's certificate tables using the R13 prediction archive")
    parser.add_argument("--verify-source", action="store_true", help="Compile the ZIP alone and compare every page's text")
    args = parser.parse_args()
    if args.verify_compression:
        expected = next(row["sha256"] for row in read(ARTIFACTS)
                        if row["path"] == COMPRESSION.relative_to(ROOT).as_posix())
        if digest(args.verify_compression) != expected:
            raise ValueError("Compression archive does not match the release ledger")
        verify_compression(args.verify_compression, args.verify_source)
        return
    if args.screening_release:
        expected = next(row["sha256"] for row in read(ARTIFACTS)
                        if row["path"] == SCREENING.relative_to(ROOT).as_posix())
        if digest(SCREENING) != expected:
            raise ValueError(f"Archived release does not match the ledger: {SCREENING}")
        if args.verify_source:
            with TemporaryDirectory() as staging, ZipFile(SCREENING) as pack:
                folder = Path(staging)
                for name in ("minimum-vote.pdf", "minimum-vote-source.zip"):
                    (folder / name).write_bytes(pack.read(name))
                verify_source(folder / "minimum-vote-source.zip", folder / "minimum-vote.pdf")
        verify_screening(SCREENING, DECISION)
        return
    if args.verify_screening:
        verify_screening(args.verify_screening, DECISION)
        return
    if args.frozen:
        expected = next(row["sha256"] for row in read(ARTIFACTS)
                        if row["path"] == args.frozen.relative_to(ROOT).as_posix())
        if digest(args.frozen) != expected:
            raise ValueError(f"Archived release does not match the ledger: {args.frozen}")
        if args.verify_source:
            with TemporaryDirectory() as staging, ZipFile(args.frozen) as frozen:
                folder = Path(staging)
                for name in ("minpi.pdf", "minpi-source.zip"):
                    (folder / name).write_bytes(frozen.read("thesis-paper/" + name))
                verify_source(folder / "minpi-source.zip", folder / "minpi.pdf")
        verify_transport(args.frozen)
        return
    if args.verify_transport:
        verify_transport(args.verify_transport)
        return
    pdf = PAPER / "minpi.pdf"
    source = PAPER / "minpi-source.zip"
    outputs = [pdf, source]
    with TemporaryDirectory(prefix="paper-", dir=PAPER) as staging:
        folder = Path(staging)
        copyfile(PAPER / "build/main.pdf", folder / pdf.name)
        write_archive(folder / source.name, paper_source())
        if args.verify_source or args.compression_release:
            verify_source(folder / source.name, folder / pdf.name)
        if args.compression_release:
            PACKAGES.mkdir(exist_ok=True)
            members = compression_members(folder / pdf.name, folder / source.name)
            write_archive(folder / COMPRESSION.name, members)
            verify_compression(folder / COMPRESSION.name)
            outputs.append(COMPRESSION)
        for path in outputs:
            (folder / path.name).replace(path)
    record_artifacts([*outputs, *FROZEN_STUDY, *FROZEN_SCREENING])


if __name__ == "__main__":
    main()
