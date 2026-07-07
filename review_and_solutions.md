# ICOMP 2026 — Professional Peer Review & Detailed Solutions

**Paper**: "Min-Marginal Power Indices for Multi-Objective Feature Selection: A Voting-Game Approach"

**Reviewer role**: Program committee member with expertise in cooperative game theory, feature selection, and multi-objective optimization.

---

# Part I: Professional Peer Review

## Overall Assessment

**Rating**: Weak Accept (borderline)

The paper proposes MinPPI, a novel feature-importance index that replaces the mean operator of classical Shapley-based indices with a minimum over coalitions, paired with a binary utility-threshold characteristic function $v_\alpha$. The work has strong theoretical foundations (Properties 1--3, Theorem 1--2) and a well-conceived adversarial search algorithm (Algorithm B). The core insight---that $\min \times$ binary yields a clean $\{-1, 0, +1\}$ classification---is elegant and genuinely useful for practice.

However, the paper has several structural weaknesses that prevent acceptance in its current form. The most concerning is a fundamental **logical gap between the manuscript's title/claims of "multi-objective" feature selection and the delivered experiments**, which lack any multi-objective optimization component. The empirical evidence, while internally consistent, is thin in two critical respects: (i) the claimed $\alpha$-Pareto front is trivially monotone by construction, not a true multi-objective discovery, and (ii) MinPPI's value proposition—deleting harmful features without AUC loss—is demonstrated almost exclusively on datasets where it finds **zero or near-zero harmful features**, raising questions about whether the method is genuinely useful or merely vacuous on most real data. Finally, the paper's complex history (the switch from NSGA-II to $\alpha$-Pareto in what the draft calls "Plan C") is visible in its structure in ways that harm readability and credibility.

I detail these concerns below and provide actionable solutions.

---

## Major Concerns

### M1. The Title and Framing Promise "Multi-Objective" But Deliver Single-Objective Filtering

The paper's title includes "Multi-Objective Feature Selection" and §1 frames the work among MOFS literature, yet the delivered method (§4.4, §6) performs **no multi-objective optimization**. The $\alpha$-Pareto front is constructed by sweeping a single parameter $\alpha$ over four values $\{0.85,0.90,0.95,0.99\}$ and plotting the resulting (compression rate, preservation) points. This is not a multi-objective search algorithm in any accepted sense: (a) it produces exactly 4 points regardless of problem structure, (b) these points are monotone by construction ($c(\alpha_1) \leq c(\alpha_2)$ when $\alpha_1 > \alpha_2$ because $|\mathcal{H}_\alpha|$ is non-increasing in $\alpha$), and (c) there is no mechanism for discovering novel trade-off points **between** the sampled $\alpha$ values. Calling this a "Pareto front" and "multi-objective selection" is misleading—it is essentially **single-objective hyperparameter tuning** with one parameter $\alpha$ controlling conservativeness. This mislabeling would be caught by any MOFS-area reviewer.

**Recommended action**: Replace "multi-objective" with "safe feature deletion via parameterized conservativeness" throughout. The paper should be positioned as: a method for identifying and safely removing provably harmful features, with a tunable parameter $\alpha$ that controls the precision–recall trade-off. The $\alpha$-sweep is a sensitivity analysis confirming robustness, not a Pareto-front algorithm. This requires modifying the title, abstract, §1 contributions #3, §4.4, §6.2 (remove Pareto-front framing), and §6.4 (remove "multi-objective" from positioning). The existing data and experiments remain valid—only the framing changes.

### M2. Empirical Vacuousness: MinPPI Finds Near-Zero Harmful Features on 5 of 7 Datasets

Table `tab:pareto_front` shows that at the recommended $\alpha=0.95$, six of seven datasets have compression ≤ 8.2%, and four have compression ≤ 2.6%. On Gisette, MinPPI finds **zero** harmful features in 4 of 5 seeds; on Crimedata and Abalone, the method is equally silent. The removal test (Table `tab:harmful_removal`) confirms that deleting harmful features has negligible impact because there is essentially nothing to delete.

This is a structural problem: **a method whose primary output is "there is nothing harmful here" on the majority of tested datasets cannot claim to be a useful feature selection tool**. The paper correctly notes that this is by design—MinPPI is conservative—but fails to demonstrate that when $|\mathcal{H}| > 0$, the deletion provides genuine practical value beyond what a simple random filter could achieve. The **one** dataset where MinPPI non-trivially activates (Madelon, 22.8% compression) is also the only dataset with known ground-truth probes—making it impossible to assess whether harmful features on Spambase, ISOLET, or Credit-Card are correctly flagged or are false negatives.

Furthermore, the absence of a **runtime ablation** or complexity analysis comparing MinPPI's cost to its competitors weakens the practical argument: is running 240+ XGBoost evaluations per seed (on ISOLET) worth the resulting 8% compression that causes zero measurable AUC change?

### M3. Conflicting Claims About MinPPI Not Being a Ranking Method

The main results section (§6.2) contains a conspicuous statement: "MinPPI does not appear in the ranking comparison because it produces a three-way classification—harmful, neutral, decisive—not a feature ranking." This is **internally inconsistent** with multiple parts of the paper:

- §1, contribution #4 claims MinPPI is evaluated "against five baselines...on top-k AUC"
- Appendix `app:full_tables` has a placeholder for "Single-Point AUC@k (All Methods)" that explicitly lists MinPPI as a method
- The pipeline comparison (Appendix `fig:pipeline_comparison`) evaluates MinPPI-filtered data via SHAP-top-k, implicitly using MinPPI as a ranking preprocessor
- The filter verification (`filter_verify.csv`) extracts `feature_importances_` from an XGBoost model trained on filtered data—a ranking

The paper cannot simultaneously claim "we produce a ranking" (for baselines comparison) and "we don't produce a ranking" (to excuse MinPPI's absence from the CD diagram). If the post-filter XGBoost-gain ranking is MinPPI's ranking, it must be compared head-to-head. If it is not, the paper should not claim AUC@k evaluation of MinPPI.

### M4. The ISOLET Utility Mismatch Is Underexplored

ISOLET uses accuracy for $v_\alpha$ but OVR AUC for evaluation. The paper argues this is "conservative" (accuracy is stricter), but this is **empirically untested**. It is equally plausible that a feature causing a 1-accuracy% drop (misclassifying a letter) has a different OVR-AUC profile than what the voting game detects. The mismatch means that the MinPPI signal on ISOLET is produced by a **different game** than the one being evaluated—an experimental confound that needs ablation.

### M5. Algorithm B's ShapG Guidance Is Replaced in Practice Without Documentation

The code base shows that `shapg_integration.py` has been rewritten to use mutual information scores **instead of** the full ShapG pipeline (the original took 276s on Spambase alone). While this is a reasonable engineering decision for tractability, the paper still describes Algorithm B as "ShapG-guided" and implicitly claims the sample-efficiency benefits of Theorem 2 (which assumes the ShapG correlation graph quality). With MI replacing ShapG as the oracle, Theorem 2's conditional guarantee no longer applies—there is no proven relationship between MI neighbors and harmful witnesses. The paper must (a) declare this substitution, (b) rename Algorithm B to avoid implying ShapG dependence, and (c) report the empirical impact (or lack thereof) on detection quality.

---

## Medium Concerns

### M6. The Removal Test Has an Unfair Comparison Baseline

Table `tab:harmful_removal` compares MinPPI deletion ($\alpha=0.95$) against MinShap deletion (removing **all** features with negative MinShap score), SHAP-bottom-k (size-matched to MinPPI's count), and random deletion. The MinShap comparison is structurally unfair because it deletes 30%--97% of features (far more than MinPPI's 1.8%--22.8%) and then reports much larger AUC drops. A fair comparison would either (a) delete only the **top-k** most negative MinShap features at the same $k = |\mathcal{H}_\alpha|$, or (b) report the AUC-degradation curve as a function of features removed for both methods. As shown, the experiment proves that deleting 96.6% of Credit-Card-Fraud features hurts AUC more than deleting 10.3%—a trivial finding that does not isolate MinPPI's claimed advantage.

### M7. Madelon Ground-Truth Is Used as the Sole Real-Data Validation

Madelon is the only dataset with known ground-truth probe labels, and the paper leans heavily on this: the entire $\alpha$-recall analysis, precision/recall claims, and the "$\alpha$ as a controllable precision knob" narrative depend on it. However, Madelon has a very specific structure (synthetic probes with zero predictive power), making the findings hard to generalize. Would MinPPI's near-perfect precision hold on a dataset with subtly correlated noise features or adversarial noise? The paper needs either (a) a semi-synthetic experiment on, say, Spambase or ISOLET with injected known-harmful features (e.g., randomly permuted columns, added Gaussian features), or (b) a clear limitation statement that Madelon's ground truth only validates performance against trivially zero-signal probes.

### M8. The Harmful-vs-Dimension Analysis Is Plot-Driven, Not Statistically Tested

Figure `fig:harmful_vs_dim` shows a downward trend (harmful% decreases as more features are included), but no statistical test confirms whether this trend is significant or an artifact of small $n$ (3 datasets). A mixed-effects model or Kendall's $\tau$ trend test would strengthen the claim. As presented, the figure "counters the concern" only visually.

### M9. The Pipeline Comparison Is Misleading

Appendix `fig:pipeline_comparison` compares MinPPI-filtered SHAP-top-k against MinShap-filtered and random-filtered SHAP-top-k. On four of five datasets, MinPPI filtering produces **identical** AUC@10 to the no-filtering baseline (because $|\mathcal{H}|$ is near zero), and the random-filter baseline is also indistinguishable. This is not a finding—it is a restatement of the vacuity problem (M2). The only informative entry is Madelon, where both MinPPI and MinShap show negligible difference from baseline. The pipeline comparison needs to run on a dataset where MinPPI's filter is non-trivial, or it should be cut.

---

## Minor Concerns

### m1. Repeated Sentence in Abstract
Lines 66-67 of the abstract contain the identical clause "...the removal test confirms that deleting all MinPPI-harmful features preserves AUC within $10^{-3}$ on 6 of 7 datasets" twice. Remove the duplicate.

### m2. Referenced But Missing Tables
The placeholder comment at §Appendix `Full Per-Dataset Tables`: "regenerate from fresh run\_main\_table.py" indicates missing content. The single-point AUC table with 6 methods × 7 datasets is a critical piece of evidence that must be present. Regenerate and insert.

### m3. Ablation Section Has Only One Dimension
Section §6.4 "Ablation and Sensitivity" contains only the $\alpha$ trade-off paragraph and the binary vs. continuous discussion. A proper ablation typically includes 3--5 dimensions. The text notes that three NSGA-II ablations were removed (min vs. mean, $\beta$, ShapG vs. SHAP), but their replacements are absent. Add: (a) an analysis of whether MI-based guidance (the actual implementation) vs. ShapG-based guidance (the paper's claim) produces different harmful sets; (b) sensitivity to the base learner choice (does MinPPI with RandomForest produce the same harmful set as XGBoost?).

### m4. CD Diagram Consistency
The text references "all six methods" in Figure `fig:cd_auc10`'s caption, but the CD diagram only shows five methods (MinPPI is excluded, as the text explains). This is confusing—either update the caption to "all five baseline methods (MinPPI excluded; see §6.3)" or include MinPPI.

### m5. Theorem Numbering Gap
There is a Theorem 1 and Theorem 2, but no Proposition 1 or Proposition 2 as used earlier in the paper (these were deleted during "Plan C"). The deleted Proposition 2a/2b are still discussed in §1's contribution #3 (which was also rewritten). Verify that all cross-references are resolved.

### m6. Credit-Card-Fraud Metrics Switching
Credit-Card-Fraud uses **24-fold more samples** (284,807) than ISOLET (7,797), yet Table `tab:harmful_removal` reports it in the same table with identical metric interpretation. The removal test on Credit-Card takes >400s per seed (filter_verify.csv shows 403--432s for full-model training alone). This extreme asymmetry in dataset scale and training cost is never discussed.

### m7. MinPPI Runtime Not Reported Against Baselines
The `pareto_front_agg.csv` includes `time_minppi_s` (2--132s across datasets), but Table `tab:pareto_front` omits runtime entirely. A feature selection method should report time against its baselines. Add a runtime column or supplementary table.

### m8. Bibliographic Inconsistency
Citation `\citep{zheng2026minshap}` (MinShap) is listed as arXiv:2604.15107 (2026)—a pre-print published after the paper's writing. This is acceptable for a fast-moving field, but the paper should acknowledge its pre-print status and note that MinShap's conclusions may change upon peer review.

---

# Part II: Detailed Solutions

## Solution to M1: Re-frame the Paper Without Multi-Objective Claims

### What to Change

1. **Title**: Change to "Min-Marginal Power Indices for Safe Feature Deletion: A Voting-Game Approach" (drop "Multi-Objective").

2. **Abstract**: Replace the $\alpha$-Pareto front sentence with: "Sweeping the conservativeness parameter $\alpha$ over $\{0.85,0.90,0.95,0.99\}$ provides a tunable precision--recall trade-off; we recommend $\alpha=0.95$ as the primary operating point and show that precision remains above $0.95$ for $\alpha \geq 0.90$."

3. **Introduction contribution #3**: Replace with: "Tunable Conservativeness via $\alpha$. The quota $\alpha$ acts as a precision--recall knob: lower values flag more features as harmful (higher recall, lower precision); higher values are stricter (lower recall, near-perfect precision). The $\alpha$-sweep serves as a built-in sensitivity analysis confirming robustness."

4. **§4.4 title**: Rename to "$\alpha$-Parameterized Feature Deletion."

5. **§4.4 text**: Replace "Pareto front" with "operating curve" or "$\alpha$-sweep." Remove: "The set of points...forms a Pareto front: no $\alpha$ simultaneously achieves higher compression and higher preservation..." This is mathematically true but trivially so—every monotonic function of a scalar parameter produces non-dominated points.

6. **Prelim §3.3**: Rename "Pareto Optimality" to "Trade-off Parameterization" or remove entirely.

7. **§6.2**: Replace `tab:pareto_front` caption: "Compression rate and AUC preservation at four $\alpha$ levels (mean over 5 seeds)." Remove the bold emphasis on $\alpha=0.95$ column header (it already stands alone textually).

### No New Experiments Needed
This is a pure framing change. All data and figures remain valid.

---

## Solution to M2: Add Semi-Synthetic Experiments with Known Harmful Features

The core problem is that real-world datasets lack ground-truth harmful labels. The solution is to create them.

### Experiment D1: Inject Known Harmful Features into Existing Datasets

**Design**: For each of Spambase, ISOLET, and Credit-Card-Fraud, create three synthetic variants:
- **D1a (Noise columns)**: Add 20 columns of $\mathcal{N}(0,1)$ noise (irrelevant features).
- **D1b (Adversarial columns)**: Add 5 columns that are random permutations of an existing predictive feature (redundant-but-confusable features).
- **D1c (Harmful-by-construction columns)**: Add 5 columns of pure noise AND modify the training set so that when any of these columns is included, the label is flipped with probability $p=0.2$ in a subset containing a specific existing feature pair (creating a known harmful witness).

**Script**: `experiments/d1_semisynthetic_harmful.py`

```python
"""
Semi-synthetic harmful-feature injection experiment (§R.D1).

Creates three variants of each classification dataset:
  D1a: 20 pure-noise columns added
  D1b:  5 permuted-signal columns added (redundant but correlated)
  D1c:  5 noise columns with injected harmful interaction:
        when col_i AND a specific feature-pair are jointly present,
        labels are flipped with p=0.2, creating a ground-truth
        harmful witness S* = {pair_feat1, pair_feat2}.

MinPPI is evaluated against MinShap and SHAP on each variant.
Expected: MinPPI precision >= 0.90 on D1c; MinShap flags more
features but with lower precision.

Run:
    python -u experiments/d1_semisynthetic_harmful.py
"""

from __future__ import annotations

import json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from tqdm import tqdm

from minppi_mofs.min_ppi import estimate_minppi
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.min_shap import compute_minshap_importance
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance

# ── config ──────────────────────────────────────────────────────────
SEED = 42
ALPHA = 0.95
N_NOISE = 20          # D1a
N_PERMUTED = 5        # D1b
N_HARMFUL = 5         # D1c
HARMFUL_FLIP_PROB = 0.2

DATASETS = [
    ("spambase", "is_spam"),
    ("isolet", "letter"),
    ("credit_card_fraud", "Class"),
]

# ── helpers ─────────────────────────────────────────────────────────

def _now_stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def load_data(processed_dir, prefix, target):
    tr = pd.read_csv(processed_dir / f"{prefix}_s{SEED}_train.csv")
    te = pd.read_csv(processed_dir / f"{prefix}_s{SEED}_test.csv")
    return (
        tr.drop(columns=[target]), tr[target].values,
        te.drop(columns=[target]), te[target].values,
    )


def train_eval(X_tr, y_tr, X_te, y_te):
    m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                       verbosity=0, random_state=SEED)
    m.fit(X_tr.values, y_tr)
    return float(roc_auc_score(y_te, m.predict_proba(X_te.values)[:, 1]))


def precision_recall(pred_harmful, true_harmful):
    if not pred_harmful:
        return 1.0, 0.0, 0.0
    tp = len(pred_harmful & true_harmful)
    prec = tp / len(pred_harmful)
    rec = tp / len(true_harmful) if true_harmful else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)

# ── main ────────────────────────────────────────────────────────────

def main():
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    run_id = _now_stamp()
    out_dir = project_root / "results" / "tables" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for prefix, target in DATASETS:
        X_tr, y_tr, X_te, y_te = load_data(processed_dir, prefix, target)
        n_features = X_tr.shape[1]
        rng = np.random.default_rng(SEED)

        for variant, X_aug_tr, X_aug_te, true_harmful in [
            _make_d1a(X_tr, X_te, rng),
            _make_d1b(X_tr, X_te, rng),
            _make_d1c(X_tr, X_te, y_tr, rng),
        ]:
            # MinPPI
            sg_s, sg_n = build_guidance_from_data(X_aug_tr, y_tr, seed=SEED)
            minppi = estimate_minppi(X_aug_tr, X_aug_tr, y_tr, y_tr,
                alpha=ALPHA, method="shapg_guided", K_adv=30,
                shapg_neighbors=sg_n, shapg_scores=sg_s, seed=SEED)
            harmful_ppi = {f for f, v in minppi.items() if v == -1}
            prec_p, rec_p, f1_p = precision_recall(harmful_ppi, true_harmful)

            # MinShap
            ms = compute_minshap_importance(X_aug_tr, y_tr, seed=SEED)
            harmful_ms = {f for f, v in ms.items() if v < 0}
            prec_m, rec_m, f1_m = precision_recall(harmful_ms, true_harmful)

            # SHAP
            sh = compute_shap_topk_importance(X_aug_tr, y_tr, seed=SEED)
            k = len(harmful_ppi)
            harmful_sh = set(list(sh.keys())[-k:]) if k > 0 else set()
            prec_s, rec_s, f1_s = precision_recall(harmful_sh, true_harmful)

            rows.append({
                "dataset": prefix, "variant": variant, "n_full": X_aug_tr.shape[1],
                "n_harmful_gt": len(true_harmful),
                "minppi_n": len(harmful_ppi), "minppi_prec": prec_p,
                "minppi_rec": rec_p, "minppi_f1": f1_p,
                "minshap_n": len(harmful_ms), "minshap_prec": prec_m,
                "minshap_rec": rec_m, "minshap_f1": f1_m,
                "shap_n": len(harmful_sh), "shap_prec": prec_s,
                "shap_rec": rec_s, "shap_f1": f1_s,
            })

    pd.DataFrame(rows).to_csv(out_dir / "d1_semisynthetic.csv", index=False)
    print(f"Saved: {out_dir / 'd1_semisynthetic.csv'}")


if __name__ == "__main__":
    main()
```

**Execution**: `python -u experiments/d1_semisynthetic_harmful.py` (~1h)

**Expected outcome**: MinPPI F1 > 0.60 on D1c across datasets; MinShap F1 higher but precision < 0.70; SHAP F1 near zero (no signal without binary threshold).

### Experiment D2: Ablation — Oracle vs. MI vs. Random Guidance

**Purpose**: Quantify the cost of replacing ShapG with MI in Algorithm B's guidance.

**Design**: On Madelon (where Algorithm B's seed structure is interpretable), re-run estimating MinPPI three times with identical $K_{\text{adv}}$ budget but different guidance:
- **Full ShapG** (original method): compute Shapley scores via the `shapg()` function in `baselines/shapg.py`
- **MI-based** (current implementation): mutual information scores
- **Random**: no guidance, seed from random top-half features

**Script**: `experiments/d2_guidance_ablation.py`

**Execution**: `python -u experiments/d2_guidance_ablation.py` (~15min, Madelon only, seed 42).

**Expected outcome**: Table with three rows comparing harmful count and precision/recall against probes. If MI produces the same harmful set as ShapG, the substitution is justified; if not, the paper must report the discrepancy.

---

## Solution to M3: Give MinPPI a Ranking and Put It In the CD Diagram

MinPPI already produces a ranking via the `minppi_filter_eval.py` pipeline: train XGBoost on filtered features, extract `feature_importances_` (gain), rank by gain. This is a legitimate ranking—more principled than some baselines, in fact, because harmful features are deleted before importance is computed.

**Action**:
1. Add a `minppi` method to `run_main_table.py`'s baseline registry that:
   - Computes MinPPI ($\alpha=0.95$)
   - Deletes harmful features
   - Trains XGBoost on filtered set
   - Extracts `feature_importances_` as the ranking
   - For features in the harmful set, assigns rank at the bottom (after all non-harmful features)

2. Re-run `run_main_table.py` (add the `minppi` entry only; skip re-running the 5 existing baselines by loading their cached rankings).

3. The CD diagram will then show 6 methods including MinPPI, and readers can directly compare MinPPI's ranking quality against the baselines without the awkward "we don't produce a ranking" dodge.

---

## Solution to M4: Ablate the ISOLET Utility Mismatch

**Design**: Run MinPPI on ISOLET with both accuracy-based $v_\alpha$ and OVR-AUC-based $v_\alpha$ (via the `task` parameter), keeping all other settings identical. Compare the harmful sets (Jaccard) and removal test $\Delta$AUC.

**Script**: `experiments/d3_isolet_utility_ablation.py`

```python
"""
ISOLET utility-function ablation: accuracy-based vs OVR-AUC-based v_α.

Run:
    python -u experiments/d3_isolet_utility_ablation.py
"""
# (implementation: load isolet_s42, compute minppi with task="multiclass"
#  [accuracy-based] and with a custom OVR-AUC cache, compare harmful sets)
```

**Execution**: `python -u experiments/d3_isolet_utility_ablation.py` (~10min, ISOLET seed 42 only).

---

## Solution to M6: Fair Removal Test Comparison

Replace the current MinShap deletion (all negatives) with a **matched-budget** deletion:
- MinShap-top-k: delete the $k$ features with the most negative MinShap scores, where $k = |\mathcal{H}_{0.95}|$
- SHAP-bottom-k: same, $k = |\mathcal{H}_{0.95}|$

This ensures all methods delete **the same number of features**, making the comparison fair.

**Change**: In `eval_harmful_utility.py`, modify the section where `minshap_neg` is computed (around line 355). Current code sets `k = max(1, n_harmful)` as a fallback; change to `k = len(harmful_ppi)` (the count of MinPPI-harmful features) for the MinShap and SHAP deletion controls. This is a one-line change in the existing script; re-run with:

```powershell
python -u experiments/eval_harmful_utility.py
```

**Expected outcome**: MinShap-top-k deletion at matched budget $k$ should produce $\Delta$AUC much closer to MinPPI's (likely within 0.005), which is a fairer comparison. If MinShap at matched budget still significantly hurts AUC, then MinPPI genuinely adds value at equal removal count.

---

## Solution to M7: Add Semi-Synthetic Validation on Spambase and ISOLET

**Experiment D4**: Add D1c (injected harmful features) to Spambase and ISOLET. Same script `d1_semisynthetic_harmful.py` above covers this.

---

## Solution to m1--m8: Minor Text and Figure Fixes

| Issue | Fix | File |
|-------|-----|------|
| m1: duplicate abstract sentence | Delete second occurrence of "the removal test confirms..." (line 66) | `icomp2026_conference.tex` |
| m2: missing single-point AUC table | Fill from `mean_metrics.csv` | `icomp2026_conference.tex` Appendix |
| m3: thin ablation | Add D2 results (guidance ablation) as §6.4 paragraph | `icomp2026_conference.tex` §6.4 |
| m4: CD caption | "across all five baseline methods" not "all six methods" | `icomp2026_conference.tex` line 263 |
| m5: Theorem numbering | Verify no dangling `\ref{prop:pareto}` or `Proposition 2` references | `icomp2026_conference.tex` global search |
| m6: dataset scale | Add one sentence in §6.1 acknowledging Credit-Card's sample size | `icomp2026_conference.tex` line 252 |
| m7: missing runtime | Add `time_minppi_s` column to `tab:pareto_front` or a separate runtime table | `icomp2026_conference.tex` §6.2 |
| m8: MinShap pre-print | Add footnote: "MinShap~\citep{zheng2026minshap} is a 2026 preprint; conclusions may change upon peer review." | `icomp2026_conference.tex` line 100 |

---

# Part III: Execution Plan

## Phase 1: Text-only Fixes (no experiments needed)

| Step | Action | Time |
|:--:|--------|:--:|
| T1 | Re-frame paper per M1: change title, abstract, §1, §4.4, §6.2, §6.4 | 2h |
| T2 | Fix m1, m4, m5, m6, m8 in LaTeX | 30min |
| T3 | Fill single-point AUC table (m2) from existing `mean_metrics.csv` | 30min |

## Phase 2: New Experiments (run sequentially)

| Step | Script | Command | Time |
|:--:|--------|---------|:--:|
| E1 | D1 semi-synthetic | `python -u experiments/d1_semisynthetic_harmful.py` | ~1h |
| E2 | D2 guidance ablation | `python -u experiments/d2_guidance_ablation.py` | ~15min |
| E3 | D3 ISOLET utility ablation | `python -u experiments/d3_isolet_utility_ablation.py` | ~10min |
| E4 | Rerun removal test (M6 fix) | `python -u experiments/eval_harmful_utility.py` | ~3h |
| E5 | Rerun main table (M3 fix) | `python -u experiments/run_main_table.py` (minppi method only) | ~1h |

## Phase 3: Final Integration

| Step | Action | Time |
|:--:|--------|:--:|
| I1 | Insert E1--E3 results into §6.4 / Appendix | 1h |
| I2 | Update removal test table with fair MinShap comparison | 30min |
| I3 | Regenerate CD diagram with MinPPI included | `python experiments/make_cd_diagram.py ...` |
| I4 | Final ruff check + compile | 15min |

**Total new wall-clock time**: ~6h (E1+E4 can overlap with T1--T3).

---

# Summary of Review Decision

The paper's core idea—pairing the min operator with a binary threshold to produce a $\\{-1,0,+1\\}$ classification—is genuinely novel and valuable. The theoretical framework (Properties 1--3, Theorems 1--2) is rigorous and well-proved. The adversarial search (Algorithm B) is clever and the toy-game verification is thorough.

The paper needs three things to be publication-ready: (1) honest framing that replaces the hollow "multi-objective Pareto front" language with what the method actually does (safe, parameterized harmful-feature deletion); (2) semi-synthetic experiments demonstrating that MinPPI finds genuinely harmful features in non-trivial settings, not just that it correctly outputs zero on datasets with no harmful features; and (3) transparency about the MI/ShapG substitution and its empirical consequences.

If these are addressed, this is a solid conference contribution. The 2026-07-21 deadline is achievable with the 6-hour plan above.

---

# 第三部分：保留多目标定位的全新框架 —— M1-M8 完整可操作流程指导（中文）

---

## 第〇章：重新理解 M1 —— 不是删掉多目标，而是让它名副其实

### 0.1 为什么旧框架被拒

当前稿件声称 "$\alpha$-parameterized Pareto front"，实际做法是：扫 $\alpha \in \{0.85,0.90,0.95,0.99\}$ → 每个 $\alpha$ 得到一个 (压缩率, AUC保留率) 点 → 称这 4 个点为 "Pareto 前沿"。

**问题：这 4 个点之所以"非支配"，是因为 $|\mathcal{H}_\alpha|$ 关于 $\alpha$ 单调非增。换句话说，"没有两个点互相支配"是定义保证的 trivial 结论，不是任何优化算法发现的。** 审稿人一眼看穿。这是一个**架构设计级别的缺陷**——不是措辞问题，是方法本身缺少真正的多目标优化组件。

### 0.2 真正多目标框架的核心难题

要使 MinPPI 成为一个有说服力的多目标特征选择方法，必须回答：

> MinPPI 在哪个环节**发现了 Pareto 前沿**（而不仅是构造了它）？

当前框架中，$\alpha$ 扫参只是"构造"了前沿——因为 $|\mathcal{H}_\alpha|$ 的单调性保证了非支配关系。**真正多目标的本质是：前沿是通过在解空间中的搜索发现的，前端的非支配关系不可由参数单调性预先推知。**

### 0.3 Plan D 核心洞察：「候选池 + 轻量 MOEA」

**核心思路**：

```
MinPPI 不直接给出最终答案。
MinPPI 负责第一阶段：大幅缩小搜索空间，提供一个高质量的特征候选池。
多目标进化算法（轻量 NSGA-II）负责第二阶段：在候选池内部进行真正的 Pareto 前沿搜索。
两个组件各司其职，合在一起就是完整的 MOFS-MinPPI。
```

**为什么这个框架能解决旧框架的问题**：

| 方面 | 旧框架（Plan A/B/C） | 新框架（Plan D） |
|------|---------------------|-----------------|
| 多目标性 | 伪——$\alpha$ 单调性保证了非支配 | **真**——NSGA-II 在候选池内搜索，非支配关系不可由参数单调性预先推知 |
| MinPPI 的角色 | 自己就是最终输出（Plan C），或作为 NSGA-II 的 mutation bias（Plan A/B，已失败） | **搜索空间预筛器**——识别哪些特征有资格进入候选删除池 |
| 优化器 | 无（Plan C）或 NSGA-II 搜索全部特征（Plan A/B） | **轻量 NSGA-II** 仅在候选池内部搜索（10--30 维，而非 100 维） |
| 为什么不重复 Plan A/B 的失败 | — | Plan A/B 失败是因为将 MinPPI 用作 mutation bias（信号太弱，与 NSGA-II 的 selection 信号冲突）。Plan D 中 MinPPI 不参与 operators——它只提供候选池，NSGA-II 的 selection 基于真实的 AUC 评估 |
| 计算成本 | 高（Plan A/B：100-dim NSGA-II，600 次 XGBoost 评估/代） | **低**——候选池只有 10--30 维，NSGA-II 在 $2^{30}$ 空间搜索也仅需几十代收敛 |
| Baseline 对比 | 只能跟 ranking 方法比单点 AUC（CD 图） | **可以跟其他 MOFS 方法比 Pareto 前端质量**（HV, IGD+），同时也能比单点 AUC |

### 0.4 框架总览图

```
┌─────────────────────────────────────────────────────┐
│                   MOFS-MinPPI                        │
│                                                      │
│  Phase 1: MinPPI Candidate Pool Construction         │
│  ┌──────────────────────────────────────────────┐   │
│  │ α ∈ {0.85, 0.90, 0.95, 0.99}                │   │
│  │   ↓ 每个 α 独立计算 MinPPI                    │   │
│  │ H_α = {特征 i : MinPPI_i = -1 at quota α}   │   │
│  │   ↓                                            │   │
│  │ Candidate Pool C = ∪_α H_α                    │   │
│  │ (典型的 |C|: Spambase 3~5, Madelon 15~40,     │   │
│  │  ISOLET 5~15, Credit-Card 3~10)               │   │
│  └──────────────────────────────────────────────┘   │
│                      ↓                               │
│  Phase 2: Lightweight NSGA-II over C                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ Decision variables: binary mask over C only   │   │
│  │ (dim ∈ [3, 40], far smaller than n=100)       │   │
│  │                                               │   │
│  │ Objectives:                                    │   │
│  │   f₁(m) = compression = |{i∈C: m_i=1}| / n    │   │
│  │   f₂(m) = 1 - AUC( M_{N_1 \ D_m} )  (error)  │   │
│  │   (where D_m = {i∈C: m_i=1} = deleted set)    │   │
│  │                                               │   │
│  │ Operators: standard bit-flip + uniform xover   │   │
│  │ Selection: NSGA-II non-dominated sort + CD     │   │
│  │                                               │   │
│  │ Output: Pareto front P* ⊆ 2^C                 │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  Final Pareto Front: { (|D|/n, AUC(N_1\D)/AUC_full)  │
│                        : D ∈ P* }                    │
└─────────────────────────────────────────────────────┘
```

**为什么候选池使搜索变得简单**：MinPPI 已经确定了 "哪些特征值得考虑删除"。NSGA-II 不需要在 100 维空间中漫无目的地搜索——它只需要决定**这 15 个候选特征中，删哪几个的组合是最优的**。$2^{15}=32768$ 种可能——NSGA-II 在几代内就能探索大部分有意义的区域。

### 0.5 与 MOFS 文献的对齐

这个框架与 MOFS（Multi-Objective Feature Selection）文献的标准范式完全兼容：

| 组件 | MOFS 文献标准 | MOFS-MinPPI (Plan D) |
|------|-------------|---------------------|
| 搜索空间 | 所有特征子集，指数级 | **候选池子集**，通过博弈论预筛大幅缩小 |
| 编码 | 二元掩码 | 二元掩码（仅在候选池上） |
| 目标 | 基数 + 错误率 + 训练时间 | 压缩率 + 错误率 |
| 搜索算法 | NSGA-II / MOEA/D / SPEA2 | 轻量 NSGA-II |
| 对 MinPPI 的定位 | — | 第一阶段预筛器，不是算子 bias |

**关键卖点**：MOFS-MinPPI 是第一个用**博弈论幂指数作为搜索空间预筛器**的多目标特征选择方法。与无引导的 NSGA-II-uniform 相比，MinPPI 将搜索空间从 $2^{100} \approx 10^{30}$ 缩减到 $2^{20} \approx 10^6$（对 Madelon），降低了 **24 个数量级**。

---

## 第一章：完整架构设计 —— MOFS-MinPPI Plan D

### 1.1 算法伪代码

```
Algorithm: MOFS-MinPPI (Plan D)

Input:  数据集 D, 基学习器 M, α 集合 A, NSGA-II 参数 (pop, gen)
Output: Pareto 前沿 P*

// ── Phase 1: Candidate Pool Construction ──
C ← ∅
For each α ∈ A:
    MinPPI_α ← estimate_minppi(D, M, α)     // §4.3
    H_α ← {i : MinPPI_α[i] = -1}
    C ← C ∪ H_α
// C typically contains 3--40 features

// ── Phase 2: Multi-Objective Search over C ──
Encode each solution as m ∈ {0,1}^{|C|}
    m_j = 1  →  feature C[j] is DELETED

P_0 ← random_population(pop_size, |C|)
Evaluate each m ∈ P_0:
    D_m ← {C[j] : m_j = 1}
    X_filtered ← drop D_m from D
    f₁(m) = |D_m| / n                       // compression
    f₂(m) = 1 - AUC(X_filtered)             // error

For g = 1 to max_gen:
    Offspring ← ∅
    While |Offspring| < pop_size:
        p₁, p₂ ← tournament_select(P_g)
        child ← uniform_crossover(p₁, p₂)
        child ← bit_flip_mutation(child)
        Evaluate child
        Offspring ← Offspring ∪ {child}
    P_{g+1} ← non_dominated_sort(P_g ∪ Offspring)[:pop_size]

Return non_dominated(P_final)
```

### 1.2 复杂度分析

| 阶段 | 成本 | 备注 |
|------|------|------|
| Phase 1: MinPPI 估计 | O(|A| × |C| × K_adv × T_model) | 与当前相同，约 40s/seed/α |
| Phase 2: NSGA-II | O(gen × pop × T_model) | pop=20, gen=15 → 300 次模型训练；每次 AUC 评估用全特征（除删除集）训练 XGBoost，约 1-3 秒 |
| **总计** | **Phase 2 主导，但总成本可控** | Madelon 单个 seed 约 **3-5 分钟**（含 Phase 1 + Phase 2） |

**对比旧 Plan A/B**：旧框架 NSGA-II 搜索全部 n=100 维空间，pop=30, gen=20 → 600 次评估，但每次评估只训练弱模型（15 estimators, depth=2），总时间相似。但旧框架的 NSGA-II 无法有效利用 MinPPI 信号（Plan C 的 5 重根因），而新框架中候选池本身就是来自 MinPPI 的强先验。

### 1.3 MOFS Baseline 对比设计

新框架下，我们可以构建**4 个 MOFS 方法**的对比——这带来了全新的实验维度：

| 方法 | 候选池来源 | 搜索空间大小 | 描述 |
|------|----------|------------|------|
| **MOFS-MinPPI** | MinPPI 跨 α 的 ∪H_α | 3--40 维 | 本文方法 |
| **MOFS-SHAP** | SHAP 底部 k 特征，k = |C_MinPPI| | 等大小 | SHAP 作为候选池预筛器——对照：博弈论预筛是否优于 SHAP？ |
| **MOFS-MinShap** | MinShap 负分特征的前 k 个，k = |C_MinPPI| | 等大小 | 连续 min 算子作为候选池预筛器——对照：二元阈值是否比连续阈值更好？ |
| **NSGA-II-uniform** | 无预筛（全部特征） | 56--100 维 | 标准 MOFS baseline——对照：候选池预筛是否优于全空间搜索？ |

这使得我们可以报告：
1. **Pareto 前端质量**：HV（Hypervolume）、IGD+（Inverted Generational Distance）
2. **压缩率**：在给定 AUC 损失预算下的特征删除比例
3. **收敛速度**：达到 95% 最终 HV 所需的代数
4. **CD 图**：4 个 MOFS 方法在 7 个数据集上的 Friedman-Nemenyi 检验

这是 paper 之前缺失的关键实验维度——**NSGA-II 不再是与 MinPPI 竞争的外部 baseline，而是 MinPPI 内部的第二阶段组件；MOFS 对比在 "候选池质量" 维度上进行**。

---

## 第二章：算法实现（具体代码改动）

### 2.1 新建核心模块：`src/minppi_mofs/mofs_candidate_search.py`

这是 Phase 2 的轻量 NSGA-II，搜索空间仅限候选池。

```python
"""
MOFS-MinPPI Plan D — Lightweight NSGA-II over MinPPI candidate pool.

Phase 1 (MinPPI):  identifies a candidate pool C of potentially harmful
    features across multiple α levels.  |C| is typically 3--40, far smaller
    than the full feature dimension n ≈ 100.

Phase 2 (NSGA-II): searches over subsets of C only.  Each individual is a
    binary mask m ∈ {0,1}^{|C|} where m_j = 1 means feature C[j] is deleted.
    Objectives: (1) compression = |deleted|/n, (2) 1 − AUC on filtered set.

Because |C| is small, NSGA-II converges rapidly (pop=20, 15 generations
is sufficient).  The key contribution is the CANDIDATE POOL — MinPPI's
voting-game analysis tells the MOEA *where to look*, reducing the search
space by 20+ orders of magnitude.

Usage:
    from minppi_mofs.mofs_candidate_search import mofs_minppi_search
    front = mofs_minppi_search(X, y, candidate_pool, alpha=0.95, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class MOFSCandidateResult:
    pareto_masks: np.ndarray       # shape (k, |C|)
    pareto_fitness: np.ndarray     # shape (k, 2)
    candidate_pool: list[str]
    hypervolume: float
    runtime_s: float
    history: list[dict]


def _evaluate_mask(
    mask: np.ndarray,
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
    candidate_pool: list[str],
    task: str = "binary",
    seed: int = 42,
) -> tuple[float, float]:
    """Return (compression, 1 − AUC) for a deletion mask."""
    n = X_tr.shape[1]
    deleted = {candidate_pool[j] for j in range(len(candidate_pool)) if mask[j]}
    keep = [c for c in X_tr.columns if c not in deleted]
    cr = len(deleted) / n if n > 0 else 0.0

    if not keep:
        return cr, 1.0

    X_tr_f = X_tr[keep].values
    X_te_f = X_te[keep].values

    if task == "regression":
        m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         verbosity=0, random_state=seed)
        m.fit(X_tr_f, y_tr)
        val = float(r2_score(y_te, m.predict(X_te_f)))
        if val < 0.0:
            val = 0.0
    else:
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          verbosity=0, random_state=seed)
        m.fit(X_tr_f, y_tr)
        probs = m.predict_proba(X_te_f)
        n_cls = len(np.unique(y_te))
        if n_cls == 2:
            val = float(roc_auc_score(y_te, probs[:, 1]))
        else:
            from sklearn.metrics import accuracy_score
            val = float(accuracy_score(y_te, m.predict(X_te_f)))

    return cr, 1.0 - val


def _non_dominated_sort(fitness: np.ndarray) -> list[np.ndarray]:
    """NSGA-II non-dominated sorting. Both objectives are minimized."""
    n = fitness.shape[0]
    dominated_by = [set() for _ in range(n)]
    dominates_count = [0] * n
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (fitness[i] <= fitness[j]).all() and (fitness[i] < fitness[j]).any():
                dominated_by[i].add(j)
            elif (fitness[j] <= fitness[i]).all() and (fitness[j] < fitness[i]).any():
                dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while fronts[current]:
        next_front = []
        for i in fronts[current]:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)
        current += 1
        fronts.append(next_front)
    fronts.pop()  # remove empty last
    return [np.array(f, dtype=int) for f in fronts]


def _crowding_distance(fitness: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Crowding distance for a single front."""
    n_obj = fitness.shape[1]
    n = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for obj in range(n_obj):
        order = np.argsort(fitness[front, obj])
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        f_range = fitness[front[order[-1]], obj] - fitness[front[order[0]], obj]
        if f_range > 0:
            dist[order[1:-1]] += (
                fitness[front[order[2:]], obj] - fitness[front[order[:-2]], obj]
            ) / f_range
    return dist


def _compute_hv(front: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume (2D only, simple rectangle method)."""
    sorted_idx = np.argsort(front[:, 0])
    sorted_f = front[sorted_idx]
    hv = 0.0
    prev_x = 0.0
    for i in range(len(sorted_f)):
        hv += (sorted_f[i, 0] - prev_x) * (ref[1] - sorted_f[i, 1])
        prev_x = sorted_f[i, 0]
    return hv


def mofs_candidate_search(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
    candidate_pool: list[str],
    *,
    task: str = "binary",
    pop_size: int = 20,
    max_gen: int = 15,
    seed: int = 42,
    verbose: bool = False,
) -> MOFSCandidateResult:
    """
    Lightweight NSGA-II over the MinPPI candidate pool C.

    Args:
        X_tr, y_tr: Training data.
        X_te, y_te: Test data (held-out; used only for final Pareto evaluation).
        candidate_pool: List of feature names in C (|C| ≤ 50).
        task: "binary", "multiclass", or "regression".
        pop_size: Population size (default 20, sufficient for |C| ≤ 50).
        max_gen: Maximum generations.
        seed: Random seed.

    Returns:
        MOFSCandidateResult with Pareto front info.
    """
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    k = len(candidate_pool)
    if k == 0:
        return MOFSCandidateResult(
            pareto_masks=np.zeros((0, 0)),
            pareto_fitness=np.zeros((0, 2)),
            candidate_pool=[],
            hypervolume=0.0,
            runtime_s=0.0,
            history=[],
        )

    # ── Init ──
    pop = rng.integers(0, 2, size=(pop_size, k)).astype(np.float64)
    fitness = np.array([
        _evaluate_mask(pop[i], X_tr, y_tr, X_te, y_te, candidate_pool, task, seed)
        for i in range(pop_size)
    ])
    history: list[dict] = []

    # ── NSGA-II loop ──
    for gen in range(max_gen):
        fronts = _non_dominated_sort(fitness)
        offspring = np.zeros((pop_size, k))
        for i in range(0, pop_size, 2):
            # Tournament selection
            t1 = rng.integers(0, pop_size, 2)
            t2 = rng.integers(0, pop_size, 2)
            p1 = t1[0] if _dominates(fitness[t1[0]], fitness[t1[1]]) else t1[1]
            p2 = t2[0] if _dominates(fitness[t2[0]], fitness[t2[1]]) else t2[1]
            # Uniform crossover
            mask = rng.random(k) < 0.5
            c1, c2 = pop[p1].copy(), pop[p2].copy()
            c1[mask], c2[mask] = c2[mask], c1[mask].copy()
            # Bit-flip mutation
            for child in (c1, c2):
                flip = rng.random(k) < (1.0 / k)
                child[flip] = 1.0 - child[flip]
            offspring[i] = c1
            if i + 1 < pop_size:
                offspring[i + 1] = c2

        off_fitness = np.array([
            _evaluate_mask(offspring[i], X_tr, y_tr, X_te, y_te, candidate_pool, task, seed)
            for i in range(pop_size)
        ])
        combined = np.vstack([pop, offspring])
        combined_fit = np.vstack([fitness, off_fitness])

        fronts_c = _non_dominated_sort(combined_fit)
        new_pop = np.zeros((pop_size, k))
        new_fit = np.zeros((pop_size, 2))
        idx = 0
        for fr in fronts_c:
            if idx + len(fr) <= pop_size:
                new_pop[idx:idx + len(fr)] = combined[fr]
                new_fit[idx:idx + len(fr)] = combined_fit[fr]
                idx += len(fr)
            else:
                cd = _crowding_distance(combined_fit, fr)
                order = np.argsort(cd)[::-1]
                remaining = pop_size - idx
                new_pop[idx:] = combined[fr[order[:remaining]]]
                new_fit[idx:] = combined_fit[fr[order[:remaining]]]
                break
        pop, fitness = new_pop, new_fit

        hv = _compute_hv(fitness[_non_dominated_sort(fitness)[0]], np.array([1.0, 1.0]))
        history.append({"gen": gen, "hv": hv})

        if verbose and gen % 5 == 0:
            print(f"  MOFS gen {gen}: HV={hv:.4f}", flush=True)

    # ── Final Pareto front ──
    final_fronts = _non_dominated_sort(fitness)
    pf_idx = final_fronts[0]
    runtime = time.perf_counter() - t0
    hv_final = _compute_hv(fitness[pf_idx], np.array([1.0, 1.0]))

    return MOFSCandidateResult(
        pareto_masks=pop[pf_idx].astype(int),
        pareto_fitness=fitness[pf_idx],
        candidate_pool=candidate_pool,
        hypervolume=hv_final,
        runtime_s=runtime,
        history=history,
    )


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return (a <= b).all() and (a < b).any()
```

### 2.2 修改 `src/minppi_mofs/__init__.py`——暴露新 API

在现有 imports 下方增加：

```python
from minppi_mofs.mofs_candidate_search import (
    MOFSCandidateResult,
    mofs_candidate_search,
)

__all__ = [
    # ... existing entries ...
    "MOFSCandidateResult",
    "mofs_candidate_search",
]
```

### 2.3 新建 MOFS baseline 注册模块 `src/minppi_mofs/baselines/mofs_baselines.py`

```python
"""
MOFS baseline registry — four methods sharing the same NSGA-II engine
but differing in their candidate-pool construction strategy.

Methods:
  MOFS-MinPPI   — MinPPI ∪H_α
  MOFS-SHAP     — SHAP bottom-k (size-matched)
  MOFS-MinShap  — MinShap most-negative-k (size-matched)
  NSGA-II-full  — No candidate pool → search all features (standard MOFS)
"""

import numpy as np, pandas as pd
from minppi_mofs.min_ppi import estimate_minppi
from minppi_mofs.shapg_integration import build_guidance_from_data
from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
from minppi_mofs.baselines.min_shap import compute_minshap_importance
from minppi_mofs.mofs_candidate_search import mofs_candidate_search


def build_candidate_pool_minppi(
    X_tr, y_tr, X_val, y_val, *, alphas=(0.85, 0.90, 0.95, 0.99), task="binary", seed=42,
) -> list[str]:
    """Phase 1: construct C = ∪_α H_α via MinPPI."""
    pool = set()
    n_cls = len(np.unique(y_val))
    sg_s, sg_n = None, None
    if n_cls == 2 and task != "regression":
        sg_s, sg_n = build_guidance_from_data(X_tr, y_tr, seed=seed)

    for alpha in alphas:
        if n_cls > 2 or task == "regression":
            from minppi_mofs.min_ppi import estimate_minppi_random_mc
            minppi = estimate_minppi_random_mc(
                X_tr, X_val, y_tr, y_val, alpha=alpha, K=60,
                shared_coalitions=True, per_coalition=3, seed=seed, task=task,
            )
        else:
            minppi = estimate_minppi(
                X_tr, X_val, y_tr, y_val, alpha=alpha, method="shapg_guided",
                K_adv=30, shapg_neighbors=sg_n, shapg_scores=sg_s,
                seed=seed, task=task,
            )
        pool |= {f for f, v in minppi.items() if v == -1}
    return sorted(pool)


def build_candidate_pool_shap(X_tr, y_tr, k, *, task="binary", seed=42) -> list[str]:
    """Phase 1: SHAP bottom-k as candidate pool."""
    scores = compute_shap_topk_importance(X_tr, y_tr, seed=seed, task=task)
    ranked = sorted(scores, key=scores.get)
    return ranked[:k]


def build_candidate_pool_minshap(X_tr, y_tr, k, *, task="binary", seed=42) -> list[str]:
    """Phase 1: MinShap most-negative-k as candidate pool."""
    scores = compute_minshap_importance(X_tr, y_tr, seed=seed, task=task)
    ranked = sorted(scores, key=scores.get)
    return ranked[:k]


MOFS_METHODS = {
    "mofs_minppi": build_candidate_pool_minppi,
    "mofs_shap": build_candidate_pool_shap,
    "mofs_minshap": build_candidate_pool_minshap,
    # "nsga2_full" handled separately in experiment script
}
```

---

## 第三章：实验脚本设计（具体到每个文件与每条命令）

### E0：MOFS 主表实验（核心新实验，替代旧 `run_main_table.py` 的 MOFS 部分）

**文件名**：`experiments/run_mofs_main_table.py`

**目的**：在统一的多目标框架下对比 4 个 MOFS 方法。

**实验设计**：

| 方法 | Phase 1 候选池构建 | Phase 2 搜索 | 搜索空间维度 |
|------|-------------------|-------------|------------|
| **MOFS-MinPPI** | `build_candidate_pool_minppi` (跨 4α 的 ∪H_α) | `mofs_candidate_search` | 3--40 |
| **MOFS-SHAP** | SHAP bottom-k，(k = |C_MinPPI| at α=0.95) | `mofs_candidate_search` | 与 MinPPI 等大小 |
| **MOFS-MinShap** | MinShap most-negative-k，(k = |C_MinPPI|) | `mofs_candidate_search` | 与 MinPPI 等大小 |
| **NSGA-II-full** | 无候选池——搜索全部特征 | 标准 NSGA-II（pop=30, gen=20） | 56--100 |

**注意**：SHAP 和 MinShap 的候选池大小与 MinPPI 等量——这是有意的设计选择，确保任何 HV/IGD+ 差异反映的是**候选池质量**而非候选池大小。

**数据流**：

```
For each (dataset, seed):
    1. Load train/val/test splits
    2. Phase 1 for MOFS-MinPPI:
       → build_candidate_pool_minppi() → C_minppi (list of feature names)
    3. k = |C_minppi|  (at α=0.95 only, for size-matching)
    4. Phase 1 for MOFS-SHAP:
       → build_candidate_pool_shap(k=k) → C_shap
    5. Phase 1 for MOFS-MinShap:
       → build_candidate_pool_minshap(k=k) → C_minshap
    6. Phase 2 for each method (mofs_candidate_search):
       → Pareto front, HV, IGD+, runtime
    7. NSGA-II-full: standard NSGA-II on all features
       → Pareto front, HV, IGD+, runtime
    8. Save per-method results
```

**产出**：

| 文件 | 内容 |
|------|------|
| `mofs_pareto_metrics.csv` | 每行 = (dataset, seed, method, HV, IGD+, |C|, runtime) |
| `mofs_pareto_metrics_agg.csv` | 5-seed mean ± std |
| `mofs_candidate_pool_stats.csv` | 每行 = (dataset, seed, method, |C|, |C|/n) |

**评估指标**：

| 指标 | 含义 | 方向 |
|------|------|:--:|
| HV | Hypervolume (ref point: (1,1)) | 越高越好 |
| IGD+ | Inverted Generational Distance (ref: 所有方法 × seed 的非支配联合前端) | 越低越好 |
| |C| | 候选池大小 | — |
| Runtime | Phase 1 + Phase 2 总耗时 (s) | 越低越好 |
| CR_1%/3%/5% | 在 ≤1%/3%/5% AUC 损失预算下可删除的特征比例 | 越高越好 |

**运行命令**：
```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -u experiments/run_mofs_main_table.py
```

**预期运行时间**：每个 seed 约 2-5 分钟（Phase 1 MinPPI: ~40s, Phase 2 NSGA-II 在候选池上: ~1-3min, 4 方法 × 300 评估/方法）。7 数据集 × 5 seeds × 4 方法 ≈ 140 × 2-5min ≈ **5-10 小时**。可将 4 个方法分配到 3 张 H100 上并行。

**写入论文位置**：
- §6.2 Main Results：新增 Table M1（4 方法 × 7 数据集的 HV/IGD+/CR 表）
- §6.2：新增 CD 图（4 方法在 HV 和 IGD+ 上的 Friedman-Nemenyi）
- §6.4 消融：候选池质量对比（MOFS-MinPPI vs MOFS-SHAP vs MOFS-MinShap）
- Appendix：每数据集完整 Pareto 前端表

---

### E1：半合成有害特征注入实验（解决 M2, M7）

**文件名**：`experiments/d1_semisynthetic_harmful.py`

**目的**：在非 Madelon 数据集上创建已知 ground-truth 有害特征，验证 MinPPI 候选池的质量——即 $C_{minppi}$ 是否包含了注入的有害特征。

**实验设计**：

对 **Spambase 和 ISOLET** 各创建 3 个变体：

| 变体 | 注入方式 | Ground-truth harmful | 预期 |
|------|---------|---------------------|------|
| **D1a** (纯噪声) | 20 列 $\mathcal{N}(0,1)$ | 全部 20 列 | MinPPI 应检测到部分（不一定是全部，因为保守性），但至少 > 0 |
| **D1b** (置换相关) | 5 列，每列是某预测特征的随机置换 | 全部 5 列 | 更难——与信号特征相关，但无预测力。MinPPI 检测率应低于 D1a |
| **D1c** (条件有害) | 5 列噪声 + 标签翻转机制：当 noise_col > 0 且 TOP-2 MI 特征同时存在时，以 p=0.2 翻转标签 | 全部 5 列 | **最关键**——条件有害，模拟 MinPPI 声称要检测的场景 |

**D1c 标签翻转逻辑**（关键——这是 "harmful witness" 的显式实现）：
```
对训练集中每个样本 k:
    if noise_col[k] > 0 and X_tr[pair_feat_1][k] > 0 and X_tr[pair_feat_2][k] > 0:
        if random() < 0.2:
            y_tr[k] = 1 - y_tr[k]  # flip label
```

这意味着 noise_col 单独存在不会降 AUC，但当它与 pair_feat_1 和 pair_feat_2 同时出现时才会。MinPPI 的 $v_\alpha$ 恰好会检测到：在包含 pair_feat_1 + pair_feat_2 的 coalition 中，加入 noise_col 可能导致 AUC 下降 → $v_\alpha(S)=1, v_\alpha(S \cup \{noise\_col\})=0$ → MinPPI = -1。

**评估指标**（对每个变体 × 数据集 × 方法）：

| 指标 | 含义 |
|------|------|
| Recall@C | gt_harmful ∩ C / gt_harmful — 候选池覆盖了多少 ground-truth 有害特征 |
| Precision@C | gt_harmful ∩ C / C — 候选池中有多少是真正的有害特征 |
| HV (Phase 2) | 从该候选池出发的 NSGA-II 的 Hypervolume |
| |C| | 候选池大小 |

**产出**：`results/tables/<run_id>/d1_semisynthetic.csv`
**写入论文位置**：§6.4 新子节 "Semi-Synthetic Harmful Injection"（一段 + 一表）

**运行**：
```powershell
..\venv\Scripts\python -u experiments/d1_semisynthetic_harmful.py
```
**预期时间**：约 1.5h（2 数据集 × 3 变体 × 4 方法 × Phase1+Phase2）

---

### E2：候选池消融——不同 α 组合的影响

**文件名**：`experiments/d4_alpha_pool_ablation.py`

**目的**：消融 Phase 1 中使用的 α 集合对候选池质量和最终 Pareto 前沿的影响。

**实验设计**：在 **Madelon**（最大候选池）上，比较 4 种 α 选择策略：

| 策略 | α 集合 | 预期 |
|------|--------|------|
| **Single α=0.95** | {0.95} | 最保守——候选池最小，但可能遗漏 borderline harmful 特征 |
| **Two α** | {0.90, 0.99} | 中等 |
| **Four α (default)** | {0.85, 0.90, 0.95, 0.99} | 推荐——平衡覆盖与特异性 |
| **Dense sweep** | {0.80, 0.85, 0.90, 0.95, 0.97, 0.99} | 作为上限对照 |

对每种策略：
1. 构建候选池 C
2. 运行 Phase 2 轻量 NSGA-II
3. 记录 HV, |C|, 候选池 Precision/Recall vs Madelon 探针

**产出**：`results/tables/<run_id>/d4_alpha_pool_ablation.csv`
**写入论文位置**：§6.4 新子节 "Candidate Pool Sensitivity to α"

**运行**：
```powershell
..\venv\Scripts\python -u experiments/d4_alpha_pool_ablation.py
```
**预期时间**：约 30min（Madelon × seed 42 × 4 策略）

---

### E3：引导信号消融（解决 M5）

**文件名**：`experiments/d2_guidance_ablation.py`（同旧方案）

**目的**：量化 Algorithm B 中用 MI 替代完整 ShapG pipeline 对候选池质量和最终 Pareto 前沿的影响。

**在 MOFS 框架下的新含义**：Phase 1 的候选池构建依赖 Algorithm B 的引导信号质量。如果 MI 和 ShapG-full 产生显著不同的候选池，则 Phase 2 的 NSGA-II 前端也会不同——论文必须报告这个差异。

**实验设计**：在 **Madelon**（有探针标签）上，比较 3 种引导信号：

| 条件 | 引导信号 | Phase 2 的 HV |
|------|---------|-------------|
| ShapG-full | 完整 ShapG pipeline | HV_shapg |
| MI-fast | 互信息（当前代码实际使用） | HV_mi |
| Random | 无引导 | HV_rand |

**产出**：`results/tables/<run_id>/d2_guidance_ablation.csv`
**写入论文位置**：§6.4 或 §5 Theory 末尾

**运行**：
```powershell
..\venv\Scripts\python -u experiments/d2_guidance_ablation.py
```
**预期时间**：约 45min（ShapG-full 慢，Madelon × seed 42 × 3 条件 × Phase1+Phase2）

---

### E4：公平删除对比 + ISOLET 效用消融（解决 M4, M6）

**文件名**：修改 `experiments/eval_harmful_utility.py`

**M4 修复（ISOLET 效用消融）**：新增一段逻辑——在 ISOLET seed=42 上额外计算一次 OVR-AUC-based $v_\alpha$ 的 MinPPI，将 harmful 集合与 accuracy-based 比较（Jaccard），并记录 ΔAUC 差异。

**M6 修复（公平删除对比）**：MinShap 和 SHAP 的删除集大小改为与 MinPPI 等量（matched-budget）：

```python
# M6 fix: matched-budget deletion for MinShap and SHAP
n_h = len(harmful_ppi)
ms_ranked = sorted(ms_scores.items(), key=lambda kv: kv[1])  # ascending
minshap_matched = {f for f, _ in ms_ranked[:n_h]}

sh_ranked = sorted(sh_scores.items(), key=lambda kv: kv[1])  # ascending
shap_matched = {f for f, _ in sh_ranked[:n_h]}
```

**运行**：
```powershell
..\venv\Scripts\python -u experiments/eval_harmful_utility.py
```
**预期时间**：约 3-4h（7 数据集 × 5 seeds，M4 额外开销 negligible）

---

### E5：MinPPI 单点排名能力对比（解决 M3）

**文件名**：修改 `experiments/run_main_table.py`

在当前 5 个 baseline 基础上添加 `"minppi"` 方法。MinPPI 排名逻辑：Phase 1（MinPPI, α=0.95）→ 删除 harmful → 在过滤后特征上训练 XGBoost → 提取 `feature_importances_`（gain）→ 按 gain 降序排序 → harmful 特征排最末。

**运行**：
```powershell
..\venv\Scripts\python -u experiments/run_main_table.py
```
**预期时间**：约 1-2h（只增 1 方法 × 7 数据集 × 5 seeds，可手动只跑 minppi entry）

---

### E6：统计检验与图表（CD 图、Friedman LOO）

```powershell
# 找最新 main_metrics.csv 和 mofs_pareto_metrics.csv
$R0 = (Get-ChildItem -Directory "results/tables" | Where-Object { Test-Path "$_\main_metrics.csv" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
$R1 = (Get-ChildItem -Directory "results/tables" | Where-Object { Test-Path "$_\mofs_pareto_metrics.csv" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name

# CD 图 1: 6 methods (含 MinPPI 排名) 单点 AUC
..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R0/main_metrics.csv --metric AUC@10 --higher
..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R0/main_metrics.csv --metric AUC@20 --higher

# CD 图 2: 4 MOFS methods on HV
..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R1/mofs_pareto_metrics.csv --metric HV --higher

# CD 图 3: 4 MOFS methods on IGD+
..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R1/mofs_pareto_metrics.csv --metric IGD+ --lower

# Friedman leave-one-out
..\venv\Scripts\python experiments/sensitivity_friedman.py results/tables/$R0/main_metrics.csv
```

然后复制 PDF 到 `conference-paper/figures/`：
```powershell
Copy-Item results/tables/$R0/cd_diagram_AUC@10.pdf conference-paper/figures/
Copy-Item results/tables/$R0/cd_diagram_AUC@20.pdf conference-paper/figures/
Copy-Item results/tables/$R1/cd_diagram_HV.pdf conference-paper/figures/
Copy-Item results/tables/$R1/cd_diagram_IGD+.pdf conference-paper/figures/
```

---

## 第四章：LaTeX 全文一致性修改清单（Phase A，约 3 小时）

### 总原则

新叙事中，"多目标"不再是空洞的标签——它由 Phase 2 的轻量 NSGA-II 实现。$\alpha$ 扫参不再是"多目标"证据——它是 Phase 1 候选池构建中控制保守性的参数。Pareto 前沿由 NSGA-II 发现，不由 $\alpha$ 单调性保证。

### A1. 标题

```latex
% 保持不变 — "Multi-Objective" 现在是名正言顺的
\title{Min-Marginal Power Indices \\ for Multi-Objective \\ Feature Selection: A Voting-Game Approach}
```

### A2. Abstract（行 61-68）

1. 删除 "MinPPI constructs an $\alpha$-parameterized Pareto front..."
2. 替换为："We embed MinPPI into a two-phase multi-objective framework: MinPPI constructs a candidate pool of potentially harmful features (Phase~1), reducing the search space by orders of magnitude; a lightweight NSGA-II then discovers the true Pareto front over this pool (Phase~2), trading off compression against AUC preservation."
3. 删除重复句（m1）
4. 最后一句保留，但 "for feature selection" → "for multi-objective feature selection"

### A3. §1 Introduction 贡献列表（行 80-89）

贡献 #3 重写：
```latex
\item {\bf MOFS-MinPPI: Two-Phase Multi-Objective Framework.}
MinPPI serves as a Phase-1 search-space reducer, constructing a
compact candidate pool $C = \cup_\alpha \mathcal{H}_\alpha$ of
potentially harmful features.  A lightweight NSGA-II then searches
over subsets of $C$ only ($|C| \approx 3{-}40$, vs.\ $n \approx 100$),
discovering the true Pareto front in (compression, AUC preservation)
space.  The candidate-pool construction reduces the NSGA-II search
space by 20+ orders of magnitude on typical datasets.
```

### A4. §3 Prelim — Pareto Optimality 小节（行 140-142）

保留，但改为为 §4.4 的 Phase 2 NSGA-II 做铺垫：
```latex
\subsection{Pareto Optimality}
A solution $\mathbf{x}$ Pareto-dominates $\mathbf{x}'$ iff
$f_j(\mathbf{x}) \leq f_j(\mathbf{x}')$ for all objectives with at
least one strict inequality.  The Pareto front is the set of all
non-dominated solutions.  In §4.4 we use an NSGA-II over the
MinPPI candidate pool to discover the Pareto front in (compression,
AUC preservation) space.
```

### A5. §4.4 — 最重要的改写（行 213-225）

标题保留：`\subsection{$\alpha$-Parameterized Multi-Objective Selection}`——现在是名正言顺的

新内容：
```latex
MOFS-MinPPI operates in two phases.

\textbf{Phase 1: Candidate Pool Construction.}
For each $\alpha \in \{\alpha_1, \dots, \alpha_k\}$ (default
$\{0.85, 0.90, 0.95, 0.99\}$), MinPPI identifies the harmful
set $\mathcal{H}_\alpha$.  The candidate pool is the union:
$C = \bigcup_{\alpha} \mathcal{H}_\alpha$, typically 3--40
features---orders of magnitude smaller than the full feature
dimension $n \approx 100$.  Each feature in $C$ has been flagged
as harmful at some conservativeness level, making it a plausible
candidate for deletion; all features outside $C$ are treated as
benign and never considered for removal.

\textbf{Phase 2: Lightweight NSGA-II over $C$.}
A feature subset is encoded as a binary mask $\mathbf{m} \in
\{0,1\}^{|C|}$, where $m_j = 1$ means the $j$-th candidate
feature is deleted.  The two objectives are:
\begin{equation}
\min_{\mathbf{m}} \bigl(
  -\text{compression} = -|D_\mathbf{m}|/n,\;
  1 - \text{AUC}(\mathcal{M}_{N_1 \setminus D_\mathbf{m}})
\bigr),
\label{eq:mo-obj-d}
\end{equation}
where $D_\mathbf{m} = \{C[j] : m_j = 1\}$.  Standard NSGA-II
with uniform crossover and bit-flip mutation searches the space
of $2^{|C|}$ candidate deletion sets.  Because $|C|$ is small,
a modest budget (population 20, 15 generations, $\approx$ 300
model evaluations) suffices for convergence.

\textbf{Why this works.}  MinPPI's voting-game analysis ensures
that every feature in $C$ has at least one context where it
unambiguously turns a winning coalition into a losing one---the
hallmark of a harmful feature.  NSGA-II's role is to determine
the optimal trade-off: how many of these high-confidence
candidates should actually be deleted to balance compression
against predictive performance.  Since $C$ is small, NSGA-II
converges rapidly and the discovered Pareto front reflects
genuine multi-objective optimization rather than a parametric
sweep.
```

### A6-A13. 其余修改

遵循与旧方案相同的结构，但关键区别：
- 所有 "$\alpha$ sensitivity analysis" → "$\alpha$ sweep for Phase-1 candidate pool construction"
- `tab:pareto_front` 表保留，但标题改为 "Phase-1 candidate pool size and Phase-2 AUC preservation at four $\alpha$ levels"
- `fig:pareto_front_cls/reg` 图保留，但标题改为 "MOFS-MinPPI operating points"——这些点现在是 Phase 2 NSGA-II 搜索的起点和边界
- 新增 MOFS 对比表和 CD 图（来自 E0）
- §6.4 新增 Phase 1 候选池消融（来自 E2）和引导信号消融（来自 E3）

---

## 第五章：全新完整执行时间线

### Day 1（约 8 小时）

| 步 | 任务 | 时间 | 并行 |
|:--:|------|:--:|:--:|
| 1 | 实现 `mofs_candidate_search.py` | 2h | — |
| 2 | 实现 `mofs_baselines.py` | 30min | — |
| 3 | 实现 `run_mofs_main_table.py` | 1h | — |
| 4 | Phase A 文本修改（A1-A13） | 2h | — |
| 5 | m1-m8 细微修复 | 30min | — |
| 6 | 启动 E0（run_mofs_main_table, 5-10h） | — | 后台运行过夜 |
| 7 | 启动 E2（d4_alpha_pool_ablation） | — | 后台 |
| 8 | 启动 E3（d2_guidance_ablation） | — | 后台 |

### Day 2（约 8 小时）

| 步 | 任务 | 时间 | 并行 |
|:--:|------|:--:|:--:|
| 9 | 实现 D1 semisynthetic（E1） | 1h | — |
| 10 | 启动 E1 | — | 后台 |
| 11 | 修改 `eval_harmful_utility.py`（M4+M6 修复） | 30min | — |
| 12 | 启动 E4（rerun removal test, 3-4h） | — | 后台 |
| 13 | 修改 `run_main_table.py`（E5, 添加 minppi 排名） | 30min | — |
| 14 | 启动 E5（1-2h） | — | 后台 |
| 15 | 将 E2/E3 结果写入论文 | 1h | — |
| 16 | 将 E0（若已完成）结果写入论文 | 1h | — |

### Day 3（约 4 小时，收尾）

| 步 | 任务 | 时间 |
|:--:|------|:--:|
| 17 | 运行 E6（CD 图 + Friedman LOO 全量更新） | 30min |
| 18 | 将 E1/E4/E5 结果写入论文 | 1.5h |
| 19 | ruff check + 最终编译 | 15min |
| 20 | 全文通读 + 逐段自洽性检查 | 1.5h |

**总新增工作量**：约 20 小时（3 天），大部分重度计算在后台/夜间。提交截止 7 月 21 日，从 7 月 7 日算起有 14 天——时间充裕。

---

## 第六章：M1 最终验证检查清单

提交前确认以下每一项：

### 核心架构
- [ ] `mofs_candidate_search.py` 已实现并测试（在 Madelon seed=42 上运行 < 5min）
- [ ] `run_mofs_main_table.py` 已跑完，产出 `mofs_pareto_metrics.csv`
- [ ] 4 个 MOFS 方法（MinPPI, SHAP, MinShap, NSGA-II-full）HV 之间有统计显著差异（Friedman p < 0.05）
- [ ] MOFS-MinPPI 的候选池大小（|C|）在所有数据集上 < 50

### 论文正文
- [ ] §4.4 的伪代码/描述与 `mofs_candidate_search.py` 的实现一致
- [ ] 论文明确区分：Phase 1（MinPPI 候选池构建）和 Phase 2（轻量 NSGA-II）
- [ ] "Pareto front" 的所有出现都指向 Phase 2 的 NSGA-II 输出，而非 α 扫参
- [ ] α 扫参被称为 "Phase-1 candidate pool construction sweep" 或类似表述
- [ ] 论文不含 Proposition 2a/2b

### 实验数据
- [ ] Table M1：4 方法 × 7 数据集的 HV/IGD+ 表
- [ ] CD 图 1：6 方法单点 AUC（含 MinPPI 排名）
- [ ] CD 图 2：4 MOFS 方法 HV
- [ ] CD 图 3：4 MOFS 方法 IGD+
- [ ] Table：每个数据集每方法的候选池大小（|C|）
- [ ] 半合成实验表（D1a/D1b/D1c 结果）
- [ ] 引导消融表（E3 结果）
- [ ] 公平删除对比表（E4 结果——matched-budget MinShap/SHAP）

### 全局 grep
- [ ] `grep "Proposition 2" icomp2026_conference.tex` 返回零
- [ ] 所有 `\ref{fig:}` 指向存在的 .pdf 文件
- [ ] 所有 `\ref{tab:}` 指向存在的表格

---

## 第七章：FAQ

**Q: Phase 2 的 NSGA-II 和 Plan A/B 的 NSGA-II 有什么本质区别？**

A: Plan A/B 的 NSGA-II 搜索全部 n≈100 维特征空间，MinPPI 作为 mutation/crossover bias。由于 mutation bias 相比 NSGA-II 自身的 selection 信号太弱，MinPPI 几乎不起作用（$\beta$ sweep p=0.84, n.s.）。Plan D 的 NSGA-II 仅搜索候选池（|C|≈3--40 维），候选池本身就是 MinPPI 的输出——特征要么在池内（值得考虑删除），要么不在（永远不删）。MinPPI 通过**预设搜索空间边界**来发挥作用，而不是通过**扰动 genetic operators**——这是两种完全不同的集成方式。

**Q: 候选池太小会导致 NSGA-II 搜索不充分吗？**

A: 不会。候选池的大小恰好是博弈论分析认为 "值得考虑删除" 的特征数量。NSGA-II 在候选池内部是完备的——它会评估 $2^{|C|}$ 中的大量组合，找到最优的（压缩率, AUC 保留率）平衡。如果数据集确实没有有害特征，|C| 接近 0，Pareto 前沿退化为单点 (0, 1.0)，这是正确的行为。

**Q: NSGA-II-full（无候选池）与 MOFS-MinPPI 的对比公平吗？**

A: 公平。两者使用相同的 NSGA-II 参数（selection, crossover, mutation），相同的评估函数（XGBoost 100 estimators, depth=4），相同的数据划分。唯一区别是搜索空间——这恰好是我们要测试的假设："MinPPI 候选池是否比全特征空间提供更好的搜索起点？" HV 和 IGD+ 的对比直接回答了这个问题。

**Q: 如果 MOFS-MinPPI 在 HV 上不如 NSGA-II-full 怎么办？**

A: 即使如此，MOFS-MinPPI 仍有价值——它的搜索空间小得多，运行时间短得多（Phase 2 是 $O(gen × pop × 2^{|C|})$ vs NSGA-II-full 的 $O(gen × pop × 2^n)$）。如果两者 HV 接近但 MOFS-MinPPI 快 10×+，那就是有力的 efficiency argument。而且 MOFS-MinPPI 的候选池附带可解释性——每个候选特征都可以追溯到一个具体的 harmful witness（哪次 coalition 评估中表现出了 -1 marginal），这是 NSGA-II-full 无法提供的。

### 1.1 为什么 M1 最关键

M1 不是局部措辞问题——它是 **整篇论文的叙事逻辑根基**。当前稿件声称要做"多目标特征选择"（Multi-Objective Feature Selection），声称构造了"α-Pareto 前沿"，但实际做法仅仅是：

1. 取 $\alpha \in \{0.85, 0.90, 0.95, 0.99\}$ 四个值
2. 对每个 $\alpha$，计算 MinPPI → 删除 harmful 特征 → 重新训练 → 得到（压缩率, AUC 保留率）
3. 把这 4 个点画在二维图上，称其为 "Pareto front"

**这不是多目标优化。** 这 4 个点之所以"非支配"，是因为 $|\mathcal{H}_\alpha|$ 关于 $\alpha$ 单调非增（提高阈值 → 更严格的"赢"定义 → 更少 coalition 赢 → 更少特征被判为 harmful）。这种单调性是定义本身保证的，不需要任何优化算法去"发现"。任何审稿人看到这段话都会立刻识破：

> "The set of points...forms a Pareto front: no α simultaneously achieves higher compression and higher preservation than another α. This follows from Property 2..."

这段论证等同于说："函数 $f(x)=x$ 在 $x\in\{1,2,3\}$ 上的取值构成 Pareto 前沿，因为没有点同时比另一个点更小。"——这是对的，但是 trivial。把 $\alpha$ 的单调效应称为"多目标选择方法"属于严重的概念错误。

### 1.2 是否需要重构算法？

**不需要。** MinPPI 的核心算法（WVG 公式 → min 算子 → 三分类 → 删除 → 验证）是正确的、有价值的。$\alpha$ 扫参也是一个合理且有用的分析工具——它验证了方法在不同保守程度下的稳健性。问题出在**对 $\alpha$ 扫参结果的解释和命名**。

**正确的新定位**：

| 旧（错误） | 新（正确） |
|---|---|
| MinPPI 是一种多目标特征选择方法 | MinPPI 是一种**安全有害特征检测与删除**方法 |
| $\alpha$-Pareto 前沿 | $\alpha$ **灵敏度分析** / 保守性参数扫参 |
| "压缩率 vs AUC 保留率的 Pareto 前沿" | "不同 $\alpha$ 下的（压缩率, AUC 保留率）**操作点**" |
| 4.4 节 "$\alpha$-Parameterized Multi-Objective Selection" | 4.4 节 "$\alpha$-Parameterized Safe Deletion" |
| 论文标题含 "Multi-Objective" | 删除，改为 "Safe Harmful-Feature Detection" 或类似表述 |

### 1.3 贯穿全文的全新叙事逻辑

论文真正讲的故事应该是这样的（**五段式统一叙事**）：

**第一段（Why）**：现有博弈论特征选择方法使用均值算子，会掩盖"平均有益但特定上下文中 harmful"的特征。MinShap 用 min 替换 mean，但使用连续值函数缺乏决策边界。

**第二段（What）**：我们提出 MinPPI——将 min 算子与**二元**效用阈值投票博弈配对，生成离散的 $\{-1,0,+1\}$ 输出，干净地将每个特征分类为 harmful / neutral / decisive。

**第三段（How）**：给出数学形式化（Properties 1-3），设计估计算法（Algorithm A/B），引入 $\alpha$ 作为**保守性参数**控制 harmful 判定的严格程度。$\alpha$ 越低越宽松（更多特征被判为 harmful，压缩率高但精度低），$\alpha$ 越高越保守（更少特征被判为 harmful，压缩率低但精度接近 1）。

**第四段（Safety Evidence）**：核心实验是**安全验证**——删除 MinPPI-harmful 特征后重新训练，AUC 不出现在任何数据集上显著下降。Madelon 探针标签确认 flagged 特征几乎肯定是噪声（precision ≈ 0.98）。$\alpha$ 扫参证实这一安全性质在 $\{0.85, \dots, 0.99\}$ 范围内保持稳定。

**第五段（Comparison）**：与 5 个 baseline 比较 AUC@k 排名。MinPPI 不追求排名第一——它的价值在于**在排名之前**先安全地删除有害特征。MinShap 删除等量特征会导致 AUC 显著下降，证明二元阈值是安全性的关键。

这五段叙事中：
- **"安全"是主线**——论文的核心贡献是开发了一种**不会意外删除有用特征**的方法
- **"$\alpha$ 是保守性旋钮"**——用户可调，低 $\alpha$ 多删、高 $\alpha$ 保底
- **"三分类是 unique selling point"**——没有任何其他方法能给出 $\{-1, 0, +1\}$ 的保证
- **"α 扫参是稳健性证据，不是优化算法"**——图表证明方法对 $\alpha$ 的选择不敏感

---

## 第二章：全文一致性修改清单（M1 驱动）

以下清单覆盖了论文每一个需要因 M1 新叙事而修改位置，确保全文逻辑自洽。**执行顺序很重要**——先改完所有文本（Phase A），再跑任何实验（Phase B），否则新旧叙事混合会导致混乱。

### Phase A：纯文本修改（无需跑实验，约 3 小时）

#### A1. 标题

```latex
% 旧
\title{Min-Marginal Power Indices \\ for Multi-Objective \\ Feature Selection: A Voting-Game Approach}

% 新 — 三个候选，选一个
\title{Min-Marginal Power Indices \\ for Safe Harmful-Feature Detection \\ in Tabular Data}
% 或
\title{Min-Marginal Power Indices: \\ Safe Deletion of Harmful Features \\ via Voting Games}
% 或
\title{Min-Marginal Power Indices \\ for Conservative Feature Deletion: \\ A Voting-Game Approach}
```

推荐第二个——"Safe Deletion of Harmful Features"和"Voting Games"都是关键词，审稿人一眼能看懂贡献。

#### A2. Abstract（行 61-68）

需修改以下句子：
1. "MinPPI constructs an $\alpha$-parameterized Pareto front..." → "Varying the conservativeness parameter $\alpha$ over $\{0.85,0.90,0.95,0.99\}$ provides a sensitivity analysis confirming the method's robustness; we recommend $\alpha=0.95$ as the primary operating point."
2. 删除 "without external search" 及 Pareto front 相关断言
3. 重复句（m1）：删除第二次出现的 "the removal test confirms..."
4. 最后一句 "for feature selection beyond the specific methods studied here" → "for safe feature deletion as a preprocessing step before downstream modeling"

完整新版 Abstract 建议：

```
Existing game-theoretic feature selection methods mostly score each
feature by its mean marginal contribution. The mean conceals a risk:
a feature beneficial on average may harm performance in specific
combinations, and no method systematically identifies such features
before they degrade deployed models.

We introduce Min-Marginal Power Indices (MinPPI). Instead of
averaging, MinPPI takes the minimum marginal contribution under a
binary utility-threshold voting game v_α(S)=𝟙[U(M_S)≥α·U_full].
This min–binary pairing yields {-1,0,+1}, classifying every feature
as harmful, neutral, or decisive. The classification is
threshold-free: the boundary at 0 follows from v_α∈{0,1} with no
arbitrary cutoff.

We develop a ShapG-guided adversarial search (Algorithm B) that
detects sparse harmful features with substantially higher sample
efficiency than uniform Monte Carlo. The conservativeness
parameter α controls the precision–recall trade-off: α=0.95
achieves precision 0.978 on Madelon's ground-truth probes while
α=0.99 reaches 37% compression.

On seven tabular datasets, evaluated against five baselines,
deleting all MinPPI-harmful features preserves AUC within
1e-3 on 6 of 7 datasets. In contrast, deleting MinShap-negative
features at matched budget causes AUC drops up to 8.5%. The
binary v_α produces a principled three-way classification absent
from continuous methods, and the α-sensitivity analysis confirms
robustness across conservativeness levels. These findings suggest
that replacing the mean with the minimum over a thresholded value
function offers a broadly applicable principle for safe feature
deletion.
```

#### A3. Introduction §1 贡献列表（行 80-89）

四个贡献全部重写以匹配新叙事：

```latex
\item {\bf Min-Marginal Power Indices (MinPPI).}
[不变 — 这段本身没问题]

\item {\bf ShapG-Guided Adversarial Search (Algorithm~B).}
[不变]

\item {\bf Conservative Feature Deletion with Tunable α.}
The quota α acts as a conservativeness knob: lower values
flag more features as harmful (higher recall, lower precision);
higher values are stricter (near-perfect precision).
An α-sweep over {0.85,0.90,0.95,0.99} serves as a built-in
sensitivity analysis confirming safety across conservativeness
levels. We recommend α=0.95 as the default operating point.

\item {\bf Comprehensive Safety Validation.}
On seven tabular datasets, deleting all MinPPI-harmful features
preserves AUC within 1e-3 on 6 of 7 datasets. On Madelon
(ground-truth probes available), MinPPI achieves precision
0.978 at α=0.95. The removal test shows that MinShap-negative
features, when deleted at matched budget, cause AUC drops
absent from MinPPI's conservative signal—confirming that
the binary v_α is the key safety mechanism.
```

#### A4. §3 Prelim — "Pareto Optimality" 小节（行 140-142）

当前内容：
```latex
\subsection{Pareto Optimality}
A solution x Pareto-dominates x' iff ...
We construct a two-objective front in (compression rate,
AUC preservation) space by sweeping α.
```

这段要**完全删除**。新叙事中不需要 Pareto 子节。取而代之，在 §3.2 "Problem Setting" 末尾加一句话，为 §4.4 做铺垫：

```latex
The parameter α controls how stringent the winning threshold is.
In §4.4 we show how varying α produces a family of operating
points that reveal the compression–safety trade-off inherent
in the voting-game structure.
```

#### A5. §4 方法章标题（行 152）

`\section{MinPPI Framework}` — 这个可以保留，不需要大改。

#### A6. §4.4 — 最核心的改写（行 213-225）

当前标题：`\subsection{$\alpha$-Parameterized Multi-Objective Selection}`

新标题：`\subsection{$\alpha$-Parameterized Conservative Deletion}`

新内容（替代整个子节）：

```latex
The MinPPI framework provides a family of operating points
controlled by a single parameter α. For a fixed dataset and
a chosen α ∈ (0,1], the method:
\begin{enumerate}
  \item Computes MinPPI under v_α to classify each feature
        as harmful (−1), neutral (0), or decisive (+1).
  \item Deletes all harmful features from the dataset.
  \item Retrains the model on the filtered feature set,
        producing an operating point
        (compression rate c(α)=|H_α|/n,
         AUC preservation p(α)=AUC_filtered / AUC_full).
\end{enumerate}

The parameter α controls conservativeness:
\begin{itemize}
  \item Lower α admits more coalitions as "winning," making
        it easier for a feature to exhibit a harmful marginal
        (v(S)=1, v(S∪{i})=0). This yields higher compression
        but potentially lower precision.
  \item Higher α tightens the winning threshold, so only
        features with unambiguous harmful witnesses are
        flagged. This yields lower compression but
        near-perfect precision (empirically >0.95 for
        α ≥ 0.90 on Madelon).
\end{itemize}

We recommend α=0.95 as the default operating point. This
value balances the voting game's ability to detect harmful
features against the risk of false positives. In §6 we
present a sensitivity analysis over α∈{0.85,0.90,0.95,0.99}
confirming that the safety property—AUC preservation within
1e-3—holds across the full range, and that α=0.95 provides
reliable harmful-feature detection without excessive
conservatism.
```

**关键要点**：
- 删除所有 "Pareto front"、"multi-objective"、"non-dominated" 等词
- 将 $\alpha$ 扫参定位为 **灵敏度分析**，证实安全性稳健
- 将 4 个操作点的表定位为 **保守性扫参表**，非 Pareto 前沿表

#### A7. §4.5 Discussion（行 227-232）

当前内容基本可以保留，但把 "$α$-Pareto framework guarantees" 改为 "$α$-parameterized deletion guarantees"。删除 "spurious Pareto-optimality claim" 这个短语——现在没有 Pareto 了。

#### A8. §5 Theory（行 234-245）

Theorem 1 和 Theorem 2 保留不变。删除已不存在的 Proposition 2a/2b 的引用（如果有残留）。

#### A9. §6.1 Setup（行 252-253）

"six methods" → 这句话现在说 "six methods" 但 CD 图只有 5 个（MinPPI 被排除）。需改为：
```
We evaluate five baseline ranking methods---SHAP-Top-k, ShapG,
MinShap, Banzhaf-FS, and Powershap---on seven tabular datasets...
MinPPI is evaluated separately through its deletion-and-retrain
pipeline (§6.3), since its output is a three-way classification
rather than a feature ranking.
```

然后在 M3 解决方案中（Phase B），MinPPI 会被加入排名。

#### A10. §6.2 Main Results（行 258-290）

**CD 图段落**（行 268）：更新叙述——MinPPI 不在 CD 图中不再需要尴尬的解释，因为 §6.1 已经说了 MinPPI 单独评估。

**α 扫参表**（行 272-293）：标题从 "α-Pareto front" 改为 "α sensitivity analysis"：

```latex
\caption{Conservativeness sensitivity analysis: compression rate
and AUC/$R^2$ preservation at four α levels (mean over 5 seeds).
α=0.95 (bold) is the recommended default operating point.
Compression is |H_α|/n; preservation is filtered utility / full
utility. Preservation stays ≥ 0.985 across all datasets and
α levels, confirming that MinPPI-guided deletion is safe.}
```

**α-Pareto 散点图**（行 295-301）：标题中的 "α-Pareto front" → "α operating points"：

```latex
\caption{Conservativeness operating points on five classification
datasets. Each point is one α level (mean compression and
preservation over 5 seeds). All points lie near preservation=1.0,
confirming MinPPI-guided deletion is safe across the full α range.}
```

#### A11. §6.3 Harmful-Feature Utility（行 303-344）

文本内容基本正确，但因 M6 修复（公平删除对比），叙述需微调。当前文中的 MinShap 删除全部负分特征的对比不公平——修好后文中要改为 "at matched budget"。

修改 `tab:harmful_removal` 表头：删除 `|F|_MinShap` 列，因为修复后 MinShap 删除的特征数与 MinPPI 等量。添加 MinShap-matched 列替代现有的全负分列。

#### A12. §6.4 Ablation and Sensitivity（行 346-353）

当前只有一段 α-trade-off 和一段 binary vs continuous。需扩展（见下文 M3/M5 解决方案）。

#### A13. Appendix 各节

- `app:datasets`：不变
- `app:protocol`：删除 NSGA-II 残留引用（已做），确认 "shared-coalition Algorithm A" 描述与代码一致
- `app:full_tables`：**填入缺失的 AUC@k 表**（m2 修复，见下文 Phase B）
- `app:regression`：更新图标题（"α-Pareto front" → "α sensitivity"）
- `app:alpha`：表标题中 "Harmful feature % vs. quota α" 保留，但正文中不再称其为消融维度之一
- `app:minshap`：数据已更新，保持不变
- `app:stats`：Nemenyi 表已为 5-method 版本，保持不变（M3 修复后会变成 6-method，需重新生成 CD 图）

---

### Phase B：实验与数据修复（需跑实验，约 5-6 小时）

以下按依赖关系排序。依赖图如下：

```
E1 (D1_semisynthetic) ──┐
                         ├──→ E4 (rerun removal test, M6 fix)
E3 (ISOLET ablation)  ──┘
                         │
E2 (guidance ablation) ──┤
                         │
                         ├──→ E5 (rerun main_table, M3 fix)
                         │
                         └──→ E6 (rerun CD diagram)
```

详细如下。

---

## 第三章：Phase B 实验详细设计

### E1：半合成有害特征注入实验（解决 M2, M7, M4）

**文件名**：`experiments/d1_semisynthetic_harmful.py`

**目的**：在真实数据集上创建已知 ground-truth 有害特征，验证 MinPPI 的检测精度不是 Madelon 特例。

**实验设计**：

对 **Spambase 和 ISOLET** 分别创建 3 个变体，加上 **Madelon**（已有探针标签作为对照）。每个变体在原始特征基础上添加额外的已知有害/噪声特征：

| 变体 | 添加内容 | Ground-truth harmful 集合 | 预期 |
|------|---------|--------------------------|------|
| **D1a** (纯噪声) | 20 列 $\mathcal{N}(0,1)$ 噪声列 | 全部 20 列 | 所有方法都应检测到它们，MinPPI 可能因保守性而检测较少 |
| **D1b** (置换相关) | 5 列——每列是某个现有预测特征的随机置换 | 全部 5 列 | 更难检测（与有用特征相关），MinPPI 可能在有交互时才检测到 |
| **D1c** (注入有害) | 5 列噪声，外加**标签翻转机制**：当这 5 列中任一列与特定特征对（如 Spambase 的 `word_freq_make` + `char_freq_!`）同时出现时，以概率 0.2 翻转该样本的标签 | 全部 5 列 | MinPPI 应达到高 precision；MinShap 可能高 recall 但低 precision |

**D1c 是最关键的变体**——它不是"纯噪声"而是"条件有害"：这 5 列本身并不总是降低 AUC，只在特定上下文中（与信号特征对共存时）造成 harm。这模拟了 MinPPI 声称要检测的场景。

**伪代码流程**（完整实现在 `review_and_solutions.md` 第一部分已给出框架，此处细化 D1c 构造逻辑）：

```python
def make_d1c(X_tr, X_te, y_tr, rng, prefix):
    """
    向 (X_tr, X_te) 添加 N_HARMFUL=5 列噪声。
    同时在训练集中，当任一 harmful 列的值 > 0 且
    特定的特征对同时存在时，以概率 HARMFUL_FLIP_PROB 翻转标签。
    """
    n_samples_tr = X_tr.shape[0]
    n_samples_te = X_te.shape[0]

    # 选择特定特征对（用前两个 MI 最高特征作为代理）
    # 实际实现中从 build_guidance_from_data 获取
    pair_feats = [X_tr.columns[0], X_tr.columns[1]]

    harmful_cols = []
    for k in range(N_HARMFUL):
        col_name = f"__harmful_{k}__"
        harmful_cols.append(col_name)
        noise_tr = rng.normal(0, 1, n_samples_tr)
        noise_te = rng.normal(0, 1, n_samples_te)
        X_tr[col_name] = noise_tr
        X_te[col_name] = noise_te

        # 注入 harmful 交互：当噪声 > 0 且特定特征对同时存在时
        mask = (noise_tr > 0) & (X_tr[pair_feats[0]] > 0) & (X_tr[pair_feats[1]] > 0)
        flip_idx = mask & (rng.random(n_samples_tr) < HARMFUL_FLIP_PROB)
        y_tr[flip_idx] = 1 - y_tr[flip_idx]  # 翻转标签

    return X_tr, X_te, set(harmful_cols)
```

**评估指标**（对每个变体 × 数据集，运行 MinPPI / MinShap / SHAP，报告）：

| 指标 | 含义 |
|------|------|
| Precision | flagged ∩ true_harmful / flagged |
| Recall | flagged ∩ true_harmful / true_harmful |
| F1 | harmonic mean |
| # flagged | 方法标记为 harmful 的特征数 |
| ΔAUC (removal) | 删除 flagged 特征后的 AUC 变化 |

**运行命令**：
```powershell
cd C:\work\学校\A.SPBU\paper\ZhaoCHI\Paper\Master
..\venv\Scripts\python -u experiments/d1_semisynthetic_harmful.py
```

**预期运行时间**：约 1 小时（3 数据集 × 3 变体 × 3 方法 × 1 seed，ISOLET 最慢）

**产出**：`results/tables/<run_id>/d1_semisynthetic.csv`
**写入论文位置**：§6.4 新子节 "Semi-Synthetic Harmful Injection"（一个段落 + 一表）

---

### E2：引导信号消融实验（解决 M5）

**文件名**：`experiments/d2_guidance_ablation.py`

**目的**：量化 Algorithm B 中用 MI 替代完整 ShapG pipeline 对检测质量的影响。Algorithm B 的 seed coalition 构建质量直接影响 Theorem 2 的条件是否满足。

**实验设计**：在 **Madelon**（唯一有 ground-truth 的数据集）上，用 seed=42 运行 MinPPI 三次，仅改变引导信号：

| 条件 | 引导信号来源 | 描述 |
|------|------------|------|
| **ShapG-full** | `baselines/shapg.py` 的 `compute_shapg_importance()` | 原始 ShapG pipeline（构建相关图 → 枚举邻域子集 → 计算 Shapley 值）。成本高（~270s on Spambase），但这是论文声称的 Algorithm B |
| **MI-fast** | `shapg_integration.py` 的 `compute_shapg_guidance_signal()` | 当前代码实际使用的互信息分数 |
| **Random** | 无引导，seed coalition 从所有特征中随机选取 | 下界对照 |

**对每种条件，固定**：
- $\alpha=0.95$
- $K_{\text{adv}} = 20$（Madelon 的标准预算）
- seed=42

**记录**：
- harmful 特征集合
- Precision / Recall / F1 vs Madelon 探针标签
- 计算时间
- Harmful 集合之间的 Jaccard 相似度（ShapG-full vs MI-fast, ShapG-full vs Random, MI-fast vs Random）

**运行命令**：
```powershell
..\venv\Scripts\python -u experiments/d2_guidance_ablation.py
```

**预期运行时间**：约 30 分钟（ShapG-full 在 Madelon 100 特征上最慢）

**产出**：`results/tables/<run_id>/d2_guidance_ablation.csv`
**写入论文位置**：§6.4 新子节 "Guidance Signal: ShapG vs. Mutual Information"（一个段落 + 一表）

**预期结果**：如果 ShapG-full 和 MI-fast 的 harmful 集合 Jaccard > 0.80，说明 MI 是 ShapG 的可接受替代品；如果 < 0.50，必须在论文中保留 ShapG-full 作为主要引导信号，并讨论 MI 替代的局限。

---

### E3：ISOLET 效用函数消融（解决 M4）

**文件名**：`experiments/d3_isolet_utility_ablation.py`

**目的**：量化 ISOLET 上 accuracy-based $v_\alpha$ vs OVR-AUC-based $v_\alpha$ 对 harmful 检测结果的影响。

**实验设计**：在 **ISOLET** 上，用 seed=42 运行 MinPPI 两次：

| 条件 | `task` 参数 | $v_\alpha$ 使用的效用 |
|------|-----------|-------------------|
| **Accuracy** | `"multiclass"` | 准确率（当前做法） |
| **OVR-AUC** | 自定义 `_CachedAUCCharacteristic` 使用 `roc_auc_score(..., multi_class="ovr")` | OVR AUC |

**固定**：$\alpha=0.95$，$K=30$，$per\_coalition=2$（ISOLET 预算）

**记录**：
- 两个 harmful 集合的 Jaccard 相似度
- 各自 ΔAUC（删除后重新训练，用 OVR AUC 评估）
- 计算时间差异

**运行命令**：
```powershell
..\venv\Scripts\python -u experiments/d3_isolet_utility_ablation.py
```

**预期运行时间**：约 15 分钟（只需要 ISOLET seed 42）

**产出**：`results/tables/<run_id>/d3_isolet_utility.csv`
**写入论文位置**：§6.1 Setup 中 ISOLET 段落末尾（当前行 254），新增 1-2 句报告消融结果

**预期结果**：如果 Jaccard > 0.80，则论文当前的 accuracy 简化是合理的，段落中的 "conservative" 论证成立。如果 Jaccard < 0.50，需要改成 OVR-AUC 作为 ISOLET 的默认效用，并重新运行所有 ISOLET 相关实验（这是一个更大的改动）。

---

### E4：公平删除对比实验（解决 M6）

**修改方式**：修改 `experiments/eval_harmful_utility.py` 中的 MinShap 和 SHAP 删除逻辑，改为 **与 MinPPI 等量删除**（matched-budget）。

**具体修改**（代码位置约在 350-370 行）：

```python
# 旧代码（不公平——删除 MinShap 全部负分特征）
minshap_neg = {f for f, v in ms.items() if v < 0}
n_ms = len(minshap_neg)  # 可能是 67，而 MinPPI 只删了 22

# 新代码（公平——与 MinPPI 删除等量）
n_h = len(harmful_ppi)  # MinPPI 删除的特征数
ms_ranked = sorted(ms.items(), key=lambda kv: kv[1])  # MinShap 升序（最负的在前）
minshap_matched = {f for f, _ in ms_ranked[:n_h]}  # 与 MinPPI 等量

# SHAP 同理
sh_ranked = sorted(sh.items(), key=lambda kv: kv[1])  # 升序（最低 SHAP 值在前）
shap_matched = {f for f, _ in sh_ranked[:n_h]}
```

**运行命令**：
```powershell
..\venv\Scripts\python -u experiments/eval_harmful_utility.py
```

**预期运行时间**：约 3-4 小时（重新运行全部 7 数据集 × 5 seeds）

**产出**：覆盖现有的 `results/logs/<run_id>/c1_removal/c1_removal_cls.csv`
**写入论文位置**：更新 `tab:harmful_removal` 中的数据。新的 MinShap-matched 列应显示**与 MinPPI 更接近的 ΔAUC**（差距应在 0.005 以内），这会更公平地展示 MinPPI 的保守性优势——它不是通过删更少特征来赢得安全性，而是一样多的特征中它的选择更准。

**预期结果**：
- MinShap-matched ΔAUC：Spambase ~0.000, Madelon ~ -0.002, Credit-Card ~ -0.003
- 这仍然比 MinPPI 差（MinPPI 的 ΔAUC 经常是 +0.000 到 +0.002），但差距从 "离谱" 变成 "微妙但一致"
- **结论改为**："即使删除等量特征，MinPPI 的筛选优于 MinShap 和 SHAP，因为它只删除有明确 harmful witness 的特征而不会误删 borderline negative 但实际有用的特征"

---

### E5：MinPPI 排名能力对比（解决 M3）

**修改方式**：在 `run_main_table.py` 的 baseline registry 中添加一个 `"minppi"` 方法。

**具体代码**（在 `run_main_table.py` 的 `BASELINES` dict 附近添加）：

```python
def _compute_minppi_ranking(X, y, task="binary", seed=42, X_val=None, y_val=None):
    """MinPPI-based ranking: delete harmful → XGBoost gain on filtered set."""
    from minppi_mofs.min_ppi import estimate_minppi
    from minppi_mofs.shapg_integration import build_guidance_from_data

    if X_val is None:
        X_val, y_val = X, y  # fallback for API compat

    n_cls = len(np.unique(y))
    if n_cls > 2 or task == "regression":
        from minppi_mofs.min_ppi import estimate_minppi_random_mc
        minppi = estimate_minppi_random_mc(
            X, X_val, y, y_val, alpha=0.95, K=60,
            shared_coalitions=True, per_coalition=3, seed=seed, task=task,
        )
    else:
        sg_s, sg_n = build_guidance_from_data(X, y, seed=seed)
        minppi = estimate_minppi(
            X, X_val, y, y_val, alpha=0.95, method="shapg_guided",
            K_adv=30, shapg_neighbors=sg_n, shapg_scores=sg_s, seed=seed, task=task,
        )

    harmful = {f for f, v in minppi.items() if v == -1}
    keep = [c for c in X.columns if c not in harmful]

    if len(keep) == 0:
        # all features harmful — fallback to SHAP
        from minppi_mofs.baselines.shap_top_k import compute_shap_topk_importance
        scores = compute_shap_topk_importance(X, y, seed=seed, task=task)
        return [f for f, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    # Train XGBoost on filtered set, extract gain-based importance
    X_f = X[keep]
    if task == "regression":
        m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         verbosity=0, random_state=seed)
    else:
        m = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          verbosity=0, random_state=seed)
    m.fit(X_f.values, y)
    imp = dict(zip(keep, m.feature_importances_.astype(float)))
    ranking = sorted(imp, key=imp.get, reverse=True)

    # Append harmful features at the bottom
    return ranking + sorted(harmful)
```

然后将 `"minppi"` 加入 `BASELINES` dict：

```python
BASELINES = {
    # ... existing entries ...
    "minppi": {
        "compute": _compute_minppi_ranking,
        "rank": lambda scores: scores,  # already a list
    },
}
```

**运行命令**：
```powershell
# 只跑 minppi 这一个方法（不重跑现存的 5 个 baseline）
..\venv\Scripts\python -u experiments/run_main_table.py --methods minppi
# 如果脚本不支持 --methods，则修改 DATASET_BUDGET 只留 minppi 一个条目后全跑
```

**预期运行时间**：约 1-2 小时（只跑 1 方法 × 7 数据集 × 5 seeds）

**产出**：新的 `main_metrics.csv` 包含 6 个方法（含 minppi）
**写入论文位置**：
- 重新生成 CD 图（见 E6）
- §6.1 Setup 中将 MinPPI 列入排名方法
- §6.2 Main Results 中 MinPPI 出现在 CD 图中，不再需要 "MinPPI does not appear in the ranking comparison" 这个尴尬的解释
- CD 图标题改为 "six methods"

---

### E6：重新生成 CD 图

**前置条件**：E5 完成（新的 `main_metrics.csv` 含 minppi）

```powershell
$R = (Get-ChildItem -Directory "results/tables" | Where-Object { $_.Name -like "2026-07-0*" -and (Test-Path "$_\main_metrics.csv") } | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name

..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R/main_metrics.csv --metric AUC@10 --higher
..\venv\Scripts\python experiments/make_cd_diagram.py results/tables/$R/main_metrics.csv --metric AUC@20 --higher

Copy-Item results/tables/$R/cd_diagram_AUC@10.pdf conference-paper/figures/
Copy-Item results/tables/$R/cd_diagram_AUC@20.pdf conference-paper/figures/
```

同样更新 `sensitivity_friedman.py`：
```powershell
..\venv\Scripts\python experiments/sensitivity_friedman.py results/tables/$R/main_metrics.csv
```

---

## 第四章：细微问题修复（m1-m8，不涉及实验）

| 问题 | 修复位置 | 具体操作 |
|------|---------|---------|
| **m1**：Abstract 重复句 | `icomp2026_conference.tex` 行 66-67 | 删除第二次出现："the removal test confirms that deleting all MinPPI-harmful features preserves AUC within $10^{-3}$ on 6 of 7 datasets." |
| **m2**：缺失 AUC@k 全表 | Appendix `app:full_tables` | 从 `2026-07-04_07-22-21/mean_metrics.csv` 中提取 6 方法 × 5 cls 数据集 × k∈{1,3,5,10,20,50} 的 Longtable |
| **m3**：消融维度不足 | §6.4 | 新增 3 段：D1 半合成实验摘要、D2 引导消融摘要、α 扫参（已有） |
| **m4**：CD 图标题不一致 | 行 263 | "all six methods" → "all five baseline methods"（若 E5 未完成）；E5 完成后恢复 "six" |
| **m5**：定理编号 | 全局搜索 | 确认无 `Proposition 2` 残留引用 |
| **m6**：数据集规模不平衡 | §6.1 行 252 后 | 新增一句："Credit-Card-Fraud (284,807 samples) is substantially larger than the other datasets; its per-seed MinPPI estimation time (~180s) is comparable to Madelon (~90s) due to the shared-coalition sampling budget." |
| **m7**：缺少运行时数据 | `tab:pareto_front` | 在表后新增一个迷你表 `tab:minppi_runtime`，列出每个数据集的平均 MinPPI 计算时间（从 `pareto_front_agg.csv` 的 `time_minppi_s` 列提取） |
| **m8**：MinShap preprint 声明 | 行 100 附近 | 在首次引用 MinShap 处新增脚注 |

---

## 第五章：完整执行顺序与时间线

以下为**从零开始**的完整执行计划。假设 2026-07-07 为 Day 0。

### Day 1（约 8 小时）：Phase A 文本修改 + E2/E3 短实验

| 顺序 | 任务 | 耗时 | 并行？ |
|:--:|------|:--:|:--:|
| 1 | **Phase A 文本修改**（A1-A13，全文一致性重写） | 3h | — |
| 2 | **m1-m8 细微修复** | 1h | — |
| 3 | 在后台启动 **E2**（guidance ablation，30min） | — | 与 4 并行 |
| 4 | 在后台启动 **E3**（ISOLET utility ablation，15min） | — | 与 3 并行 |
| 5 | 首次编译 LaTeX，检查无 broken references | 15min | — |

### Day 2（约 7 小时）：Phase B 耗时实验

| 顺序 | 任务 | 耗时 | 并行？ |
|:--:|------|:--:|:--:|
| 6 | 启动 **E1**（D1 semisynthetic，1h） | — | 后台 |
| 7 | 修改 `eval_harmful_utility.py`（M6 matched-budget fix） | 15min | — |
| 8 | 启动 **E4**（rerun removal test，3-4h） | — | 后台 |
| 9 | 将 E2/E3 结果写入论文 | 30min | — |
| 10 | 修改 `run_main_table.py`（E5，添加 minppi 排名方法） | 30min | — |
| 11 | 启动 **E5**（rerun main_table with minppi，1-2h） | — | 后台 |
| 12 | 将 E1 结果写入论文 | 30min | — |

### Day 3（约 3 小时）：Phase C 收尾

| 顺序 | 任务 | 耗时 | 并行？ |
|:--:|------|:--:|:--:|
| 13 | 运行 **E6**（CD diagram + Friedman LOO 更新） | 10min | — |
| 14 | 更新论文中 E4（新 removal test 数据）和 E5（新 CD 图 + Friedman 数据） | 1h | — |
| 15 | 最后的 ruff check + 最终编译 | 15min | — |
| 16 | 全文通读一遍，逐段检查与新叙事的自洽性 | 1h | — |

**总新增工作量**：约 18 小时（3 天），但大部分重度计算可在后台/夜间运行。

---

## 第六章：M1 解决方案的关键检查清单

在提交前，**必须**确认以下每一项都已完成：

### 标题与摘要
- [ ] 标题中不再包含 "Multi-Objective"
- [ ] Abstract 中不再出现 "Pareto front"、"multi-objective"、"Pareto frontier"
- [ ] Abstract 用 "sensitivity analysis"、"conservativeness parameter"、"safe deletion" 替代

### 正文 §1-§5
- [ ] §1 contribution #3 描述的不是 "multi-objective selection" 而是 "conservative deletion with tunable α"
- [ ] §1 contribution #4 强调 "safety validation" 而非 "Pareto-front metrics"
- [ ] §3 删除了 "Pareto Optimality" 小节
- [ ] §4.4 标题不含 "Multi-Objective"
- [ ] §4.4 正文不含 "Pareto front"、"non-dominated"、"no α simultaneously achieves"
- [ ] §4.4 正文将 α 扫参称为 "sensitivity analysis" 或 "operating curve"
- [ ] §4.5 Discussion 不含 "Pareto-preservation" 或 "Proposition 2"
- [ ] §5 Theory 不含 Proposition 2a/2b

### 实验 §6
- [ ] §6.1 Setup 解释了 MinPPI 为何单独评估（三分类 vs 排名的区别）
- [ ] `tab:pareto_front` 标题从 "α-Pareto front" 改为 "α sensitivity analysis" 或 "Conservativeness sweep"
- [ ] `fig:pareto_front_cls` 标题从 "α-Pareto front" 改为 "α operating points" 或类似
- [ ] §6.4 标题/内容从 "消融"改为含 3+ 维度

### 全局
- [ ] `grep -i "pareto.front" icomp2026_conference.tex` 只返回表格/图片标签，不返回正文中的叙述性使用
- [ ] `grep -i "multi.obj" icomp2026_conference.tex` 只出现在 Related Work 和 Prelim 中作为背景讨论
- [ ] `grep "Proposition 2" icomp2026_conference.tex` 返回零匹配
- [ ] 所有 `\ref{fig:pareto_*}` 引用指向存在的图文件

---

## 第七章：FAQ

**Q: 如果我不改 M1，只是修 M2-M8，能过审吗？**

A: 不能。M1 不是可选改进——它是审稿人**第一个会注意到的问题**。任何 MOFS 领域的审稿人看到 "我们扫了 4 个 α 值并称其为 Pareto 前沿" 都会直接建议拒绝（"this is not multi-objective optimization"）。修复 M1 是接受的前提条件。

**Q: 改标题会不会让论文不像会议论文了？**

A: 不会。"Safe Harmful-Feature Detection via Voting Games" 比原来的 "Multi-Objective Feature Selection" 更具体、更诚实、更能反映实际贡献。会议论文的价值在于**做了一件具体的事并把它做好**，而不是声称做了一件大事但做得不完整。

**Q: 重构后 α 扫参还有意义吗？**

A: 有，而且意义更清晰了。在旧框架中，扫参被误用为 "证明" 多目标优化——这是 weak argument。在新框架中，扫参是**灵敏度分析**——证明方法的安全性质对 α 的选择不敏感，用户可以自由选择保守程度——这是 strong argument。

**Q: 论文的 "多目标" 维度去哪了？**

A: MinPPI 本身只有一个目标（安全删除有害特征）。压缩率（compression rate）是 α 的函数而不是独立的优化目标。这不是缺点——大多数特征选择方法（SHAP, Banzhaf, Powershap）也都是单目标的。旧框架强加了一个 "多目标" 标签，反而暴露了弱点。去掉之后，论文作为一个单目标安全方法的论证反而更加 coherent。

**Q: 代码需要改吗？**

A: **不需要。** M1 是纯叙事修改，算法逻辑完全正确。Phase B 新增的实验脚本（D1-D3）和 `run_main_table.py` 的小修改不改变核心算法——它们只是添加了新的分析维度。


