# Can Masked-Model Votes Predict Which Feature Deletions Exceed a Refit Loss Tolerance?

The unit of the primary comparison is a proposed retained set within a fold. The outcome compares two fresh models trained on the same development rows, one on the proposed set and one on its reference anchor. The rules being compared all vote on the same proposals. [Loss definition and sampling](../../configs/context_review/refit_target.json) · [Eight-rule comparison](../../results/context_review/refit-target/counts.csv)

| Experiment | Protocol and records |
|---|---|
| Change the restored columns, evaluation rows and bootstrap weights separately on six historical tasks | [Crossed replay](factorial-replay.md) · [Exact two-feature prediction and falsification](../../thesis-paper/audit/pair-model-protocol.md) |
| Test 400 search-path proposals and 379 separately drawn deletions on 14 previously evaluated PMLB tasks | [Selection and refits](refit_target.py) · [Per-task effects and sensitivity to loss thresholds](../../thesis-paper/audit/refit-review.md) |
| Check the original direct Gain comparator, including the validation score of every width | [Validation objective](../../configs/context_review/exhaustive_gain.json) · [1,209 evaluated widths on 42 folds](../../thesis-paper/audit/refit-review-gain.csv) |
| Give the voting support and two Gain prefixes the same opportunity to be checked by refitting | [Guard protocol](../../configs/context_review/refit_guard.json) · [Eligible folds](../../results/context_review/refit-guard/eligibility.csv) · [Per-fold scores](../../results/context_review/refit-guard/per-fold.csv) |
| Try one further deletion using the candidate's worst restoration vote, pruning as soon as it cannot win | [Challenger protocol](../../configs/context_review/challenger.json) · [Nomination cost](../../results/context_review/challenger/costs.csv) · [Proof of the pruning bound](../../thesis-paper/formal/Challenger.lean) |
| Time complete selection and refitting under three equivalent implementations | [Timer boundaries, ordering, machine and repetitions](timing-process.md) |
| Predict activities for people absent during selection, then test transfer to later batches on a gas sensor array | [Human activity holdout](external_har.md) · [Forward drift study](drift.md) |
| Give conventional selectors access to different histories for selecting columns, with identical final model training | [All 14 retrospective controls and timings](matched_history.md) |
| Cast votes on genuine refit predictions at four candidate widths and take the minimum across class and global contexts | [Protocol, results and limits](refit_vote_grid.md) · [Per-fold scores](../../results/context_review/refit-vote-grid/per-fold.csv) · [Independent grid reconstruction](../../thesis-paper/audit/refit-review-grid.py) |

The PMLB follow-ups reuse the recorded splits of the [14-task benchmark](../context_study/README.md). The historical six-task replay additionally needs the archived frozen-model predictions and masks listed in the [original diagnostic](../context_diagnostics/README.md). The UCI studies obtain hash-checked archives. In the initial HAR, gas-sensor and Optdigits comparisons, the local scorer read the public test labels after predictions were saved. Later policies on these same tasks were retrospective.

## Proposal-level diagnosis and guarded deployment

| Experiment | Estimand and comparison |
|---|---|
| [Clustered uncertainty and error costs](cluster_cost.md) | Pooled and task-weighted detection and false-veto rates on 400 search-path and 379 drawn proposals, with paired resampling. Against endpoint tests, the full minimum broke even at a 20/11 miss-to-false-alarm cost ratio on the observed paths. |
| [Operating-rule sensitivity](rule_sensitivity.md) | 360 settings per voter and two refitted learners. Changing the refitted learner changed observed failure on 111 of the 400 search proposals. |
| [Cost transfer](cluster_cost_transfer.md) | Choose a voting rule or a size threshold using 13 tasks and charge its errors on the remaining task. On drawn proposals the size library had lower pooled held-out cost at all four miss costs examined. |
| [Incremental prediction](incremental_prediction.md) | Add worst-context dissent to a size-based risk model trained on other tasks. Ranking AUC improved by 0.024 on paths and 0.046 on drawn proposals. The paired log-loss intervals crossed zero. |
| [Fixed-budget selection, learned nomination and stand-alone clocks](budgeted_guard.md) | Compare actual deployed supports with equal candidate-refit budgets on 41 PMLB folds and separately on Optdigits. At half width, adding votes raised task-mean compression within tolerance by 0.79 points and PMLB breaches from one to three. All 168 order-balanced executions verified nominees, supports and predictions. |

## Run the cost, risk and deployment follow-ups

First complete the PMLB refit-target and refit-guard campaigns below. Cost transfer reads the rule-sensitivity grid. The learned guard uses the fitted incremental models and fixed-budget candidate pools. Reserve the timing session for the two policies being compared.

```powershell
.\thesis-python.ps1 -m experiments.context_review.cluster_cost
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity --jobs 4
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity_report
.\thesis-python.ps1 -m experiments.context_review.rule_sensitivity_audit
.\thesis-python.ps1 -m experiments.context_review.cluster_cost_transfer
.\thesis-python.ps1 -m experiments.context_review.incremental_prediction
.\thesis-python.ps1 -m experiments.context_review.incremental_prediction_audit
.\thesis-python.ps1 -m experiments.context_review.budgeted_guard
.\thesis-python.ps1 -m experiments.context_review.budgeted_guard_report
.\thesis-python.ps1 -m experiments.context_review.learned_guard
.\thesis-python.ps1 -m experiments.context_review.guard_clock
.\thesis-python.ps1 thesis-paper/audit/operating-guard-audit.py --learned
.\thesis-python.ps1 thesis-paper/operating_results.py --collect
```

The [operating evidence index](../../thesis-paper/evidence/operating/README.md) identifies each portable table. The [operating archive](../../packages/thesis-operating-evidence.zip) accompanies the three earlier evidence packages. [Manuscript and formal checks](../../thesis-paper/README.md) · [Artifact hashes](../../thesis-paper/evidence/revision/artifacts.json)

## Replay the per-proposal and per-fold comparisons

In PowerShell at the thesis checkout, use the project's launcher. First obtain the upstream data and original fold receipts as described in the linked benchmark instructions. Executing these commands with locally verified results already present reuses the completed folds. The audit commands below independently reconstruct quantities from the source data.

```powershell
.\thesis-python.ps1 -m experiments.context_review.refit_target
.\thesis-python.ps1 -m experiments.context_review.exhaustive_gain
.\thesis-python.ps1 -m experiments.context_review.adjudication
.\thesis-python.ps1 -m experiments.context_review.refit_vote_grid
.\thesis-python.ps1 thesis-paper/audit/refit-review.py target --fresh --ballots
.\thesis-python.ps1 thesis-paper/audit/refit-review.py gain
.\thesis-python.ps1 thesis-paper/audit/refit-review.py guard
.\thesis-python.ps1 thesis-paper/audit/refit-review.py challenger
.\thesis-python.ps1 thesis-paper/audit/refit-review-ranks.py
.\thesis-python.ps1 thesis-paper/audit/refit-review-pairs.py
.\thesis-python.ps1 thesis-paper/audit/refit-review-grid.py
```

The adjudication entry point runs both the guard and challenger using the same eligibility check. Among the six ranking-set classes on Wine Quality Red fold 2, one appeared only once. No stratified division into two disjoint halves could represent that class in both. Each table therefore covers 41 folds and also gives an analysis restricted to the 13 fully observed tasks. During initial development of the repair, writing a NumPy Boolean to JSON stopped the first worker before any outer scoring. The original local receipt is in `results/context_review/guard-before-serialization-fix/`. The evaluated run uses the corrected serialization.

## Inspect results without refitting

The [portable tables](../../thesis-paper/evidence/reassessment/), [independent audit](../../thesis-paper/audit/refit-review.md), and [manuscript build instructions](../../thesis-paper/README.md) give the inputs used in the paper. The two proposal sources overlap on 75 fold–support pairs. Standard errors that treat all 779 entries as independent would disregard shared test rows, anchors and supports. The held-out person and the future batch are the relevant reporting units in the two external studies. Feature count measures neither device count nor acquisition cost.
