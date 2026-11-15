# Inputs for the refit reassessment

These small files accompany the [paper](../../minpi.pdf). The [study index](../../../experiments/context_review/README.md) describes protocols and full prediction archives. Each row of `decisions.csv` is one candidate within fold and proposal source. A positive `fail` indicates an observed loss relative to the refitted anchor. A positive rule column indicates a veto on the original fitted predictor.

| Files | Quantities |
|---|---|
| `decisions.csv`, `exposure.csv`, `incremental-vetoes.csv` | Candidate outcomes, paired rules, shared supports and task contributions for the two proposal sources |
| `eligibility.csv`, `refit_guard.csv`, `challenger.csv`, `challenger_costs.csv` | Guard splits, final scores and counts of unique masks to nominate another deletion |
| `factorial.csv`, `timing.csv`, `timing_counts.csv` | Historical mechanism replay and paired complete-process timing under controlled order |
| `har.csv`, `har_by_subject.csv`, `har_anchor_random.csv`, `drift.csv`, `drift_batches.csv` | Official participant holdout and future-batch results with each reporting unit identifiable |
| `har_history.csv`, `gas_history.csv` | Fixed-width baselines with different amounts of labeled selection history. All used their original final fitting rows. |
| `refit_grid.csv`, `refit_grid_costs.csv` | Three decision rules applied to the same short list of genuinely refitted prefixes. All selection scores used disjoint guard rows. |

[Regenerate the two tables and the incremental-veto figure](../../reassessment_results.py) from this directory's CSVs. Running the generator with `--collect` first refreshes the copies from the complete experiment folders. The [operating-rule evidence](../operating/) covers the later rule, cost and guard comparisons. The [archive manifest](../revision/artifacts.json) gives the SHA-256 digests of the delivered artifacts.
