# A budget for one nominated candidate

The comparison uses the same original Gain XGBoost slates as the transport experiments. Point uses one panel on the empty restoration, full margin evaluates the original four-panel family with a provably exact early cutoff, and the eight-round control fits a small model on the anchor and each candidate. After nomination, every actionable arm checks the same two full-size fits on the guard (anchor and nominee). It fits the returned support on development rows and predicts on the outer features. No outer label enters these stages.

Three repetitions used cyclic rotations of a shuffled order within each fold, so every arm was first, second and third once. An arm rebuilt its own Gain anchor and model. The two reported clocks measure nomination and complete execution. Disk I/O, verification, outer-label scoring and the retrospective full-slate research fits are outside both clocks. Totals sum within-fold medians; percentage outcomes instead average folds within source, then weight sources equally. The initial timing source is retained byte for byte with its SHA-256 and machine details. The later metadata writer records XGBoost and the pinned CPU directly. The clock observations are unchanged.

| Source panel | Rule | Effective compression | Breach | Nomination, s | Complete, s |
|---|---|---:|---:|---:|---:|
| Ten tasks | Point | 7.29% | 6.67% | 0.13 | 7.87 |
| Ten tasks | Full margin | 7.29% | 13.33% | 0.43 | 8.11 |
| Ten tasks | Eight rounds | 6.33% | 13.33% | 1.76 | 9.53 |
| Six wider tasks | Point | 7.34% | 16.67% | 0.17 | 11.29 |
| Six wider tasks | Full margin | 7.34% | 16.67% | 0.89 | 12.11 |
| Six wider tasks | Eight rounds | 10.13% | 16.67% | 1.68 | 12.78 |

Run `.\thesis-python.ps1 -m experiments.context_review.budget_comparison replay` from the thesis checkout to check every clock record's nominal and returned support, final prediction digest, fit counts and component durations. The `time` action requires the original input matrices and measures the policies afresh. It checkpoints each complete fold and prints elapsed and estimated remaining time. [Recorded clocks and source-level contrasts](../../thesis-paper/evidence/budget_comparison/)

## Check a proposed comparison before outer scoring

Run `.\thesis-python.ps1 -m experiments.context_review.plan_comparison --original-wide --target-pp 5` to reconstruct the original 2.78-point upper bound from the two policies' guarded supports. For a new study, pass `--input folds.json`. For each fold include the anchor, source identifier, fold identifier, and the probabilities of each possible return after the guard for the left and right policies. Represent an ordinary deterministic return with probability one and a fallback with the full anchor. Reversed feature order and repeated appearances of the same support refer to the same outer pass bit. Probability strings such as `"1/3"` support exact lotteries.

```json
[{
  "source": "example", "fold": 0, "anchor": ["a", "b", "c"],
  "left": [{"columns": ["a", "b"], "probability": 1}],
  "right": [{"columns": ["a", "b"], "probability": "1/2"}, {"columns": ["a", "c"], "probability": "1/2"}]
}]
```

This example bounds the expected left-minus-right effective compression between −16.67 and +16.67 percentage points. The calculation requires both policies' guard returns in a comparison study. It reads no outer labels. It uses only support identity, savings and selection probabilities, so known dependencies among the fitted predictors can tighten it. A desired mean exceeding the upper endpoint is impossible on the supplied folds; lying below it does not establish power or guarantee a gain.
