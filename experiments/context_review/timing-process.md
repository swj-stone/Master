# Timing an entire selection and refit

This follow-up uses the 42 previously evaluated PMLB folds. It compares the fixed restoration order, the order that promotes a recent veto, and a classification-specific implementation of the same recent-veto order. All three use the identical four ranking panels and minimum vote. The [locally fixed design](../../configs/context_review/timing_process.json) specifies the methods, endpoints and decision criterion.

When both requested masks are missing from the cache, the third implementation makes one prediction call on their stacked observations. Each distinct mask is still evaluated. When every panel violates the accuracy allowance, all four complete metric votes are false regardless of the AUC or average precision, so their computation can be deferred. A later comparison may need a full score for the cached mask. The shortcut requires a predictor whose predictions are independent of the other rows in a batch, finite class probabilities, and the same weighted accuracy definition and tolerance in the full scorer. [Implementation](../../src/minppi_mofs/lazy_ballot.py)

The in-memory timer starts after reading the raw rows, encoded labels and frozen split indices. Each timed arm prepares the splits, imputes, trains the initial 100-round tree ensemble, chooses its Gain anchor, runs the deletion search and refits a fresh predictor. It ends after outer prediction. Verification, file access and Python startup are outside this interval. Fold workers were bound to logical CPU 2 on this shared Windows 11 machine (Intel64 Family 6 Model 198 Stepping 2). XGBoost used one thread. Wall time includes any competing activity. Process CPU time on Windows has coarser resolution, particularly for short stages. Three cyclically ordered repetitions put each arm in each sequence position. Every arm starts with empty mask caches.

For each timed run the verifier compared the proposal path and final support with the archived study, the outer probabilities element by element, and each four-panel vector for each visited restoration against an ordinary unbatched evaluator. The observations retain hashes of all visited ballots. A given fold's recent and lazy implementations must give the same hashes across all six runs. None of the four binary PMLB tasks triggered the rare-positive average-precision criterion. A separate regression test exercises that path with 4% positive labels.

Run from the thesis checkout with the original PMLB receipt and source tables available.

```powershell
.\thesis-python.ps1 -m experiments.context_review.timing_process
.\thesis-python.ps1 -m experiments.context_review.timing_report
.\thesis-python.ps1 -m pytest -q tests/test_lazy_accuracy_ballot.py
```

The [result directory](../../results/context_review/timing-process/) contains the nine independent timing records for each fold. The analysis takes the median of three wall or CPU measurements within a fold, then averages the three folds within each dataset. Each of the 14 named tasks enters a task-level ratio summary once. The work counts sum one pass per fold. Repeating a deterministic search does not increase the number of independent feature-selection problems.

| Implementation | Mean complete wall time (s) | Mean search wall time (s) | Masks in one pass | Prediction calls | Full score computations |
|---|---:|---:|---:|---:|---:|
| Fixed order | 0.4814 | 0.10982 | 3,304 | 3,304 | 3,304 |
| Recent veto | 0.4731 | 0.09964 | 3,004 | 3,004 | 3,004 |
| Recent veto, batched predictions and deferred scores | 0.4669 | 0.09377 | 3,004 | 2,770 | 2,436 |

Recent veto saved 193 of its 300 distinct masks on Movement Libras, with zero net change on eight tasks. Adding batching and deferred scoring saved 234 prediction calls and 568 full score computations per 42 folds. Its search wall time was lower on 13 tasks than recent veto alone. The median task-level search time ratio was 0.9100. Its full-pipeline wall time ratios relative to the fixed and recent orders had medians of 0.9788 and 0.9868, respectively, with lower wall times on 12 tasks in each comparison. Both missed the prespecified 5% threshold. The paired median full CPU time ratio was 1.000 for every method pair. Conditions such as single-process startup, filesystem overhead or a different learner require separate measurement.
