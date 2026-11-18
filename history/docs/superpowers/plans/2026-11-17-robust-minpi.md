# Robust document minPI implementation plan

The user has authorized continued algorithm improvement. Existing score and total-time criteria apply.

The implementation changes the estimate inside each minPI peer-count stratum. Three training sketches contribute their held-loss endpoints. Each stratum takes the median before the minimum across zero and one peer is taken. The automatic positive-vote-mass budget remains defined by the current algorithm. The specification and frozen evaluation splits are recorded in ../../minpi-study/refined-document-protocol.md.

The work uses the current Python, NumPy, SciPy and optional prebuilt C backend in history. Every Python command uses this checkout's thesis-python.ps1. Native and reference predictions must agree. Timing includes all repeated sketches, column slicing, training and prediction.

- [x] Implement and explicitly verify repeated_document_minpi.py against per-feature held ridge losses.
- [x] Archive unsuccessful full-row, leave-one-out, standardized and joint-model proposals on the four development configurations.
- [x] Implement robust_document_minpi.py and document_repeated_kernel.c with shared full-training tie statistics. Hash both the new C source and its included predecessor in build_native_repeated_document.py.
- [x] Verify native/reference gains and supports for binary, multiclass fallback, groups, forward groups, count matrices, ties and singleton inputs.
- [x] Run refined_document_benchmark.py for the frozen existing twenty configurations and three Turkish SMS splits in two independent timing processes, then replay supports and predictions.
- [x] Recompute same-configuration successes from raw trials with audit_refined_document.py and keep development, previous validation, and new external evidence identifiable.
- [x] Add the tested rule to PivotalSupportSelector and test its normal fit/transform interface, including sparse values and seeds.
- [x] Report all outcomes and how the changed estimator affects width, predictive score, total time and seed sensitivity.

Review focuses on retaining training-only fitting, preserving group isolation, using a minimum after stratum aggregation, avoiding duplicate full-data scans, rejecting invalid native indices, and comparing candidate-specific matched widths fairly. A fixed random seed guarantees computational replay. Measured score stability across seeds requires its own experiment.

## Follow-up driven by the measured cost

The median-three candidate compressed every one of the 23 evaluated text configurations, but its extra voting cost prevented a consistent improvement over the single-sketch method. The 24-configuration seed recheck also failed to establish uniformly smaller score variation. These outcomes remain archived. The follow-up fixes the zero/one-peer rule to the minimum of the two existing cross-fit directions, avoiding additional sketches. Its formula is frozen in ../../minpi-study/reliability-document-protocol.md before the follow-up evaluation. This is a disagreement penalty, not a calibrated confidence bound.

- [x] Implement reliability_document_minpi.py and check its endpoints against explicit ridge fits and its native/NumPy agreement.
- [x] Complete two timing processes on the existing twenty configurations and the three Turkish SMS rechecks.
- [x] Verify those outputs in fresh processes and compare each actual width against twenty random supports.
- [x] Evaluate genuinely new Turkish email data with the frozen rule, preserving the original archive's 825 labelled records and documenting its discrepancy with the advertised 826.
- [x] Recheck 24 short-spam seed configurations and the six original CNAE/Dexter configurations, retaining all failed cases.
- [x] Test both new public rules against their actual core implementations, including grouped multiclass execution and the missing-native fallback.
- [x] Publish an evidence-linked experimental summary without changing the frozen conference manuscript or claiming success on unsupported continuous-valued datasets.

The implementation and this evaluation iteration are complete, with 350 related tests passing and all 1,048 method outputs replayed. The broader empirical goal remains unmet: the first Turkish-mail source achieved 0/3 full-pipeline successes, Dexter regressed to 0/3, and 5/8 short-message folds had wider seed score ranges. See ../../minpi-study/reliability-document-results.md. No universal improvement or stability claim follows from completing these tasks.
