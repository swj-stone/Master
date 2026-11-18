# minPI combination and shared-budget exploration

The user authorizes continued algorithm exploration with the same practical objective: a useful accuracy, time and selected-feature tradeoff against actual baselines. Prior authorization covers implementation and local experiments. Existing conference files and archived runs remain unchanged. Work uses history/thesis-python.ps1 and shared installed dependencies.

Three mechanisms are examined. The first replaces positive-vote mass with the actual calibration loss of nested supports in a cheap fixed predictor. The second proposes a bounded number of conditional swaps and checks the entire exchange block using exact squared-loss changes. The third returns to the original grouped nearest-neighbor pivotal minPI: make its representative pool independent of requested width, compute the vote ordering once, and reuse dual ridge sufficient statistics across the candidate prefixes. Existing ISOLET evidence motivates the third mechanism, but it is development evidence and is not a new holdout.

The grouped prototype uses a fixed 0.90 correlation filter, the existing 0/4/16 peer strata, and at most 256 witness rows. Ranking and representative selection must exclude the inner calibration observations. Pilot fitting uses at most 512 training rows, validation at most 256 separate rows, with stratified sampling where possible. A candidate grid in eighths of the original width, plus the representative count when it is a genuine reduction, is fixed before evaluation. Each prefix uses the same pilot means/scales. Its dual Gram and held-to-fit cross-product matrices grow incrementally; every returned prediction must match a directly fitted standardized unit-ridge classifier. The shortest prefix within 0.02 balanced accuracy of the full pilot may be selected. An unavailable acceptable prefix returns full width, counted as a failure. These are empirical training checks, not a test-performance guarantee.

Initial development is limited to four pre-existing text development configurations for the first two prototypes, and ISOLET first fold/seed plus Digits first fold/seed for grouped development. No candidate is chosen from external test scores. Original 69 and extra 57 sets have already been examined and constitute historical rechecks. Further formal evaluations are frozen only after an implementation is chosen. No broad outcome is promised by a prototype.

- [ ] Diagnose the mismatch between summed positive marginal votes and the joint predictive loss; record a concrete counterexample.
- [ ] Prototype the calibration budget and bounded combination exchange without changing archived algorithms.
- [ ] Implement shared grouped minPI and incremental prefix calibration; verify against direct per-prefix fits and training/validation row isolation.
- [ ] Run the designated development checks with full, old grouped, no-vote representatives, matched F/random and independently calibrated F controls.
- [ ] Select any justified follow-up using the frozen development evidence and measure all selection/calibration costs in complete pipelines.
- [ ] Report raw accuracy/time/width and tolerance curves for 0, .005, .01 and .02 loss; distinguish exact Pareto dominance from tolerated score loss.
- [ ] Preserve negative experiments and state which algorithmic mechanism, if any, improves the tradeoff.

Review focuses on actual use of voting, duplicated model cost, identity of paired baseline supports, no calibration labels in ranking, honest full-retention failures, and preserving group/time separation. Main one-deployment timing must never hide calibration costs; reuse is a separate measured scenario.
