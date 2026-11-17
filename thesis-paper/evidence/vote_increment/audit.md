# Incremental value of minimum voting

The previous selector has been rejected as the candidate method. Record every subsequent design tested and its development or prospective status. Declare the dataset roster, splits, learner, cap and guard before opening test scores. Report all registered datasets. The six original prepared datasets and both previously released external panels now supply development evidence.

## Operational gate

The unit for the 80% target is a dataset. First compare the three methods on each identical source and seed. Against **each** comparator, require held-out macro AUC or R² no more than 0.01 below the comparator. Then require either fewer retained columns, or fewer actual model fits **and** shorter total elapsed time. On the time route, additional retained width must stay within the earlier allowance of one column or 5% of comparator width, whichever is larger. The width route can cost more time; report its time and fit differences. A timing fluctuation with unchanged fits and width counts as zero wins.

Count a seed only if the same MinPI run beats both mean voting and direct Gain under this rule. A dataset passes on a strict majority of its registered seeds. A one-seed pilot describes a source decision without estimating seed stability. Report wins against each comparator separately as diagnostics. The released method must pass at least the ceiling of 80% of the prespecified sources. A post hoc union of different successful variants or sources does not qualify. Report the stricter no-worse-width, no-worse-fit, no-worse-time sensitivity and the result of demanding at least 5% lower wall time on the cost route. Show accuracy or raw MAE for every source; AUC alone can miss material label errors on ISOLET.

## Fair controls

The three arms receive the same maximum number of native refits, predictor family, fitting rows, common validation rows, utility, tolerance, tie and acceptance rules, resolution of missing measurements, and final refit policy. Stop evidence and stopping rules must make sense for every arm. Any posterior probability or uncertainty estimate used to adapt the schedule needs a corresponding implementation or a fixed matched ablation. If two procedures differ in both score aggregation and width-search rule, retain that result as a pipeline comparison and cross both factors to attribute an effect to the minimum.

For the mean ablation use the identical per-scale observations, feature candidates, sampling randomness, score normalization and masks as for the minimum. The only deliberate change is how the size votes are aggregated. For direct Gain, use the native model's training gain and its documented tie rule. Report a cheap competent direct implementation allowed to reuse identical fits. Explicitly report the running budget or tolerance when an arm stops early. Perform no dummy fits. Count all label-trained helper models and all native refits separately. Include sampling, ranking, construction and evaluation of candidates, final native refit and predictions in standalone time. Cache only within an arm for paired timing. Count unique supports trained, not requests for already cached results. Timing-only wins require the fewest-retraining route; for a final timing claim, rotate execution order, warm up, pin hardware and verify decision parity between repeats.

## Certified computation and prediction

A partial minimum is a monotone upper bound on the completed minimum. Once that upper bound falls below a completed incumbent, its computation can terminate. This saves vote evaluations, not downstream native model fits unless the skipped operation itself is a paid native refit. A bound for a masked or approximate fixed predictor cannot certify the score of a newly trained predictor. Distinguish these costs in each receipt.

## Replay of the rejected version

The paired checker `experiments.vote_increment.audit` applied to the original R3 seed tables gives **0/6** original source passes (**1/30** simultaneous seed passes) and **0/5** prospective source passes (**0/25** simultaneous seed passes). Only an ISOLET seed passed against both controls. Apply the same gate to subsequent versions.

## First signed-game development audit

The restricted-subspace tree votes (seed 42, signed/subspace-s42/scores.csv) pass **0/6** original datasets against both controls. Each voting arm uses 52 auxiliary tree fits beyond its reported native XGBoost fits. Count both for a saved-fit claim and show the two components separately. Madelon retained three fewer columns than mean voting but lost 0.05425 AUC relative to it. On ISOLET the minimum retained 72 extra columns against mean voting and 55 against direct Gain. These measurements rule out promoting this candidate on the registered criterion.

A separate threshold-selection hypothesis uses size votes normalized to mean one, followed by the same quota cutoff in all arms. Minima of normalized shares typically sum to less than the corresponding means. Run a matched-scale or exact-width replay under mean ranking when reporting a gain from this hypothesis. Its result distinguishes a more informative ranking from a more aggressive width proposal.
