# External transfer benchmark

The two archived NIPS 2003 challenge files with labeled validation splits, Arcene and Dexter, were selected for a separate dense-matrix stress test at 10,000 and 20,000 columns. Dorothea, with 100,000 sparse columns, lies outside this experiment's dense implementation. The public validation file serves once as a held-out evaluation split per dataset. Competition test outcomes are unavailable here. The two data sources have independent sample units; seed repetitions within a source reuse exactly the same observations.

Before running the benchmark, fix the retained fraction at 10%, the feature-column permutation at seed 391, and the three algorithm seeds at 1129, 2039, 3049. Randomly permute columns before selection so that deterministic tie-breaking cannot use the original challenge order. Every selector uses training labels only. Compare exact minPI, Fisher's univariate F score, binary ReliefF (ten nearest hits and misses per training instance, all training instances as queries, range-normalized Manhattan distances), absolute L1 logistic coefficients, uniformly random columns, and a full-column reference. The classifier in all arms is standard deviation scaling followed by an L2 logistic regression (C=1). Each comparator receives the same column budget except for the explicitly full-width reference. Report ROC AUC, balanced accuracy and elapsed selection plus final model fitting and prediction time. Class-stratified paired bootstrap intervals compare exact minPI and the F filter on each evaluation split; these intervals condition on fitted models and do not include training-set uncertainty. Do not use either split to revise the method.

Data archives and SHA-256 digests

| File | SHA-256 |
|---|---|
| arcene.zip | c7a40e39edf23a9b744eb570592d6e917fd9a1976aaf20ad581e09afd47d5220 |
| dexter.zip | 70400df292178d8d3627411a5631a4cb13fb6b19da4a200b9a206f25255cddcc |

The competition organizers describe the split and its historical use in [Guyon et al. (2004)](https://papers.nips.cc/paper/2004/hash/5e751896e527c862bf67251a474b3819-Abstract.html).
