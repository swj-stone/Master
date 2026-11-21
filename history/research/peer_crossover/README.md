# Single-peer and nested-peer at both feature widths

[Results and matched baselines](evidence/summary.md) · [Per-configuration scores, widths and paired timings](evidence/paired-cases.csv) · [Per-dataset comparisons](evidence/by-dataset.csv)

The [six-page figure set](figures/crossover-curves.pdf) shows the [single-peer](figures/single-curves.png) and [nested-peer](figures/nested-curves.png) comparisons by input width, followed by the [four routing policies](figures/routing-curves.png), their own width-matched baselines, and the [distribution of retained width](figures/retained-widths.png). The full-range curves count all configurations faster than Full in both timing processes, ordered by signed test-score loss. The enlarged interval from 0 to 0.10 also requires feature removal. The denominator includes all configurations in the displayed group.

This experiment applies both rules in the current manuscript to each of the 387 configurations in the [paper's input register](../lowdim/evidence/all-cases.csv). The original grouping gives 56 configurations with at most 32 input features and 331 with more. The predictor, split, seed and primary test metric are held fixed within each configuration. Both rules select their own width using their training inputs. The wide single-peer implementation evaluates all single peers, including the contribution of constant columns.

Each rule is compared with the other rule, the full-feature predictor, a univariate F filter at the rule's width and a uniform random subset at that width. The matched baselines receive the width without being charged for the other rule's selection. Total time includes column selection, data preparation inside the original pipeline, fit and prediction. Dataset loading, warm-up and calculation of the test metric are outside the timed pipeline. Comparisons use within-process median times. A stable speed gain requires a strictly lower total time in both processes.

The four candidate policies use one fixed rule on each side of the 32-feature boundary. We report the number of configurations where the policy removes at least one column, beats Full on total time and stays within an absolute test-score loss of 0, 0.02, 0.05 or 0.10. We also report the corresponding count with no score constraint. The 0.02 column matches the reference point in the manuscript. A reduction in balanced accuracy, ROC AUC or regression R² is measured in that metric's own units. The results count configurations, including repeated seeds or splits. Ranking the four policies on this benchmark is descriptive.

The independent [reference check](verify_native_reference.py) compares wide single-peer columns and scores against a separately compiled copy of the original voting kernel with its static array bound raised to 128. It runs outside benchmark timings.

## Reproduction

Run from `history/` with the thesis interpreter. Each benchmark process warms all active methods once and times three deployments per method in rotated order. The second process reverses the order. When both rules choose the same width, the two F labels share one timed F run and the two random labels share one timed random run. The study writes one complete CSV per case per process and resumes from completed cases under the same protocol fingerprint.

```powershell
.\thesis-python.ps1 -m research.peer_crossover.verify_native_reference
.\thesis-python.ps1 -m pytest research/peer_crossover/test_single.py -q
$runPath = 'results/minpi-study/peer-crossover-20261121a'
.\thesis-python.ps1 -m research.peer_crossover.runner $runPath --process 1
.\thesis-python.ps1 -m research.peer_crossover.runner $runPath --process 2
.\thesis-python.ps1 -m research.peer_crossover.summarize $runPath --out research/peer_crossover/evidence
.\thesis-python.ps1 -m research.peer_crossover.plot_widths
.\thesis-python.ps1 -m research.peer_crossover.plot_curves
```
