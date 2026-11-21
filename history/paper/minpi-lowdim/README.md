# Feature selection with single-peer ridge gains

[English manuscript (PDF)](main.pdf) · [Editable LaTeX source](main.tex)

| Manuscript content | Data and implementation |
|---|---|
| Table 1. Data sources, splits, and configuration counts | [Experimental protocols](../../research/lowdim/comparison-protocol.md) · [Input fingerprints](../../research/lowdim/evidence/distinct-inputs.csv) |
| Table 2. Retained widths, test scores and end-to-end runtimes | [Matched measurements on Abalone and Diabetes](../../research/peer_crossover/evidence/paired-cases.csv) |
| Table 3. Results by prediction task | [All 387 paired configurations](../../research/peer_crossover/evidence/paired-cases.csv) · [Single-peer and its comparators by primary metric](../../research/peer_crossover/evidence/by-metric.csv) |
| [Figure 1. Time saved on the 56 low-dimensional configurations](figures/lowdim-runtime.pdf) | [Generator](plot_runtime.py) · [Selection and end-to-end timings](../../research/peer_crossover/evidence/paired-cases.csv) |
| [Figure 2. Fixed-width comparisons and exhaustive Abalone subset scores](figures/subset-quality.pdf) | [Generator](plot_subsets.py) · [31 ablation configurations](../../research/lowdim/peer_evidence/cases.csv) · [45 subset scores per split](../../research/lowdim/peer_evidence/abalone-subset-scores.csv) |
| [Figure 3. Single-peer and its width-matched baselines at all input widths](figures/matched-curves.pdf) | [Generator](plot_matched_curves.py) · [Scores and two-process timings](../../research/peer_crossover/evidence/paired-cases.csv) |
| [Companion plot. Single-peer score loss and end-to-end time against each comparator directly](figures/score-runtime.pdf) | [Generator](plot_tradeoff.py) · [Paired results](../../research/peer_crossover/evidence/paired-cases.csv) |
| Peer-layer ablation and 32 random subsets per configuration | [Selected-set and score comparisons for 31 configurations](../../research/lowdim/peer_evidence/cases.csv) · [Exhaustive comparison of the 45 eight-column Abalone subsets](../../research/lowdim/peer_evidence/abalone-all-subsets.csv) |
| Single-peer implementation for all input widths | [Selector entry point and wide-matrix implementation](../../research/peer_crossover/single.py) · [Tiled reduction](../../research/peer_crossover/all_peers.c) · [Original small-matrix kernel](../../research/lowdim/native_votes.c) · [Independent numerical check](../../research/peer_crossover/verify_native_reference.py) |
| Background reading | [Notes on the two reference papers](../../research/theory/README.md) |

The primary method is `research.peer_crossover.single.select_single(X, y, task=..., seed=...)`. The score, ranking and width rule are the same on both sides of 32 input columns. The crossover records label the primary method `single`, the full-feature predictor `full`, and the two comparisons at the single-peer width `F_single` and `random_single`. The `nested` arm is an internal comparator. The [replay protocol and commands](../../research/peer_crossover/README.md) reproduce both independent timing passes. Each pass has one warm-up and three timed deployments per distinct arm. Figure 2 draws on a separate fixed-width ablation using the same single-peer subsets and primary scores on those 31 configurations.

For Table 2, first take the median end-to-end runtime of the three repeated measurements within each process, separately for each method and configuration. Average the two process medians to obtain one runtime per configuration, then take the median across configurations for each dataset. Test scores in the table are arithmetic means over configurations within the same dataset. A configuration counts as faster only if both within-process comparisons show a shorter end-to-end time. Test-score loss in Figure 3 is the full-feature model score minus the score of the subset model. Full retention counts as a failure to reduce width. The rare-event example also uses [average precision and detection counts on the same credit-card split](../../research/lowdim/fraud_evidence/cases.csv).

The random-label simulation has the literal dataset name `null`. In pandas, read the CSV files with `keep_default_na=False` to preserve that name.

## Reproduce the figures

From the `history` directory, run the following commands to regenerate the vector figures and their PNG previews from the recorded measurements. Figure 2 uses one copy of each Abalone split. Its 135 subset scores were extracted from the original per-seed `exhaustive_scores` arrays in `results/minpi-study/lowdim-peer-ablation-20261118a/`.

```powershell
.\thesis-python.ps1 paper/minpi-lowdim/plot_runtime.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_subsets.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_matched_curves.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_tradeoff.py
```

## Build the PDF

Run XeLaTeX twice from this directory to resolve the references. The bibliography is included in the source.

```powershell
xelatex -halt-on-error main.tex
xelatex -halt-on-error main.tex
```
