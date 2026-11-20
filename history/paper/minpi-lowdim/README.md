# Conditional ridge gains for low-dimensional feature selection

[English manuscript (PDF)](main.pdf) · [Editable LaTeX source](main.tex)

| Manuscript content | Data and implementation |
|---|---|
| Table 1. Data sources, splits, and configuration counts | [Experimental protocols](../../research/lowdim/comparison-protocol.md) · [Input fingerprints](../../research/lowdim/evidence/distinct-inputs.csv) |
| Table 2. Retained widths, test scores, and end-to-end runtimes for five methods | [Scores and process-level timings for each configuration](../../research/lowdim/evidence/all-cases.csv) |
| Table 3. Paired comparisons for binary classification, multiclass classification, and regression | [All 387 paired comparisons](../../research/lowdim/evidence/all-cases.csv) · [Results by task and metric](../../research/lowdim/evidence/by-task-and-metric.csv) |
| [Figure 1. Selection and end-to-end time saved on 56 low-dimensional configurations](figures/lowdim-runtime.pdf) | [Generator](plot_runtime.py) · [Paired timings](../../research/lowdim/evidence/all-cases.csv) |
| [Figure 2. Fixed-width comparisons and exhaustive Abalone subset scores](figures/subset-quality.pdf) | [Generator](plot_subsets.py) · [31 ablation configurations](../../research/lowdim/peer_evidence/cases.csv) · [45 subset scores per split](../../research/lowdim/peer_evidence/abalone-subset-scores.csv) |
| [Figure 3. Single-peer rule, F and random selection against the same full model](figures/matched-curves.pdf) | [Generator](plot_matched_curves.py) · [56 low-dimensional configurations with matched widths and two-process timings](../../research/lowdim/evidence/all-cases.csv) |
| Peer-layer ablation and 32 random subsets per configuration | [Selected-set and score comparisons for 31 configurations](../../research/lowdim/peer_evidence/cases.csv) · [Exhaustive comparison of the 45 eight-column Abalone subsets](../../research/lowdim/peer_evidence/abalone-all-subsets.csv) |
| Low-dimensional implementation and wide-matrix rule | [Selector entry point](../../research/lowdim/method.py) · [Cross-products and conditional ridge gains](../../research/lowdim/native_votes.c) · [Wide-matrix voting](../../research/conditional_dense/shared_method.py) · [Archive verification](../../research/lowdim/evidence/verification.json) |
| Background reading | [Notes on the two reference papers](../../research/theory/README.md) |

The method labels describe the implementations measured in each run. In the archived results, `new` uses the single-peer rule up to 32 columns and the nested-peer rule above 32. The internal comparator `previous` uses the nested-peer rule at every width. The fixed-width ablation compares the single-peer rule with ranking by the gain of each feature in isolation.

For Table 2, first take the median end-to-end runtime of the repeated measurements within each process, separately for each method and configuration. Average the two process medians to obtain one runtime per configuration, then take the median across configurations for each dataset. Test scores in the table are arithmetic means across configurations within the same dataset. A configuration counts as faster only if the evaluated selector takes less time in **both** within-process comparisons. Score differences in Table 3 and in the score-tolerance analysis are the evaluated selector's score minus the comparator's score.

The random-label simulation has the literal dataset name `null`. In pandas, read the CSV files with `keep_default_na=False` to preserve that name.

## Reproduce the figures

From the `history` directory, run the following commands to regenerate the vector figures and their PNG previews from the recorded measurements. Figure 2 uses one copy of each Abalone split. Its 135 subset scores were extracted from the original per-seed `exhaustive_scores` arrays in `results/minpi-study/lowdim-peer-ablation-20261118a/`.

```powershell
.\thesis-python.ps1 paper/minpi-lowdim/plot_runtime.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_subsets.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_matched_curves.py
```

## Build the PDF

Run XeLaTeX twice from this directory to resolve the references. The bibliography is included in the source.

```powershell
xelatex -halt-on-error main.tex
xelatex -halt-on-error main.tex
```
