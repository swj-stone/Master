# Feature selection with sampled single-peer ridge gains

[English manuscript (PDF)](main.pdf) · [LaTeX source](main.tex)

The main experiment covers 405 configurations, including 74 with at most 32 input features and 331 with more. The training data, test data, seed, model and metric are paired across methods within each configuration. The implementation uses all individual peers at up to 32 features and a sampled peer panel above that threshold. [Selector](../../research/fast_single/selector.py) · [Ridge gain kernel](../../research/fast_single/panel_votes.c) · [Experiment protocol and replay commands](../../research/fast_single/README.md)

| Manuscript content | Measurements and sources |
|---|---|
| Benchmark composition and data splits | [405 configurations with input fingerprints](../../research/fast_single/evidence-extension/all-405.csv) · [Original 387 configurations](../../research/lowdim/comparison-protocol.md) · [18 additional low-dimensional configurations](../../research/fast_single/extension.py) |
| Selected widths, test scores, paired timings and direct baseline comparisons | [405-configuration results](../../research/fast_single/evidence-extension/all-405.csv) · [Counts by group and method](../../research/fast_single/evidence-extension/by-source.csv) |
| [Figure 1. Time saved against F and uniform random at matched width](figures/lowdim-runtime.pdf) | [Generator](plot_runtime.py) · [Primary benchmark](../../research/fast_single/evidence-extension/all-405.csv), restricted to the 74 low-dimensional configurations |
| [Figure 2. Fixed-width comparisons and all eight-column subsets of Abalone](figures/subset-quality.pdf) | [Generator](plot_subsets.py) · [31 low-dimensional configurations at matched width](../../research/lowdim/peer_evidence/cases.csv) · [45 subset scores per Abalone split](../../research/lowdim/peer_evidence/abalone-subset-scores.csv) |
| [Figure 3. Speedups across test-score loss thresholds](figures/matched-curves.pdf) | [Manuscript entry point](plot_matched_curves.py) · [Shared plotting code](../../research/fast_single/plot_figure3.py) · [405-configuration results](../../research/fast_single/evidence-extension/all-405.csv) |
| Ablation of sampled peer gains and the width adjustment | [Per-configuration observations on 331 wider inputs](../../research/fast_single/evidence/ablation-cases.csv) · [Aggregation](../../research/fast_single/evidence/ablation.md) |
| Four additional datasets | [All methods and seeds](../../research/fast_single/evidence/transfer.csv) · [Data loading and split definitions](../../research/fast_single/transfer.py) |

The `fast` columns in the 405-configuration CSV contain the sampled Single-peer results. The `single` columns contain measurements for the [exhaustive comparator](../../research/peer_crossover/single.py). The `F_fast` and `random_fast` columns use the width chosen by the sampled method. The corresponding controls at the exhaustive method's width are `F_single` and `random_single`. The `full` columns describe the full-feature predictor. Read the CSV with `keep_default_na=False` in pandas to preserve the literal dataset name `null`.

Each of two processes recorded three timed repetitions after a warm-up. The recorded time runs from feature selection through fitting and test prediction. For an absolute time in milliseconds, average the two within-process medians for each configuration before summarizing across configurations. To count a speedup, compare the two methods within each process and require a lower median time in both. Figure 1 plots the smaller percentage saving from these two comparisons. Figure 3 plots the proportion of all configurations in each width group satisfying the time and score conditions. Its lower panels also require fewer than all input features.

Figure 2 uses a separate experiment on the same low-dimensional branch. The 31 selected supports, widths and primary scores match their entries in the 405-configuration table. The original experiment additionally scored 32 random subsets per configuration. On each of three Abalone splits it evaluated all 45 eight-column subsets. [Ablation records](../../research/lowdim/peer_evidence/README.md)

## Reproduce the figures

From `MastersThesis/history`, run the commands below. Each writes the PDF used in the manuscript and a PNG preview.

```powershell
.\thesis-python.ps1 paper/minpi-lowdim/plot_runtime.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_subsets.py
.\thesis-python.ps1 paper/minpi-lowdim/plot_matched_curves.py
```

## Build the PDF

Run XeLaTeX twice from `paper/minpi-lowdim` to resolve the cross-references. The bibliography is included in the source.

```powershell
xelatex -halt-on-error main.tex
xelatex -halt-on-error main.tex
```
