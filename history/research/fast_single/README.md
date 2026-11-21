# Single-peer selection with sampled peers

This experiment tests whether a cheaper estimate of the paper's single-peer score can close the end-to-end time gap to a uniform random subset. The reference implementation is `research.peer_crossover.single.select_single`. The candidate uses training inputs and a seed to determine its support. The test scores and deployment timings are used only for evaluation.

## Results

The historical panel contains 405 paired configurations. The candidate removed columns in 403 and ran faster than the full-feature model in 320 of these. Adding a maximum score loss of 0.02 left 215 successes, compared with 163 for the paper rule, 131 for F and 152 for uniform random at the candidate width. Random was faster at the same width in aggregate. The median paired candidate/random total-time ratio was 1.113, with the candidate faster in both processes in 16 configurations and within 10% in 177. Its test score against random was higher 286 times, equal 10 times and lower 109 times. Requiring all three conditions against Full and a time within 10% of random leaves 83 configurations for the candidate and 35 for the paper rule. [All 405 cases](evidence-extension/all-405.csv)

Across the 331 historical configurations with more than 32 columns, the median retained fraction fell from 46.1% to 37.5%. Selection, fitting and prediction took less time than the exact rule in both timing processes on 319 configurations. Using the full-feature score as the reference, 190 configurations both sped up and lost at most 0.02 while dropping columns, compared with 134 for the exact rule. In a direct score comparison with that rule, the candidate won 122 times, tied 9 times and lost 200 times. On the 115 wide regression configurations the median loss in test R² relative to Full was 0.0076, and the 90th percentile of the loss was 0.5134. [Paper panel and results by metric](evidence/summary.md) · [Historical 405-configuration comparison](evidence-extension/summary.md)

At the candidate's own width, its total time was within 10% of uniform random in 116 of the 331 wide configurations, and lower in both processes on eight. Its test score exceeded the same random subset in 233 cases, tied in six and fell below it in 92. The largest time savings over the paper occurred above 2,048 columns. Median selection time across these 43 configurations fell from 119.56 ms to 2.16 ms, and median total time from 127.03 ms to 10.94 ms. The median paired total-time ratio to same-width random fell from 7.96 to 1.65. In every one of the 38 sparse text and advertising configurations, the selected training submatrix had more nonzeros than the same-width random submatrix, with a median ratio of 3.06. [Per-configuration measurements](evidence/paired-cases.csv) · [Ablation and sparse-matrix counts](evidence/ablation.md)

Holding width and sampled rows fixed, adding peer votes to the marginal ranking improved 81 scores and reduced 53, with 197 ties. It changed the selected indices in 152 configurations. Holding the sampled peer ranking fixed and changing only the width weights reduced median retention from 42.2% to 37.5%, with 105 score gains, 33 ties and 193 losses. These two comparisons distinguish the effect of the peers from that of keeping fewer columns. [Ablation details by dataset](evidence/ablation-groups.csv)

### Four additional datasets

The table gives medians over three seeds. Positive score loss means Full scored higher. Time ratios compare the candidate with its paired reference, including selection. The last column counts seeds with fewer columns, shorter total time than Full in both processes and score loss at most 0.02. Full and random have separate time columns because beating Full can leave a substantial gap to the same-width random subset. [Per-seed scores and timings for all methods](evidence/transfer.csv)

| Data | Metric | Input columns | Kept | Score loss vs Full | Total time / Full | Total time / same-width random | Joint successes |
|---|---|---:|---:|---:|---:|---:|---:|
| Gasoline | R² | 401 | 134 | −0.0012 | 1.001 | 1.13 | 1/3 |
| Yarn | R² | 268 | 180 | −0.0001 | 1.156 | 1.10 | 0/3 |
| Olive | R² | 612 | 163 | 0.0104 | 0.817 | 1.18 | 3/3 |
| Real-sim | Balanced accuracy | 20,958 | 1,536 | 0.0595 | 0.177 | 2.16 | 0/3 |

## Candidate rule

At up to 32 features the candidate delegates to the exact paper rule. On wider inputs it uses at most 384 training rows, sampled by position as in the paper, and draws a shared panel of eight feature indices uniformly without replacement. Every feature remains eligible for selection. For feature $j$, the candidate evaluates its ridge gain alone and its gain conditional on each panel member other than $j$. The smaller of its marginal gain and the mean of these conditional gains is its ranking score $s_j$. Sampling the peers replaces a dense $p$-by-$p$ product by a $p$-by-8 product. A peer on which a particular feature depends can be missed. The conditional mean estimates the full peer mean on these sampled rows. Taking the minimum need not preserve unbiasedness.

The paper chooses the width from the participation ratio of the scores. In the candidate this ratio uses weights $w_j=s_j^2/(s_j+b)$, where $b=m\lVert Y_c\rVert_F^2/[(m-1)(m+\sqrt m)]$ for $m>1$ and $Y_c$ is the centered sampled response. For a fixed nonconstant standardized column, $b$ is its expected empty-set ridge gain under a uniform permutation of the response rows. This scale is available from the sampled response. No permutations or additional fits are needed. The weights discount small gains when calculating $K=\lceil(\sum_j w_j)^2/\sum_j w_j^2\rceil$. The selected columns are still the top $K$ according to $s_j$, with ties broken by column index. A zero-information fallback retaining the whole input counts as a failure to reduce width.

## Paired comparison

The main benchmark uses the 387 configurations in the current all-width Single-peer paper. Each selector is compared with full-feature ridge and with uniform random and univariate F selectors using its own width. The random selector receives the matching width as an input. Its timed region includes drawing its own subset, while each adaptive method pays to discover its width. Width-matched controls with identical seeds and widths are run once per repetition and share the observation. This prevents two labels for the same control from acquiring different timings.

Each process warms each distinct arm once, then times three interleaved runs. A second process reverses the run order. Timing covers selection, downstream fitting and test prediction, including the same in-pipeline data handling as the paper. For the comparison with random, take the median time of each arm in each process, form the candidate/random ratio within each process, and retain the larger ratio. A ratio below one means the candidate was faster in both processes. Ratios up to 1.05, 1.10 and 1.20 are also reported. A three-way success requires fewer than all input columns, less total time than Full in both processes, and a specified maximum paired test-score loss from Full. A loss of 0.02 is a reporting threshold. We also report continuous score differences and results by metric. These are paired observations across overlapping tasks, splits and seeds, so counts do not represent independent datasets.

## Reproduction

Run from `MastersThesis/history` using the existing thesis environment. Run the timing processes sequentially on an otherwise idle host. The runner checks archived input, support and prediction fingerprints and verifies each timed prediction against the corresponding warm-up.

```powershell
.\thesis-python.ps1 -m research.fast_single.runner results/minpi-study/fast-single-20261121b --process 1
.\thesis-python.ps1 -m research.fast_single.runner results/minpi-study/fast-single-20261121b --process 2
.\thesis-python.ps1 -m research.fast_single.transfer results/minpi-study/fast-single-transfer-20261121b --process 1
.\thesis-python.ps1 -m research.fast_single.transfer results/minpi-study/fast-single-transfer-20261121b --process 2
.\thesis-python.ps1 -m research.fast_single.transfer results/minpi-study/fast-single-transfer-20261121b --summarize
.\thesis-python.ps1 -m research.fast_single.summarize results/minpi-study/fast-single-20261121b --out research/fast_single/evidence
.\thesis-python.ps1 -m research.fast_single.ablation research/fast_single/evidence/paired-cases.csv
.\thesis-python.ps1 -m research.fast_single.extension results/minpi-study/fast-single-extension-20261121b --process 1
.\thesis-python.ps1 -m research.fast_single.extension results/minpi-study/fast-single-extension-20261121b --process 2
.\thesis-python.ps1 -m research.fast_single.extension results/minpi-study/fast-single-extension-20261121b --summarize
```

The historical 405-configuration panel consists of the 387 paper configurations plus 18 additional low-dimensional configurations from six UCI datasets. The `extension` command verifies their archived inputs and reports the combined panel. The four supplementary datasets are Gasoline, Yarn, Olive and Real-sim. Each uses three seeds. Only Gasoline gets a new test split per seed. These data were available to the project before this comparison. Their scores are reported separately from the historical panel. The score-only ablation on the original wide configurations holds the candidate width fixed when removing the peer vote, and holds its ranking fixed when removing the width adjustment. Those additional fits are outside the deployment timings.
