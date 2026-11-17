# Replaying the matched chain experiment

The eight UCI sources and six historical wide sources are previously observed benchmarks. This is the diagnostic follow-up described in `chain_search.md`. Every method used the original width-validation rule. The additional equal-width diagnostic uses half the available features, rounded up. That width is exploratory. Test metrics at the fixed width were computed after timing. The uniform comparator receives the query budget observed on the chain run for that input. Its clock excludes the separate computation needed to supply that diagnostic budget.

```powershell
.\thesis-python.ps1 -m experiments.minpi_compression.chain_replay
.\thesis-python.ps1 -m experiments.minpi_compression.chain_uniform_expectation
.\thesis-python.ps1 -m pytest tests/test_minpi_chain_replay.py -q
```

The first command measures all four selectors at each of four clock positions, verifies every negative witness by two independent SVD ridge fits and writes the 56-split results to `thesis-paper/evidence/minpi_compression/chain-replay/`. The original within-fit/vote feature rankings and per-fold budgets are preserved by deterministic seeds. The outer labels of the already published wider study are matched to its stored test-row IDs. The runs file records supports, forecast and rank hashes, width diagnostics, metrics and timings. The bounds file records every feature label and witness. Per-source and per-domain tables contain paired chain-minus-baseline metric differences and geometric chain/baseline cost and width ratios.

The second command evaluates all candidate prefixes offline to compute the conditional expected number of negatives found by a uniform sample of the matched size. It does not alter the measured selectors. For each feature, its downward edges are disjoint pairs of coalitions. An edge with two known endpoints is already detected. Among the other edges, let `s` have one unknown endpoint and `d` have two. Among `N` candidate queries, the remaining `N − s − 2d` are unrestricted. For a draw of `q` queries, the exact miss probability is the coefficient of degree `q` in `(1+2x)^d (1+x)^(N−s−2d)`, divided by the binomial coefficient `choose(N,q)`. Summing detection probabilities gives the expected count by linearity. It is a conditional benchmark on these realized anchors, interiors and votes, not a generalization estimate across sources.
