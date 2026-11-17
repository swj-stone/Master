# Scalar ballot timing

A feature's marginal at coalition size $s$ is $h_{js}=\sum_{H\ni j}q_s(H)a_H$.
Let $I_H$ indicate whether group $H$ is still active. The current deletion
effect is $\delta_j=\sum_{H\ni j}I_Ha_H$. The triangle inequality gives

```text
R_js = sum over H containing j of |I_H − q_s(H)| × ||a_H||
max(0, ||h_js|| − R_js) ≤ ||δ_j|| ≤ ||h_js|| + R_js.
```

When a group dies, add $(2q_s(H)-1)\|a_H\|$ to each affected radius.
Intersect the three intervals. If the distance of the residual norm from
the intersection exceeds the current allowed error norm, skip that candidate.
Exact candidate evaluations accumulate the squared residual in the same row
order as the contiguous native direct baseline.

The dynamic control saves the exact norm of a trial and its deletion effect.
It bounds later changes by the summed norms of intervening groups. Removing
feature $w$ changes the trial error vector for $j$ only through deleted groups
that exclude $j$. The code uses the smaller of their summed norms and the norm
of the full deletion of $w$ plus the summed norms of shared groups.

All five arms reproduced every entry of 65 locked paths. Each split used
one warm-up and five timed repetitions in random order, pinned to logical
CPU 3. Each timing includes the method's scalar or dense initialization. Source
contrasts take the median time within each split and the mean log ratio over
seeds, then average over the 13 sources. The percentile intervals resample
13 sources with replacement 10,000 times. Times cover the selector.

Out of 96,417 candidate trials, three-size ballots sent 54,030 to the scan,
dynamic bounds sent 44,036, and their combination sent 44,035. The combined
selector took 2.05 times the running time of compiled block-32 scans
(source-bootstrap 95% interval 1.71 to 2.42). Dynamic bounds alone took 1.08
times that baseline (1.01 to 1.13). Block-32 scans can also stop within a
candidate, so the number entering a scan measures only part of the computation.

To reproduce the raw times and paired intervals from the thesis checkout, run

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.scalar_tree_benchmark
```
