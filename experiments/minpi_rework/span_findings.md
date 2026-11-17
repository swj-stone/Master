# Coalition-size projections on the 65 locked deletion paths

Thirteen public sources contribute five splits each. Every screen uses the same gain tie order, exact incumbent, cache invalidation and calibration radius. We timed complete searches on the same matrix and incidence inputs, including constructing the bounds, with one CPU core, an initial JIT warm-up and three rotated measurement orders. All 1,365 measured calls reproduced the archived support and deletion sequence. The comparison isolates computation for a fixed selection rule. [Timing receipts](../../thesis-paper/evidence/minpi_rework/stratified-tree/span-timings.csv)

| Screen | Exact norm refreshes after initialization | Search time (ms) |
|---|---:|---:|
| Largest size only | 1,064 | 0.561 |
| Zero, middle, largest | 1,041 | 0.642 |
| Maximum over every size | 1,034 | 1.262 |
| Joint span of the three size vectors | 1,026 | 1.068 |
| Joint span of every size vector | 1,013 | 1.113 |
| Three sizes with the residual direction | 1,041 | 0.648 |
| All-size span with the residual direction | 1,013 | 1.099 |

All seven arms also compute 3,255 initial candidate norms. Refresh counts sum the 65 distinct searches once each. Times take the median of the three repetitions per split, average log times across seeds within source, and exponentiate the equal-source mean. Joint bounds use the squared norm of the projection onto an orthonormal basis. The residual variant also takes the maximum with the squared projection onto the current residual when its norm is positive. It reuses the cross term already present in the exact cost.

For feature j, partition its active-at-full-support increments by the original number of distinct required features. This gives a matrix B with at most D rows for maximum distinct-feature path depth D. A size-s mean marginal is w(s)B, where the qth weight for order q+1 is the ratio of falling factorials (s)_q/(p−1)_q. The literal screen evaluates all p directional projections from B times the candidate and the small Gram matrix BB′. When evaluating a nearly cancelling Gram quadratic, the code forms the direction explicitly to compute its norm. For the all-size joint bound we compute the thin SVD of B. For the three-size joint bound the input to the SVD has the three corresponding w(s)B rows. The numerical rank cutoff is the largest singular value times machine epsilon times the larger matrix dimension. Directions below this cutoff are omitted. New bounds are evaluated only when the cheaper three-size screen leaves a candidate in contention.

Among 4,905 distinct candidate/update states that required a new all-size evaluation, the first maximizing size was zero in 7, the prescribed middle in 8, another interior size in 3,790, and the largest in 1,100. Scanning sizes in ascending order resolves exact ties. In 3,779 states the all-size value exceeded the largest of the three designated votes by more than 64 machine epsilons times the larger magnitude. The all-size screen saved seven exact refreshes. The all-size span saved 28 at about 1.73 times the three-size search time. A tighter bound changes a decision only if it crosses the current incumbent or changes the queue in a way that avoids a later refresh. These counts concern candidate/update states that reached the secondary computation. They cannot estimate the distribution of strongest sizes over all candidate votes.

Run `.\thesis-python.ps1 -m experiments.minpi_rework.span_geometry` for direct projection checks and `.\thesis-python.ps1 -m experiments.minpi_rework.span_benchmark` to reproduce the clock and path comparison.
