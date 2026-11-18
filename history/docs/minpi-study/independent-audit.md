# Independent computational audit

The checks here concern the finite-game implementation and the evidence for specific manuscript claims. The editorial decision is recorded separately in [current-review.md](current-review.md).

## Exact targets

From the study root, run the [independent oracle](../../experiments/audit_minpi_study_exact_targets.py).

```powershell
.\thesis-python.ps1 -m experiments.audit_minpi_study_exact_targets
```

The run passed 30,910 support comparisons. It enumerates every indexed peer subset on all 729 two-by-three ternary matrices, for all seven nonempty peer-count sets and all three budgets. Further direct enumeration covers widths 4, 7 and 9, including odd and full peer counts. Both exact public routines receive unsigned NumPy budgets, which exercises the previously failing cutoff case. A separate check exercised 109 native-word capacity boundaries at the largest safe row count and one row above it.

For the gas ballot at seed 1129 the oracle counts the number of positive peers directly, constructs each column's four rational layer means and sorts the resulting minima with the original index tie rule. At budget 64 and peer counts 0, 1, 4 and 16, zero-based columns 45 and 115 tied at −525/32512. The exact support took 45. Both floating routines took 115 on the study environment.

## Theoretical and empirical cross-checks

- The integer-composition calculation was reconstructed on the original training ballots. The objective in every applicable row of all 201 [layer/panel records](../../results/minpi-study/order-diagnostics/layers.csv) agreed in rational arithmetic. At each reported scale, the left and right derivatives of the convex objective bracketed zero, for the original binary bound or its range extension as applicable. Exact support agreement, adjacent gaps and twice-width cutoff counts agreed on all 47 study panels, including the five artificial benchmark splits.
- Recalculation from the acquisition ballots and labelled fitting rows reproduced the same- and cross-instrument absolute pivots, class-centroid covariance and the five participant-level worst-location scores. Separate recomputation from the five raw timing observations per method and panel gave median dyadic-to-object-anchor, dyadic-to-exhaustive-object and dyadic-to-floating certificate ratios of 0.620077, 0.587757 and 1.410410 on the 42 measured panels. The more efficient native-word comparison is recorded in [word-design](../../results/minpi-study/word-design/).
- A fresh [training-witness experiment](witness-design-protocol.md) after the unsigned-budget fix reproduced all 23 support arrays and the complete case table byte for byte. At the three defaults, the generated votes also matched the production arrays and the previously archived training-ballot digests.
- Lean 4.35 compiled the [formal development](../../paper/minpi/formal/Pivotal.lean) using its standard axioms. It covers scaled products, arbitrary partial sums in the symmetric signed 64-bit range, propagation of bounds to a tied order statistic, and the resulting twice-width unresolved band. The [Wolfram Language enumeration](../../paper/minpi/formal/verify.wl) returned {798, 0, 0} for its finite games and the coupling inequality.

