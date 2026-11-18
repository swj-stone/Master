# Formal and numerical checks

From this folder run `lean Pivotal.lean` with Lean 4.35.0. The file imports only `Std`. Its executable output lists each theorem's axioms.

| Paper statement | Lean result |
|---|---|
| A row numerator on the common scale has magnitude at most the common denominator. If the row count times that denominator fits the symmetric signed 64-bit range, every subsum fits as well, including separate sign accumulations. The final score can also be negated for sorting. | `MinPI.scaledRowBound`, `MinPI.nativePartialSum` |
| A pointwise one-sided error bound passes to the h-th largest score, counting ties. Given the per-column enclosures and the three actual order statistics, every unresolved column lies within twice the grid width of the exact budget cutoff. | `MinPI.rankCutBound`, `MinPI.unresolvedAtRank` |
| Binary one-peer signed increments sum to half of the peer count times the focal vote minus half of the peer-vote sum. | `MinPI.binaryPair`, `MinPI.binaryOnePeer` |
| Comparing a proposed member's lower endpoint with an outsider's upper endpoint in score/index order guarantees the same ordering of the true scores. | `MinPI.indexedSeparation` |
| Number of indexed subsets with each specified profile of positive, negative and abstaining peers | `MinPI.profileMultiplicity` |
| Number of indexed subsets at size k | `MinPI.sizeMultiplicity` |
| Marginal increment for positive, negative and abstaining votes at every integer margin | `MinPI.positivePivot`, `MinPI.negativePivot`, `MinPI.zeroPivot` |
| Preservation of every entry of a descending top-h prefix, including its tie order, after pruning below a valid threshold | `MinPI.screenedPrefix` |
| Enclosure of one signed integer divided by a positive denominator, followed by enclosure of their sum in a layer | `MinPI.gridTerm`, `MinPI.gridLayer` |
| Equivalence between reversal of two anchor scores and the difference of their size penalties in common integer units | `MinPI.penaltyFlip` |

`profileMultiplicity` and `sizeMultiplicity` count multiplicities in a recursively enumerated list. Each position of the peer list is included or omitted exactly once along a branch. Peers with equal votes count as different indices. Binomial coefficients are defined by Pascal's recurrence. The ratio of the two counts is the probability under a uniform indexed subset. The manuscript proves the subsequent coupling and concentration statements in ordinary mathematics.

Run `Get["verify.wl"]` in Wolfram Language from this folder. It returns `{798, 0, 0}`, reporting the number of cases, disagreements between exact profile counting and indexed enumeration, and violations of the finite-population coupling bound. In the same session run `verifyMargins[]` to enumerate the complete peer-margin distribution at each feasible size for zero through five peers. It returns `{1358, 0}` (counts tested and disagreements). No external packages are required for either check.
