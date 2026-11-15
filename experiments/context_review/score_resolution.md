# Does an unmasked probability vector identify the outer row?

On the historical 14-task panel we fixed the ordered outer labels, the empirical multiset of complete outer feature rows and both refitted models. Within each fold we reassigned feature rows only between positions with identical observed codes. A witness is a complete reassignment that reverses the observed decision on the joint 3% AUC/accuracy guard. The candidate and anchor predictions move together. A certified invariant has the same decision on every allowed reassignment. A residual unresolved case can have either status.

The code in the first observer consists of the initial ensemble's unmasked probabilities alone. The second observer supplies the same code alongside two unrounded, outer-row masked predictions from that ensemble, one using the candidate support and one using the anchor, both with an empty extension. The point pair alone permits 17 witnesses on 704 proposals. These outer predictions are additional information beyond the voting procedure's development observations.

## Equal-occupancy control

We quantized each class coordinate at 2, 4 or 8 bits by taking the floor of its probability times the number of bins and assigning probability 1 to the last bin. Full precision uses the stored model output unchanged. We reconstructed the initial model from fitting rows and matched all row-equality partitions in the original archive. Each of five control draws randomly assigns score vectors to distinct raw feature-vector types of the same within-fold multiplicity. Identical observed features always receive one code. The same mapping is used at all resolutions within a draw. This preserves the exact fold-specific frequency of every code. The reassignment uses no outcomes or refit predictions. It inherits the trained model's code frequencies. Of 37,760 observed input types across folds, 37,749 belong to a multiplicity stratum with another type.

| Bits per coordinate | Signatures | Singleton rows | Code alone, genuine W / U | Code alone, control W / U | With point pair, genuine W | With point pair, control W |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 1,065 | 361 | 524 / 11 | 698–701 / 0 | 12 | 6–9 |
| 4 | 4,281 | 2,760 | 338 / 9 | 648–656 / 0–1 | 9 | 3–8 |
| 8 | 13,392 | 11,109 | 65 / 1 | 454–482 / 0–6 | 3 | 1–4 |
| Full | 36,553 | 35,619 | 0 / 0 | 9–16 / 1–2 | 0 | 0 |

W and U count witnessed and unresolved decisions. For each draw, 704 minus these two counts gives its certified invariants. The controls show the minimum and maximum over all five predeclared draws, calculated separately for each column. Every point-plus-code case was resolved. Signature and singleton counts sum over 42 separate folds with 38,021 outer rows. Both codes have identical occupancy by construction. At 8 bits, the genuine bare code permits reversals on Ionosphere (12), Phoneme (12), Sonar (17), SPECTF (20), Vehicle (3) and Wine quality red (1); the unresolved proposal is on Ionosphere.

Coarse genuine codes alone constrain the reassignments more than the occupancy-matched controls for this guard. At full precision, even a random reassignment certifies 686–694 decisions per draw. Supplying the point pair changes that comparison. At 2 and 4 bits, more witnesses survive with the genuine code than with any of the controls. The code's overlap with the masked-pair partition matters. The full-precision results condition on a known, finite feature dictionary and frozen refits. They do not establish that a score forecasts a new refit's outcome or that possible permutations have equal probability.

## Replay

From the thesis checkout, with the two archived input bundles available, run

```powershell
.\thesis-python.ps1 -m experiments.context_review.transcript_information
.\thesis-python.ps1 -m experiments.context_review.score_resolution_replay
.\thesis-python.ps1 -m experiments.context_review.score_resolution_check
```

With the original PMLB features and fold selections present, `experiments.context_review.score_resolution` reconstructs `unmasked.zip`. It stores probability coordinates and anonymous input-type indices for each fold. [Protocol](../../configs/context_review/score_resolution.json) · [Inputs and output tables](../../thesis-paper/evidence/score_resolution) · [Original observer experiment](transcript_observers.md)

For witnesses, the optimizer constructs an admissible permutation and verifies the decision on both refits with the metric library. For invariant decisions it uses homogeneous-label cells, inherited invariance under a finer observation or unrestricted coordinatewise assignment extrema. A restricted search can only supply a witness. A joint pass may require both metrics to improve simultaneously. Separate coordinate suprema can therefore leave a failure unresolved. The replay independently tests the metric formula on 120 permutations of a small tied multiclass example. A second check tests nesting across all 33,792 conditions and exhaustively enumerates 76 distinct label allocations in 16 small nontrivial observed orbits (genuine and random codes, alone and with the point pair, for both outcomes). The 16 examples verify computation and do not estimate frequencies.
