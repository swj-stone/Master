# Exchange restrictions in minimum-width teacher compression

We tested whether fixed-width exchange preferences can accelerate the search for the smallest support meeting a calibration error budget. The target was the frozen linear additive teacher from the preceding quota study, with relative score-norm tolerance .20 on the same 14 previously used sources at seed 42. This study examines optimization work on those sources. No fresh outcomes enter it.

For a retained set $S$, write the fidelity utility as $F(S)=\sum_{i\in S}d_i-2\sum_{i<j\in S}G_{ij}$, where $d_i=2\sum_jG_{ij}-G_{ii}$. For candidates $i,j$ with background $T$, the gain on substituting $i$ for $j$ is $d_i-d_j-2\sum_{t\in T}(G_{it}-G_{jt})$. Given valid cardinality bounds $L,K$, form the sorted differences between rows $i$ and $j$ after excluding the two endpoints. The adversary takes the largest $t$ values, where $t$ is the number of positive differences clipped to the allowed background cardinality interval. This gives the exact minimum exchange gain on that interval. A strictly positive minimum, exceeding the numerical guard, enforces $x_j\leq x_i$ on fidelity maximizers at every allowed width.

Take a minimum-width feasible support and maximize fidelity at its width. It satisfies all certified preferences: a violating swap would improve fidelity and preserve cardinality. This proves that at least one sparsest feasible support obeys the order. In exhaustive tests, 296 otherwise feasible nonideal supports were excluded. If an item has at least $K$ strict predecessors it cannot be selected in an ideal of width at most $K$. If it has at least $p-L$ successors it belongs to every ideal of width at least $L$. Propagation combines these implications with necessary signed projection ballots; only cover relations are sent to the solver. The unrestricted-width comparator uses all background cardinalities. A second variant introduces one binary indicator per possible cardinality and activates the edges and positional fixes certified at that size. Its control receives the same indicator encoding.

All arms started with the same best feasible solution from backward, forward, and energy-initialized swap search; fixed the same three-second wall-clock upper limit; used one CPU core; and received the same initial directional-ballot width lower bound. One backend imposes the quadratic fidelity constraint in SCIP. The other iteratively adds separating linear ballots. Timing includes construction of restrictions, propagation and solving. The 14-source runs used one seeded randomized arm order per task and one clock per arm. Timeout lower bounds can vary on replay.

| Backend and restriction | Certified / 14 | Sum of incumbent widths | Sum of proven lower bounds |
|---|---:|---:|---:|
| Native quadratic, no order | 10 | 306 | 260 |
| Native quadratic, unrestricted order | 10 | 306 | 260 |
| Native quadratic, interval order | 10 | 306 | 260 |
| Ballot master, no order | 9 | 306 | 209 |
| Ballot master, unrestricted order | 9 | 306 | 209 |
| Ballot master, interval order | 9 | 306 | 208 |
| Native quadratic, width indicators without/with local orders | 10 / 10 | 306 / 306 | 258 / 258 |
| Ballot master, width indicators without/with local orders | 9 / 9 | 306 / 306 | 204 / 203 |

All ten arms retained the same columns as the strong local incumbent on each source. Spambase had 660 interval-order pairs and 499 unrestricted pairs. The solver's fixed count rose from 6 to 13, but all three native quadratic arms still proved width 23. Madelon, Gisette and Isolet had zero preferences on their full admissible width intervals. Conditioning separately on each width exposed 141 possible pairs on Madelon and 3,847 on Gisette, largely at widths far below the incumbent. Isolet had none even after conditioning. The extra constraints did not close another proof.

On Credit Card Fraud the 305 unrestricted and interval preferences coincided. A post hoc check with seven rotated timing repeats found median ballot-master times of .1078 s without order, .0157 s with unrestricted order and .0153 s with interval order. Charging the measured common teacher, Gram and upper-bound cost gave selection times of .8087 s, .7166 s and .7162 s. Native quadratic solver medians were .0179 s, .0089 s and .0085 s. All returned the same 12 columns. This source supports a local pruning benefit, with no measurable general advantage from the interval or width-conditioned variant.

The brute-force suite evaluated 20,832 pairwise certificates over 588 intervals, including signed Gram entries, tied and null games, and empty/full cardinalities. It independently enumerated each permitted background and fixed-width fidelity maximizer. The two backends and five restriction configurations reached the exhaustive optimum in all 70 tiny solver runs.

```powershell
.\thesis-python.ps1 -m experiments.minpi_rework.quota_order_checks
.\thesis-python.ps1 -m experiments.minpi_rework.quota_order_study --run my-quota-order --seconds 3
.\thesis-python.ps1 -m experiments.minpi_rework.quota_order_study --run my-quota-indicators --seconds 3 --rules hot_none hot_range
.\thesis-python.ps1 -m experiments.minpi_rework.quota_order_study --run my-quota-clocks --names credit_card_fraud --repeats 7
```
