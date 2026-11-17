# Input columns referenced by a restricted forest

In a fixed forest that stops a tree at its first unavailable split, a supplied feature is useful to the traversal only if some reachable split reads it. For each structural split let H contain the distinct features on the path from its tree root through that split, inclusive. For supplied columns S define C(S) as the union of all H contained in S. Include every tree node, including branches absent from calibration data and paths with zero fitted score changes. The rule is data independent once the forest is fixed. A tree may have a redundant split or an unsatisfiable path for a particular input domain, so the structural count can still exceed the smallest equivalent predictor's width.

For any additional availability mask M, a structural path fits inside S ∩ M exactly when it fits inside C(S) ∩ M. The tree therefore makes the same splits on every row under either restriction. This gives C(S ∩ M) = C(C(S) ∩ M) and equal predictions under all M. The operator is monotone, idempotent and contracts every support. Its fixed sets are closed under unions. These properties apply to the deployed forest with fixed split nodes and node outputs. A new fit on the two supports may follow entirely different splits. For instance, take six rows with x₀ = y = (0,0,0,1,1,1) and x₁ = (0,0,1,0,1,1). A least-squares stump on both selects x₀. Restrict it to {x₁} and its structural support is empty. A stump refitted on x₁ predicts 1/3 or 2/3, whereas a predictor fitted to the intercept predicts 1/2.

The 20 wide-source jobs were registered for this secondary analysis after observing their structural widths and before opening their outer scores. The other 35 jobs replay seven sources with previously opened outer scores. The registration is `504dd56` and the 55 forecast receipts were sealed at `f95aedf` before wide or companion-refit scores were opened. Every S versus C(S) prediction array on the acceptance and outer inputs was bitwise identical. The collector also checked each physical path for equal availability under the two supports and compared the numerical acceptance risk curves for all coalition sizes. [Registration](../../../../experiments/minpi_rework/predictive_reachable_protocol.md) · [seals](lock.json) · [archive digest](archive-sha256.txt)

## Frozen path comparisons

All paths use the same prespecified clean and all-size acceptance criteria within each source and seed. For a fixed deletion order with feature ranks r, the last active split using j disappears at the deletion depth τⱼ = max(H containing j) min(h in H) rₕ, with τⱼ = 0 for unused j. At depth d the referenced columns are exactly those with τⱼ > d. This evaluates the physical width over the entire path without repeatedly traversing the trees. Since these sets are nested, any two tied minimum-width feasible depths along that path have the same structural support. Thus recompeting depths within one path can never change its predictions. Paths may have different minimum-width feasible supports, so we checked whether recompeting paths changes the selected forecast.

The table gives the mean number of structural columns over five seeds. Each portfolio considers MDI plus the named second path. Earlier sources also offer pooled, intact and static-ablation controls. Wide sources registered only the two MDI and MinPI paths.

| Panel | Source | Full | MDI | MinPI | MDI + MinPI | MDI + pooled | MDI + static |
|---|---|---:|---:|---:|---:|---:|---:|
| Earlier | Beijing | 12.0 | 8.2 | 6.8 | 6.8 | 6.4 | 8.0 |
| Earlier | Churn | 10.0 | 8.4 | 8.4 | 8.4 | 8.4 | 8.4 |
| Earlier | German | 12.0 | 7.6 | 8.2 | 7.6 | 7.4 | 7.4 |
| Earlier | Liver | 10.0 | 6.6 | 6.2 | 5.0 | 5.2 | 6.4 |
| Earlier | Seoul | 10.4 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| Earlier | SGEMM | 13.6 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 |
| Earlier | Votes | 16.0 | 11.4 | 11.8 | 11.4 | 11.2 | 11.4 |
| Wide | Internet Advertisements | 264.6 | 149.6 | 127.6 | 127.6 | — | — |
| Wide | QSAR oral toxicity | 537.8 | 256.0 | 214.0 | 213.4 | — | — |
| Wide | Tehran Residential Building | 71.2 | 30.4 | 40.4 | 29.0 | — | — |
| Wide | UJIIndoorLoc | 132.4 | 18.6 | 26.8 | 18.2 | — | — |

The wide MDI + MinPI portfolio used fewer reachable inputs than canonicalized MDI in 11 of 20 paired jobs and tied in nine. Mean reductions by source were 22.0, 42.6, 1.4 and 0.4 columns. The equal-source mean was 16.6 fewer columns (whole-source bootstrap interval 0.9–32.3), equivalent to 1.86 percentage points of supplied columns (0.50–3.47). The portfolio contains MDI, so it cannot be wider under the shared criterion. The earlier seven-source mean was 0.43 fewer columns (0–0.91), concentrated in seven jobs across Beijing and Liver. Pooled + MDI was 0.51 columns narrower than MDI on the same earlier panel (0.06–1.06). The minima did not yield the narrowest mean there. Intervals resample source means, with 20,000 bootstrap draws. Each panel is purposively selected, and each metric-specific wide estimate uses only two source means.

For the original wide MDI + MinPI portfolio the mean nominal width was 98.35, and structural pruning reduced it to 97.05 across the 20 jobs. For the corresponding earlier portfolio both means were 6.77. Reoptimizing physical width introduced **zero further width or prediction changes** in every portfolio tracked on either panel. With the original wide MDI-first tie breaker substituted for the registered minimum-acceptance-risk tie breaker, all 20 wide winners also stayed the same. In the earlier intact + MDI comparison 13 portfolio path names changed, all with identical outer predictions and width. Structural pruning explains which inputs a frozen artifact references and corrects comparisons with a full forest that already uses fewer than p distinct features. These outcomes provide no additional selector improvement attributable to reoptimizing canonical width.

## Outer prediction

On the wide sources, MDI + MinPI minus MDI averaged −.00334 AUC and +.00007 accuracy over two classification sources, and +.00079 R² and +.00014 scaled MAE over two regression sources. Compared with the full forest, the two regression source means both exceeded the accepted .02 R² loss on outer rows. The seven reopened sources averaged −.00104 AUC, −.00087 accuracy, −.00088 R² and +.00216 scaled MAE versus MDI. These are descriptive paired differences. All original versus reoptimized MinPI portfolio outer metrics coincide because their predictions coincide. [Wide source-by-source prediction metrics](../wide-veto/README.md) · [all paired jobs](scores.csv) · [all source means](source-means.csv) · [paired source-bootstrap contrasts](contrasts.csv)
