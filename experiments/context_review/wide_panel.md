# External validation on six wide classification tasks

The protocol was fixed at Git commit `120e240`. Commit `9c762f7` contains the unrounded probability vectors, row identities, selected supports, every candidate's scores, and execution records for all 16 splits. Raw source downloads are local and can be recovered from their pinned UCI addresses and SHA-256 checksums in `thesis-paper/evidence/wide/sources.json`.

## Sources and split units

| Source | Independent evaluation unit | Rows × predictors used | Outer design and scope |
|---|---|---:|---|
| CNAE-9 | Business-description record | 1,080 × 856 | Three stratified folds for the released word counts and nine CNAE labels. Company identifiers were unavailable. [Ciarelli and Oliveira, 2009](https://doi.org/10.24432/C51G7P) |
| Ozone Level Detection | Dated Houston-area weather record | 2,534 × 72 | Three expanding-origin evaluations on the eight-hour ozone-day target. Every row of `eighthr.data` was included. The catalog's 2,536-row headline is the size of the separate one-hour release. [Zhang, Fan and Yuan, 2008](https://doi.org/10.24432/C5NG6W) |
| Parkinson's Disease Classification | Participant | 252 × 752 | The three sustained-vowel recordings for each participant were averaged within subject before three stratified folds. The identifier, sex and diagnosis were removed from the acoustic predictors. The cohort is from one clinic. [Sakar et al., 2018](https://doi.org/10.24432/C5MS4X) |
| SECOM | Time-stamped manufacturing entity | 1,567 × 590 | Three expanding-origin evaluations predict pass/fail yield. The numeric sensor file contains 590 columns. The source's time stamp and class are stored separately. [McCann and Johnston, 2008](https://doi.org/10.24432/C54305) |
| Semeion | Digit image | 1,593 × 256 | Three stratified image-level folds. About 80 people supplied the images. Writer IDs are unavailable, so a writer could contribute to both sides of a split. [Semeion Research Center and Tattile, 1998](https://doi.org/10.24432/C5SC8V) |
| Urban Land Cover | Image object | 675 × 147 | The released 168-object training file and 507-object test file are used once for the nine-class target. Both were sampled from the same aerial image. [B. Johnson, 2013](https://doi.org/10.24432/C53S48) |

All six sources are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Parkinson data came as the original CSV inside UCI's nested RAR/ZIP archive after the direct CSV endpoint proved unavailable. The I/O route was the only source-format adjustment. Every planned split contained every class. On the chronological sources, all equal timestamps belong to one partition and each partition ends before the next begins.

## Comparison

Gain, discrete voting, continuous slack, and the empty-restoration first-panel point estimate use exactly the same implementation as the ten-source transport test. They share the Gain anchor, deterministic candidate pool of at most eight sets of half anchor width rounded up, tolerance, fitted learner, and held-out refit guard. Independent runs of each arm include fitting, imputation, selection and final prediction. The separate candidate-resolution and metric audits are excluded from policy cost. The archived Gain order spans the columns used by at least one split in the initial tree ensemble.

The primary response is the removed fraction of anchor columns times the indicator that every required *outer* metric lies within its relative loss tolerance compared with the refitted anchor. A decline larger than 3% for accuracy or AUC is a breach. Binary average precision also has a 5% tolerance if the proposal-set minority is at most 5%. The historical binary encoding gives the fit-set minority the positive code. For the Parkinson cohort this makes controls positive. Interpreting AUC there requires the same label orientation in both candidate and reference.

The primary contrast is continuous slack minus point, averaging within source before averaging the six sources equally. A five-percentage-point increase in safe removed fraction corresponds roughly to one additional successful half-width selection per ten decisions. The predeclared rule for material superiority requires its paired task-bootstrap 95% lower endpoint to exceed 0.05, together with a nonpositive observed mean breach difference. Secondary contrasts are vote minus Gain, slack minus Gain, slack minus vote, and point minus Gain. The 30,000 bootstrap draws use seed 33119. Intervals summarize variability across these six purposively selected sources, with dependent folds kept together.

## Results

Every policy produced the same safe saved fraction and breach indicator on every fold. The equal-source mean safe saved fraction was 0.07343. Of 16 folds, each policy accepted a half-width nominee on six and incurred an outer breach on three. The other ten included one singleton anchor with no available reduction. The equal-source mean accepted fraction was 0.33333 and mean breach fraction was 0.16667. Equal weighting gives the single official Urban holdout the same source weight as each three-fold cohort.

| Source | Outer folds | Mean safe saved fraction, all four policies | Accepted | Outer breaches |
|---|---:|---:|---:|---:|
| CNAE-9 | 3 | 0.00000 | 1 | 1 |
| Ozone | 3 | 0.32946 | 2 | 0 |
| Parkinson | 3 | 0.00000 | 1 | 1 |
| SECOM | 3 | 0.11111 | 2 | 1 |
| Semeion | 3 | 0.00000 | 0 | 0 |
| Urban | 1 | 0.00000 | 0 | 0 |

For slack minus point the observed paired difference was 0.00000. The task-bootstrap interval, exact enumeration of all 46,656 ordered empirical task resamples, and leave-one-source-out range were all [0, 0]. Resampling ties reproduces ties. Six purposively selected sources cannot establish population equivalence. As a sensitivity calculation fixed before opening scored outcomes, if the six sources were independent bounded draws, Hoeffding's two-sided 95% interval for the mean paired difference, intersected with its possible range, would be [−0.50, 0.50]. The data give no evidence for the predeclared five-point advantage. Each of the four secondary contrasts likewise had zero paired difference on all 16 folds.

| Policy | Task-weighted AUC | Task-weighted accuracy | Nomination masks (total) | Fits (total) | Wall time (total s) |
|---|---:|---:|---:|---:|---:|
| Gain prefix | 0.86519 | 0.86619 | 0 | 62 | 12.04 |
| Discrete vote | 0.86101 | 0.86521 | 250 | 62 | 11.64 |
| Continuous slack | 0.86775 | 0.86653 | 1,334 | 62 | 13.31 |
| First-panel point | 0.85871 | 0.86653 | 125 | 62 | 11.36 |

Slack and point nominated different sets on two folds. The guard rejected both nominations on Parkinson fold 2 and accepted both on SECOM fold 0. On the latter, AUC was 0.77062 for slack and 0.60777 for point, with identical accuracy of 0.96486. Both breached the applicable average-precision threshold relative to the anchor (anchor 0.17111, slack 0.15171, point 0.07915). This single fold accounts for the entire equal-source AUC difference of +0.00905. The discrete vote gave all candidates zero on nine of 15 eligible folds, and tied its best candidate on ten. The corresponding numbers of tied best candidates across folds were two for continuous slack and four for point. The wall-time figures are single-run measurements on this machine.

An independent proposal replay scored all 110 candidate sets using direct weighted positive-negative comparisons for AUC. All discrete votes and nominations matched; the maximum absolute continuous-score discrepancy was 4.22 × 10⁻¹³. Independent calculations also verified every outer metric in all 64 policy/folds. Removing 33 CNAE outer documents and one Urban test object with feature vectors already present in development preserved zero for every safe-compression and breach contrast. The published fold table retains these rows.

## Reproduction

Run these commands in the thesis checkout. The scorer checks the prediction package against the current Git revision before loading target vectors. The committed prediction and outcome packages suffice for scoring. Rebuilding the original split data is necessary for the independent nomination and duplicate audits. Preparing a fresh run requires a fresh output folder or checkout because sealed layouts are write-protected by the runner.

```powershell
.\thesis-python.ps1 -m experiments.context_review.wide_data acquire
.\thesis-python.ps1 -m experiments.context_review.wide_data prepare
.\thesis-python.ps1 -m experiments.context_review.wide_score
.\thesis-python.ps1 -m experiments.context_review.wide_nomination_audit
.\thesis-python.ps1 -m experiments.context_review.wide_audit
```

For a new development-only prediction replay, run `experiments.context_review.wide_predict` after preparing folds in a separate checkout. It independently executes all four arms in rotated order and writes a new ZIP, including new wall times. Compare per-fold `predictions.npz` digests with the published ones. Sealed outer targets and raw source datasets are absent from the prediction package.
