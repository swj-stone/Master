# Held-unit uncertainty for fixed prospective predictions

Compare anchored MinPI with the validation-maximizing exhaustive Gain selector on the five prospective datasets. For each seed, subtract Gain's held score from MinPI's held score on the same weighted rows. Take the median of these five paired differences. Positive differences favour MinPI for AUC, accuracy and R². Negative differences favour MinPI for MAE.

The postrelease analysis keeps both fitted models and their chosen features fixed. It varies the composition of the held sample. In each of 1,999 draws, every occurrence of a given held unit receives the same weight across methods and seeds. Vowel resamples the seven official test speakers with replacement. HCV resamples the union of 412 held input-profile groups with replacement. Each seed uses only the drawn groups present in its own held set. An undefined seed-level AUC invalidates that draw for AUC. The data supplied all classes in every draw. Numbers of valid and skipped draws are recorded for each metric.

Garment resamples circular blocks of three observed work dates from the 12 held dates. Tetouan uses five-day blocks over 73 held days. Block length is the ceiling of the cube root of the number of held units. The algorithm samples starting positions uniformly, concatenates blocks and truncates at the original number of dates. Circular wrapping and the short time windows limit what this calculation says about future periods. Shopping supplies only one held month. A between-month interval cannot be estimated from it.

| Task | Primary metric | Paired point difference | 95% held-unit interval | Companion metric | Paired point difference | 95% held-unit interval |
|---|---|---:|---:|---|---:|---:|
| Vowel | AUC | 0.0000 | [−0.0168, 0.0120] | Accuracy | 0.0000 | [−0.0476, 0.0498] |
| HCV | AUC | −0.0006 | [−0.0027, 0.0000] | Accuracy | −0.0082 | [−0.0167, 0.0000] |
| Shopping | AUC | 0.0000 | Single month | Accuracy | 0.0000 | Single month |
| Garment | R² | −0.0061 | [−0.0614, 0.0449] | MAE | 0.0002 | [−0.0027, 0.0032] |
| Tetouan | R² | −0.0015 | [−0.0066, 0.0031] | MAE | 19.64 | [−8.41, 50.29] |

The endpoints are the 2.5th and 97.5th percentiles. The intervals condition on the collected sites, held periods and fitted models. They omit training and feature-selection uncertainty and cannot support a population-wide claim about datasets. The Vowel and Garment ranges include primary-score losses greater than 0.01. [Exact results and skipped counts](held-unit-uncertainty.csv).

From the thesis worktree, run `./thesis-python.ps1 -m experiments.anchored_vote.held_unit_uncertainty` with the three completed prospective timing directories present. The script verifies the 375 receipts before reading outcomes, checks its point estimates against the released seed table, and compares its weighted scoring formulas to scikit-learn on the original sample and two resamples for each seed.
