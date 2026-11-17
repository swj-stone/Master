# Independent five-source scaling and restoration check

The classification-only policy and all comparisons were registered at `56fdf62` after the unsuccessful general scaling pilot. All 525 forecasts for 25 held splits and 21 arms were sealed at `c369845` before scoring. The five-source provider's preparation is committed at `de0f567`. The registered dataset unit is a source. CTG and Taiwan test the classification rule; the three regression sources retain unit slope by policy. The full set includes diagnostic always-scaled regression controls. Tecator has one fixed outer interpolation holdout.

The table gives five-split means. The score column uses AUC/accuracy for classification and R²/MAE for regression. Policy baselines all receive the same margin intercept, calibration radius and task-specific multiplier rule. Runtime includes teacher fitting, extraction/cache, method-specific ranking and selection, and prediction on restricted inputs. Native full uses the original XGBoost API.

| Source | Original width | Unit-slope exact | Policy exact | Matched Gain | Matched static ablation | Matched TreeSHAP | Policy exact score | Native full score | Policy / full seconds |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| CTG | 21 | 11.8 | 10.0 | 10.6 | 10.0 | 10.4 | .9693/.9097 | .9733/.9178 | .459/.386 |
| Taiwan | 20 | 16.8 | 16.0 | 17.2 | 16.0 | 16.0 | .7660/.8272 | .7706/.8259 | .167/.150 |
| Student Math | 25 | 18.6 | 18.6 | 20.2 | 18.8 | 19.0 | −.0907/3.560 | −.1041/3.603 | .142/.132 |
| Telemonitor | 18 | 3.6 | 3.6 | 3.6 | 3.6 | 3.6 | −.8185/7.842 | −.8708/7.852 | .168/.151 |
| Tecator | 100 | 22.4 | 22.4 | 27.6 | 23.4 | 25.0 | .3994/7.666 | .4535/7.164 | .152/.145 |

Over the two classification sources, scaling the exact path changes AUC by −.00187 and accuracy by −.00055 relative to its unit-slope version. Matching each method's scale policy gives geometric width ratios of .9201 versus Gain, .9904 versus static ablation, and .9661 versus native mean-absolute TreeSHAP across all five sources. Complete runtime ratios are 1.0006, .9999 and .9891, respectively. The matched static ablation yields the same two classification source-mean widths. Adding bounded restoration to both original exact and static starts gives an exact/static width ratio of 1.0024. These localized differences supply little support for a general compression advantage over static ablation.

Tecator is an explicit failure against the full teacher at the previous quality margins. Its R² drops by .054 and MAE increases by .501. Student Math and Telemonitor both have negative held-out R² even before compression, limiting interpretation. Two classification sources and three regression sources cannot establish population-level equivalence.

`scores.csv` contains every arm, score, reported physical margin error, scalar and clock; `source-means.csv` and `source-contrasts.csv` provide paired summaries. Forecasts, retained-input lists, paths, split indices and code digests are contained in the signed bundle `packages/minpi-scale-five-forecasts.zip`. The registered deployment guard checks teacher-margin fidelity. The original thirteen-source study separately reports a supervised performance guard along the visited path.
