# Official selector comparisons on the five frozen deployment tasks

This post-release experiment compares the unchanged anchored MinPI with two selectors from scikit-learn 1.9.0. Each uses the same 100-round depth-four XGBoost 3.3.0 CPU estimator, fitting rows and final development data as the prospective experiment. A fresh MinPI run reproduced all 25 prospective supports and prediction arrays exactly. No outcome or parameter from this comparison changed its policy.

- `SelectFromModel(estimator)` estimates feature importance on the original fitting rows and keeps normalized gain at or above its arithmetic mean. The implementation asserts that the fitted library threshold and support agree with that definition. One fresh XGBoost model is fitted on all development rows for deployment. [Official SelectFromModel documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html)
- `RFECV(estimator, step=.2, min_features_to_select=1, cv=PredefinedSplit(...), scoring=..., n_jobs=1)` performs recursive elimination on the fitting rows, selecting the width by the held pooled validation rows, using binary ROC AUC, macro OVR AUC or R² as appropriate. The official implementation then repeats elimination to the chosen width on all development rows and refits its final estimator there. Its own `predict_proba` or `predict` call yields the deployment forecasts. These outcomes concern the one-fold validation protocol matching the experiment. [Official RFECV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html)
- Every train-through-test-prediction timer ran on logical CPU 8 with one model thread. After synthetic warm-up, the three methods occupied each timing position once per seed. Data I/O and later test scoring were excluded throughout. The measured source files, installed estimator signatures, predictions, split row indices and per-repetition records are archived in this directory. The selector kernel used in the prospective MinPI measurements remains unchanged.

Each cell below contains median retained width, seconds and primary test metric across the five seeds, in that order. The primary metric is macro OVR AUC for the first three rows and R² for the last two. Median timings first aggregate the three complete runs for each seed.

| Source | MinPI | RFECV | SelectFromModel |
|---|---:|---:|---:|
| Vowel | 3 / .677 / .870 | 4 / 1.437 / .856 | 4 / .278 / .876 |
| HCV | 3 / .104 / .998 | 3 / .186 / .963 | 3 / .032 / .985 |
| Shoppers | 3 / .215 / .665 | 3 / .558 / .673 | 4 / .080 / .693 |
| Garment | 4 / .076 / .215 | 5 / .160 / .239 | 2 / .027 / .133 |
| Tetouan | 6 / .238 / .676 | 8 / .415 / .673 | 2 / .060 / .451 |

The registered *paired* measure of speed and width requires a time saving and width excess of at most max(1 column, 5% of comparator width). MinPI satisfied it against RFECV on 5/5 source medians and 25/25 seed comparisons, and against SelectFromModel on 0/5 and 0/25. Adding the requirement of primary score loss at most .01 reduces the RFECV results to 4/5 sources and 18/25 individual seeds. Garment's paired median R² changed by −.0247. RFECV and MinPI retained a median five and four columns there. For Tetouan the two-column reduction cost no median R². Several time-based tasks reuse the same chronological split across all five seeds.

Default SelectFromModel was faster than MinPI on 25/25 comparisons. Median paired MinPI time ratios range from 2.47 to 3.91. MinPI's median paired R² gains were .0816 on Garment and .2247 on Tetouan, with respectively two and four additional columns. On Shoppers, SelectFromModel also improved median AUC by .0283. MinPI passed the joint criterion against SelectFromModel on 0/5 sources. The primary score in this criterion does not constrain the secondary metric: Garment's default SelectFromModel has lower median MAE (.1028) than either alternative (.1049 for MinPI and .1035 for RFECV).

The [paired source results](sources.csv) use medians of within-seed differences. These may differ from differences between the marginal medians above. [Per-seed metrics](seeds.csv) include accuracy and MAE; [the timing ledger](timings.csv) contains all 225 measurements. [The signature record](signature.json) gives exact library versions and code hashes. Use `./thesis-python.ps1 -m experiments.anchored_vote.standard report` from the thesis checkout to rebuild the tables against the saved predictions and active source signatures. `measure` verifies completed receipts and checks active signatures before reusing them.

