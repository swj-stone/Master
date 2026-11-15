# Choosing a nomination rule on other data sources

This retrospective check addresses the practical use of the forest comparisons when the new source has no candidate outcomes. The [fixed design](../../configs/context_review/transfer_choice.json) holds out all folds and candidate draws of one source. Training uses the nine or five other sources in the same panel and candidate construction. The held source receives the highest training-average net-return rule. The four available procedures differ only in how they break a maximum-vote tie. A second library also admits a single point comparison over the whole candidate slate. The latter is a deployable baseline that can skip vote computation.

Expected compression and breaches for the uniform tie rule are exact over its candidate lottery. The [held-source records](../../thesis-paper/evidence/transfer_choice/held.csv) contain the chosen method, number of training sources, training net return for selected rules, held-source U, breach, AUC, accuracy, and differences against the fixed single point. The [summary](../../thesis-paper/evidence/transfer_choice/summary.csv) weights sources equally and reports wins, ties and losses. Its net measure subtracts zero, one half or one times the breach indicator from realized effective compression. None of the three penalties charges for computation. The nomination times from the shared sources are in the [matched timing experiment](matched_stopping.md).

At penalty zero, transferring the four-rule preference gains 1.14 percentage points of mean U over the fixed whole-slate point on local ten, changes nothing on Gain wide, and loses 5.00 and 1.81 points on Gain ten and local wide. Leaving Dry Bean out accounts for the Gain ten loss. At penalty one, the differences are +2.25, −7.57, −4.17 and −1.35 points respectively. Local ten selected original order on every omission. The rule learns a single preference from past sources and uses no task descriptors. Repeated omitted-source trainings overlap, and the source collections are purposive.

From the thesis checkout, regenerate both tables from the archived forest outcomes.

```powershell
.\thesis-python.ps1 -m experiments.context_review.transfer_choice
```

The [implementation](transfer_choice.py) checks that the guarded original-order outcomes coincide across the two source archives on all 184 slates before adding the whole-slate point comparison.
