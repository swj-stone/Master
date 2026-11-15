# Independent identification and refit audit

This audit examines the stump proposition, the collision counts, the historical opportunity analysis, and the external limited-training control. The independent numerical scripts read prediction arrays and plan records. Neither imports scoring, model-fitting, collision, or opportunity functions from the experiments. I used direct pair enumeration for the stump, average-rank Mann–Whitney AUC for the prediction arrays, and direct XGBoost fits for the model replay.

## Identification and empirical collisions

* The fitting example determines two distinct leaf probabilities, 1/4 and 3/4, for the reduced model. Changing the evaluation rows' retained values leaves the reduced fitted parameters fixed. Its evaluation predictions and their pairing with the fixed labels can change. The frozen full model only uses the deleted feature, so all per-row frozen mask predictions survive such a change.
* For each feasible positive/high-score overlap, the fixed class totals determine the entire two-by-two table. Directly scoring every possible assignment with two through ten equally weighted rows and at least one observation of each class recovered exactly the proposed AUC/accuracy set for each pair of margins. This covers 16,388 assignments and 870 tables. Both endpoints are attainable on each grid. Population sharpness over all compatible joint laws permits splitting a class mass, including conditioning on the fixed distribution of the deleted coordinate. A panel having arbitrary **fixed weights** preserves score invariance. Unequal row weights need a different finite attainable-set claim. The manuscript restricts the integer-grid claim to equally weighted rows.
* Simultaneous monotonicity gives the three-way endpoint decision with equality at the pass boundary. For prevalence 0.10, retained prevalence 0.08, and a perfect anchor, the best AUC is 0.90 although the best accuracy is 0.98. At balanced margins, the same observable transcript within each construction supports both a pass and a failure. Knowing the labels, fitting inputs, support identities, and seeded learner does not recover the missing pairing of **evaluation** inputs and labels. Observing that pairing does. Adaptive queries based on the common transcript also coincide when the same random seed is used in both worlds. Support selection or panel construction based on the hidden pairing would violate this premise.
* With a fixed ambiguous observation and adversarial choice of world before the independent random draw, veto probability $m/(a+m)$ minimizes the larger of $az$ and $m(1-z)$. A deterministic decision would have minimax cost $\min(a,m)$. A stated prior over the worlds would specify a Bayes problem. The stochastic minimax claim has the appropriate restricted decision problem.
* Running `lean IdentificationBounds.lean` from `thesis-paper/formal` checked all seven printed theorems, with only `propext`, `Classical.choice`, and `Quot.sound` reported collectively. The Boolean pointwise invariance result uses no axioms. The formal argument models which leaf masks reach and proves rational identities and inequalities. The distributional attainability and finite assignment arguments are supplied in the manuscript, with independent finite checks.
* From independent scoring of 704 XGBoost and 704 logistic reduced predictions, the archived XGBoost score tensors and fold-level support sets reproduce **all 30** reported collision rows. The primary separating ballot matrix yields 491 equivalence classes. The 31 mixed classes cover 136 proposals on eight tasks and force 41 binary errors in the observed union. Width and worst approval count force 88 and 71. Replacing the refitted classifier by logistic regression raises the complete-ballot floor to 51. Continuous separating metrics have no opposite-outcome class, even after individually rounding each entry to two decimal places in this replay. The lookup floor applies only to equal task/fold/width/transcript groups under a common decision. It gives no held-out error guarantee. Candidate identities, training features, or raw prediction trajectories would be different observations.

## A constructive check using the actual split-used anchor

The eight-row depth-one stump never splits on $R$. Compact Gain applied to that stump would only admit $D$, and no half-width action would be available. Its exact analytic characterization concerns a supplied anchor containing both columns. I searched synthetic binary tables for a separate example in which the experimental anchor **does** contain both features and the deployed Gain policy nominates and retains $R$.

The deterministic reproducer is `transport-forensic-xgb-witness.py`. It calls the actual `transport_predict.policy` entry point for each constructed world and uses XGBoost 3.3.0 with the experimental 100 rounds, depth four, learning rate 0.1, CPU histogram splitting, one thread and seed 20421. The partitions are supplied synthetic training and evaluation inputs. The search concerns existence, not frequency. No UCI outcomes were used in generating the witness.

| Coordinates $(D,R)$ | Negatives in proposal | Positives in proposal | Negatives in guard | Positives in guard |
| --- | ---: | ---: | ---: | ---: |
| 0,0 | 19 | 19 | 0 | 0 |
| 0,1 | 9 | 9 | 1 | 1 |
| 1,0 | 20 | 1 | 46 | 0 |
| 1,1 | 1 | 5 | 2 | 33 |

Use six copies of the 83-row proposal table for the initial fit (498 observations), one for the proposal partition and the 83 separate guard observations listed above. In both worlds all 664 development inputs and labels, support identities, learner, random seed, fitting medians $(0,0)$, nominations and guard checks are identical. The initial full predictor's positive probabilities on 00, 01, 10 and 11 are 0.4999956, 0.4999956, 0.04766528 and 0.83325994. Both features occur in splits. Compact Gain orders $R,D$; masking to the one-column prefix yields proposal AUC 0.5 and accuracy 49/83 and fails the 8% anchor allowance. Its anchor therefore contains both features. The actual half-width Gain policy nominates $R$. For the full and candidate models respectively, guard AUC is 0.97869148 and 0.96938776. Both guard accuracies equal 80/83. The candidate passes the 3% guard.

Consider 332 outer rows with 136 fixed positive labels, constant $D=0$, and exactly 136 values $R=1$. In the first world $R=Y$. In the second, all 136 high values are given to negative cases. For each of the four possible masks and each outer row the entire vector of initial class probabilities is bit-identical across the two worlds. In fact all 100 initial trees route 00 and 01 to the same leaf; so do all 100 guard-trained full trees. The final refitted anchor also emits identical positive probabilities (0.4999969) at 00 and 01 in its stored prediction precision. Some of its individual tree paths differ. Its AUC is 0.5 and its accuracy is 49/83 in both worlds. The final reduced model emits 0.30502236 at $R=0$ and 0.6438761 at $R=1$. Its AUC and accuracy are both 1 in the first world. In the second they equal 15/98 and 15/83. These follow directly from 0 true positives, 136 false negatives, 136 false positives and 60 true negatives; tied pairs get half credit. The same admitted candidate passes the measured outer tolerance in one world and breaches in the other. The implementation uses one initial fit, one candidate and one anchor guard fit and one production fit in each world.

This construction intentionally changes the outer feature-label coupling and uses repeated development vectors. All cell types have positive counts in the original fitting table, and all four partitions have the same class prevalence. It only establishes that the actual pipeline can encounter indistinguishable frozen measurements on the constructed cases. Raw labeled outer inputs distinguish the worlds. It generalizes neither the stump's exact sharp set to arbitrary ensembles nor the event's probability in future data.

## Opportunity and limited-training replay

I reconstructed half-width pools, their Gain tie order, nominal savings, both feasibility ceilings, final policy choices, and outcomes from the manifests and probabilities. For all 30 UCI folds, the low-fidelity control specified eight boosting rounds using the fitting-only rows with training-only median imputation. Each candidate and its small-model anchor were scored on the unweighted proposal rows. The resulting 126 fits produced the stored scores and nominees. Every actionable nominee was the unique top-scoring candidate to within $10^{-10}$. Ten singleton anchors had zero candidate and small-model fits.

I independently trained **all 272** UCI guard/final 100-round models (136 distinct fold-supports per stage) and **26** additional 100-round historical models on two PMLB folds. All subsequent probabilities were bit-for-bit identical to the archives. Complete-slate evaluation produces counterfactual probabilities. For deployment, only the nominated candidate and its anchor enter the full-fidelity guard check. The final production refit additionally uses the guard rows. The limited-fidelity screen averages 4.2 additional fits per UCI fold. The stored component timings were aggregated, without an independent hardware timing experiment.

| Collection | Folds | Singleton | Candidates | Neither passes | Guard only | Outer only | Both pass | Folds with any outer pass | Folds with any joint pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical PMLB | 41 | 1 | 266 | 158 | 26 | 28 | 54 | 22 | 20 |
| External UCI | 30 | 10 | 106 | 57 | 15 | 7 | 27 | 10 | 9 |

The separate official Optdigits half-width pool contains eight candidates, and none passes the measured outer criterion. All historical 1%, 3%, and 5% outer-tolerance results, with a fixed 3% guard, match the published point estimates and 30,000-draw task-resampling intervals. The 3% PMLB nominal, outer, and joint ceilings are 45.910%, 24.898%, and 22.573%. The UCI values are 29.067%, 15.178%, and 13.512%.

| UCI nomination | Within-tolerance savings | Accepted candidate / 30 | Outer breaches / 30 |
| --- | ---: | ---: | ---: |
| Gain | 0.000% | 4 | 4 |
| Minimum vote | 1.333% | 5 | 4 |
| All-context margin | 7.293% | 9 | 4 |
| Point margin | 7.293% | 7 | 2 |
| Eight-round refit | 6.333% | 8 | 4 |

The eight-round minus point-margin contrast is −0.960 percentage points, with a paired whole-task percentile interval of [−5.657, 3.889]. All 16 limited-training contrast estimates and intervals matched. These outcomes support a descriptive comparison of the fixed policies on the measured tasks. They give limited precision for the population difference between the two ranking methods.

For a uniform random draw from each of the original external candidate pools followed by the same guard, the expected per-fold savings are the nominal width saving times the fraction of candidates passing both guard and outer criteria. Each singleton contributes zero. Exact enumeration with rational arithmetic produces equal-task expected savings of 13043/237600 (5.48948%), expected breach proportion 4/45 (8.88889%) and acceptance proportion 5/24 (20.83333%). Point margin exceeds this conditional random-policy expectation by 1.80345 percentage points; a pointwise 30,000-draw whole-task percentile interval is [−0.70286, 4.30989] points. The eight-round difference is +0.84386 points [−2.99832, 5.62500]. `transport-forensic-uniform.py` reproduces these comparisons from the saved probabilities without choosing a random realization.

## Two post hoc challenges

**Guard and final refitting.** I applied each guard-trained learner and its guard-trained anchor to the outer cases, then compared those decisions with the smaller-sample guard assessments and with the all-development refit on the same outer cases. The independent guard-trained and final probabilities agree with the originally saved arrays on their respective recorded evaluation sets.

| Guard result | Same model on outer rows | Final refit on outer rows | Number of candidates |
| --- | --- | --- | ---: |
| Fail | Fail | Fail | 52 |
| Fail | Fail | Pass | 2 |
| Fail | Pass | Fail | 5 |
| Fail | Pass | Pass | 5 |
| Pass | Fail | Fail | 14 |
| Pass | Fail | Pass | 4 |
| Pass | Pass | Fail | 1 |
| Pass | Pass | Pass | 23 |

Of 15 guard-accepted candidates breaching after final refit, 14 already breach when the guard-trained models move to the outer rows. Of seven guard-rejected candidates passing after final refit, five already pass with the guard-trained models on those rows. The other outcomes show that increasing the training set can also change which side of the relative threshold a candidate falls on. All comparisons refit the anchor on the same rows as the candidate. This replay describes finite decisions and cannot assign a population cause.

**Duplicate records.** I removed outer records whose complete raw predictor vector matched any development record and re-scored the original fixed policies and the complete pool. This removed 29 Banknote, 283 Blood, 86 Dry Bean, 438 Mammographic and four QSAR records (840 total). Every remaining fold still contained all classes. All 150 method/fold breach and savings outcomes and both aggregate ceilings stayed the same. This sensitivity conditions on the recorded training, folds and nominations. It does not establish performance under group-level splitting or a new source population.

## Scope and verdict

No mathematical or numerical defect remained in the precisely scoped identification, collision or opportunity results inspected. The Gini-stump sharp region is a special construction. Marginal constraints, identical transcript, and a fixed refit learner are essential. The observed availability ceiling depends on the measured candidate pool, budget, guard, training rows and outer cases. The retrospective PMLB decomposition and the ten-task UCI test should remain distinct. The rank scores and decisions were calculated without passing outer labels into the fitting workers. Consistent checksums and code attest to this local separation; they are not an independent external timestamp.

The model itself never learns the new evaluation pairing in the stump construction. Early wording “Only the reduced fit sees the new pairing” could imply otherwise; tell the reader that parameters are fixed and the evaluator compares the resulting scores with the labels. The pricing statement should retain “randomized” and “worst case,” and the wording of transport claims should retain the observed candidate budget, tested datasets and task-level uncertainty.

Run `thesis-paper/audit/transport-forensic.py`, `thesis-paper/audit/transport-forensic-uniform.py`, `thesis-paper/audit/transport-forensic-novel-rows.py`, and `thesis-paper/audit/transport-forensic-xgb-witness.py` using `thesis-python.ps1` from the thesis checkout. The scripts do not modify experiment records. The synthetic witness uses a temporary directory for the normal policy entry point's selection receipt.
