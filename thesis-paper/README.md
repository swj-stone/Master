# Finding Negative Min-Marginal Votes for Feature Selection

[Manuscript](minpi.pdf) · [Standalone LaTeX source](minpi-source.zip) · [Replication package](../packages/thesis-minpi-compression.zip) · [Protocols and commands](../experiments/minpi_compression/README.md)

The [shared-ridge sampler](../src/minppi_mofs/sampled_linear_minpi.py) evaluates features against common base coalitions. The [partial-vote procedure](../src/minppi_mofs/dictator_minpi.py) reuses observations across features. The [chain sampler](../src/minppi_mofs/chain_minpi.py) localizes negative marginal votes by querying nested supports. Negative witnesses certify the vote change on their coalition. The held-out prediction comparison measures the effect of using these rankings to choose a model.

The [eight-source audit](audit/minpi-compression-audit.md) checks the first predictive panel and exhaustive small games. The [four-source validation](evidence/minpi_compression/chain-validation/README.md), [matched Gain comparison](evidence/minpi_compression/legacy-gain/README.md) and [wider source transfer](evidence/minpi_compression/wide-transfer/sources.csv) give separate results. The archive includes the observed clocks and predictions, the [ridge identities](formal/SampledMarginal.lean), and proofs about [positive indices](formal/Dictator.lean) and the [sharp upper bound](formal/PartialUpper.lean).

On six further sources, the frozen witness-deletion rule makes one proposal from a common coefficient-ranked prefix. Its matched controls remove the same number of low-coefficient or random columns and face the same acceptance guard. The [prospective study](evidence/minpi_compression/witness-transfer/README.md) reports the eligible columns before outer scoring, source-level predictions and complete selection times.

To rebuild the current paper and replication package from the thesis checkout:

```powershell
Push-Location thesis-paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
Pop-Location
.\thesis-python.ps1 thesis-paper/package_revision.py --compression-release --verify-source
```

The [release instructions](../experiments/minpi_compression/release.md) give the offline forecast audit and directions for rerunning the original six datasets.

## Frozen R14 screening study

[R14 paper](supplement/minimum-vote-study.pdf) · [R14 standalone LaTeX source](supplement/minimum-vote-study-source.zip) · [R14 replication package](../packages/thesis-minimum-vote-paper.zip) · [Independent checks and scope](audit/minimum-vote-review.md)

The [selector](../src/minppi_mofs/minimum_vote.py) returns the earliest candidate with the largest minimum approval count. The [formal proof](formal/VoteCertificate.lean) gives the exact condition for the winner to be fixed by the observed votes. The paper proves the query bound for fixed restoration order and gives its exception. [Per-slate comparisons](evidence/vote_certificates/folds.csv) · [Raw nomination times](evidence/vote_certificates/timings.csv).

From the thesis checkout, run the following command to regenerate the new query-count tables and verify all 1,656 timing records in an isolated extraction. Both the short paper package and the [frozen R13 prediction archive](../packages/thesis-decision-evidence.zip) are required.

```powershell
.\thesis-python.ps1 thesis-paper/package_revision.py --verify-screening packages/thesis-minimum-vote-paper.zip
```

## Earlier identification and decision study

[Historical 46-page paper](supplement/decision-study.pdf) · [Its standalone source](supplement/decision-study-source.zip) · [Data, controls and replay](evidence/decision-release.md) · [R13 reviewer response](final-review/response.md)

The matched lottery, joint candidate decisions, class-preserving controls, new full-pipeline timings and complete-source transfer checks are indexed in the [decision release](evidence/decision-release.md). Earlier protocols and their archives follow below.

The study characterizes attainable refit outcomes for fixed finite tree ensembles. On a known dictionary of outer rows, the unmasked probability vector alone determines the observed tolerance decision for all 704 historical proposals; over 93% of row occurrences have a unique vector within their fold. Frequency-matched randomized identifiers test how much this row resolution can explain. [Finite-ensemble characterization](ensemble-identification.tex) · [Quantization and random-code controls](evidence/score_resolution/)

In a matched Gain shortlist with a single masked point margin as the tie rule, adding the minimum vote changed one nomination in ten tasks and none in the six wider tasks. With a random forest for both masking and refits, the continuous margin beat the ordinal rule on the original Gain slates. In a second design with ANOVA anchors and local candidate exchanges, ordinal voting led on the ten-task average over three draws, and the sign on the wider panel changed with the draw. [Matched Gain comparisons](evidence/gain_increment/) · [Crossed learners](evidence/forest_cross/) · [Replicated neighborhoods](evidence/neighborhood_replicates/)

The paper asks how much a fitted classifier's response to masked columns can tell us about training a new classifier on the surviving columns. A finite tree construction characterizes the possible refit AUC and accuracy under a common mask transcript. Of 704 historical feature subsets, 136 shared a binary vote pattern with an equally wide subset on the same fold that had the opposite refit outcome. [Identification argument](identification-bounds.tex) · [Collision counts](audit/refit-observation-collisions.csv) · [Held-task prediction of refit failure](../experiments/context_review/matched_signal.md)

The two new UCI panels compare policies that nominate one candidate from a fixed half-width slate. Across the first ten tasks, a single continuous mask margin saved 7.29% of anchor columns within tolerance, compared with 1.33% for the minimum vote. Choosing uniformly from the same candidates gave 5.49% in expectation. On the six wider tasks all four primary policies attained 7.34%, with three outer breaches each. A full-slate evaluation quantifies the opportunity each policy had under the common refit guard. [Ten-task protocol](../experiments/context_review/transport_external.md) · [Wide-panel protocol](../experiments/context_review/wide_panel.md) · [Random-nomination comparison](evidence/chance/)

The logistic-regression replication kept the tree-selected nominees and changed the learner used for the guard and final refits. On the wide panel, the full continuous margin saved 10.83% against 8.21% for the single margin, with a paired task-bootstrap interval of 0.00 to 7.84 percentage points for their difference. Uniform nomination averaged 10.99%. [Receiver design](../configs/context_review/receiver_transport.json) · [Receiver results](evidence/receiver/means.csv)

Uniform selection within the highest-vote group gave 6.16% on the ten tasks and 9.49% on the six wider tasks with XGBoost refits. The continuous tie-break gave 7.29% and 7.34%. On the wider tasks, before outer scoring, the guarded supports already limited the full-versus-single margin difference to 2.78 percentage points in either direction. Under two 16-row masking backgrounds, this comparison ranged from a 5.44-point gain to a 5.50-point loss on the same wide data. [Tied ballots and decision bounds](../experiments/context_review/decision_opportunities.md) · [Background sensitivity](../experiments/context_review/marginal_background.md)

## Earlier operating analyses

The operating analyses estimate the cost of a veto on an individual, previously measured proposal. At a unit cost for rejecting a passing deletion and a cost of `r` for missing a failing refit, the separating minimum beats the endpoints on the search paths when `r > 20/11`. A 30,000-draw task/fold resampling range for the crossover extended from 0.46 to infinity. On drawn proposals both policies missed the same two failures and the separating minimum gave nine extra false alarms. Against a wider library including a deletion-size threshold and the constant decisions, the primary rule never minimized observed path cost. Choosing rules on 13 tasks and scoring the fourteenth favored the size library on drawn proposals at all four miss costs studied. [Clustered rates and frontier](../experiments/context_review/cluster_cost.md) · [Transferred policies](../experiments/context_review/cluster_cost_transfer.md)

The 360 settings per voter show how the measured trade changes with aggregation, quorum, tolerance and the downstream learner. Across 400 search proposals, switching to logistic refits changed 111 failure labels. Adding worst-context dissent to a size model trained on other tasks improved held-out proposal-ranking AUC by 0.024 on paths and 0.046 on drawn proposals. Both paired intervals for the primary log-loss change included zero. [Rule sensitivity](../experiments/context_review/rule_sensitivity.md) · [Incremental prediction](../experiments/context_review/incremental_prediction.md)

Giving each policy one candidate refit at half-anchor width changed the operational conclusion. The vote changed five of 41 PMLB nominations and four final sets. It raised task-mean compression within the outer tolerance from 9.33% to 10.12% and increased breaches from one to three. All the added compression credit came from one New Thyroid fold. Choosing a single nominee across widths with a learned model trained on drawn deletions raised compression credit from 8.94% to 11.92% and breaches from one to four. This was a retrospective extension. On Optdigits the fixed-budget rules selected identical columns at each budget. [Guarded policies and separate writer holdout](../experiments/context_review/budgeted_guard.md) · [Independent prediction replay](audit/operating-guard-audit.md)

Stand-alone half-width timings span **168 completed executions** on one logical CPU, with 164 PMLB runs and four separate Optdigits runs. The PMLB task-weighted means were 0.8423 s for Gain and 0.8658 s for voting (paired difference +0.0235 s, 95% task interval −0.0584 to +0.1046). Each execution checked its nomination, final support and predictions. [Timing records](../results/context_review/guard-clock/runs.csv) · [Portable inputs](evidence/operating/README.md)

## Earlier experiments and controls

| Question | Observation and scope |
|---|---|
| Did the specified two-feature mechanism occur? | A constructed classifier realizes the predicted pattern. No ordered pair explained all measured decisions for any of 226 eligible proposals from the six development tasks. This test applies to the exact pattern on the measured masks. [Mechanism audit](audit/pair-model-protocol.md) |
| Did more contexts find additional losses after refitting? | Against empty and full restorations, extra contexts detected 11 more losses and falsely rejected 20 more passing refits in 400 path proposals. Seven detections came from Sonar. On 379 separately drawn deletions, the changes were zero and nine. The samples shared 75 fold–support pairs. [Task-level figure](figures/refit-vetoes.pdf) · [Refit audit and sensitivity](audit/refit-review.md) |
| What remained after a direct refit check? | The guarded vote and its guarded same-width Gain prefix both averaged 0.93327 outer AUC. The independently sized half-anchor Gain prefix with the same check averaged 0.93507. These comparisons revisit 14 PMLB tasks. They cover 41 folds because one ranking split could not be divided as specified. [Numerical evidence](evidence/reassessment/) |
| How well did the rule transfer? | In the official HAR participant holdout, the 22-column vote reached 89.40% mean participant accuracy. Same-width RFE reached 89.47% when selecting on the 15 fitting participants, or 90.68% using all 21 training people. In three future gas-sensor batches, the six-column vote reached 35.81% mean balanced accuracy and same-width RFE reached 41.72%, 52.63% and 57.80% using the first five, six and seven batches to select. For each primary protocol the supports and predictions were saved before the local scorer opened the public holdout labels. [HAR design](../experiments/context_review/external_har.md) · [Drift design](../experiments/context_review/drift.md) · [Retrospective history controls](../experiments/context_review/matched_history.md) |
| Did evaluating the minimum make fitting faster? | Recent-veto ordering saved 300 of 3,304 unique masked evaluations over 42 folds. With batching and deferred scores, the paired median reduction in complete wall time relative to fixed order was 2.12%, below the locally specified 5% target. The timed interval includes preprocessing, selection, the final fit and prediction. [Repeated timing protocol](../experiments/context_review/timing-process.md) |
| What if the panels vote directly on retrained predictors? | On a common four-width grid, adding class-specific recall checks reduced the number of folds with an observed class loss from 16 to eight. Using four panels instead of one changed the selected width on four folds, resulting in seven class losses, three global losses and slightly lower mean AUC. [Grid, thresholds, shared fits and scores](../experiments/context_review/refit_vote_grid.md) |

The refit experiments count a deletion as harmful when the outer-fold comparison exceeds a stated relative loss tolerance. Finite test folds give an uncertain estimate of predictive loss. The historical PMLB proposals, folds and study outcomes had been seen by the time of reassessment. The later HAR, gas-sensor and Optdigits studies used locally recorded protocols and publicly available test labels. The nine HAR participants are from one study, and the gas batches come from one sensor array.

## Reproduce the historical tables and formal checks

Run the following commands in the `MastersThesis` checkout. The three historical generators read portable [benchmark evidence](evidence/revision/), [refit evidence](evidence/reassessment/) and [operating evidence](evidence/operating/README.md). The formal directory pins the Lean version. The [decision release guide](evidence/decision-release.md) covers the R13 data and scoring commands.

```powershell
.\thesis-python.ps1 thesis-paper/revision_results.py
.\thesis-python.ps1 thesis-paper/reassessment_results.py
.\thesis-python.ps1 thesis-paper/operating_results.py
Push-Location thesis-paper/formal
lean MinPI.lean
lean ContextContrast.lean
lean WorkingFamily.lean
lean EqualFrequency.lean
lean RefitTarget.lean
lean Challenger.lean
lean ProposalCost.lean
lean IdentificationBounds.lean
lean TranscriptCells.lean
lean DecisionEnvelope.lean
lean JointChoice.lean
lean VoteCertificate.lean
Pop-Location
```

The refit counterexample constructs two populations with the same mask scores. The best score available from the surviving variable has AUC 0.99 in one and 0.50 in the other. Lean verifies the finite arithmetic, and a separate Wolfram Language program enumerates all four masks. A second Lean file proves that a running minimum bounds a candidate's final score, which justifies stopping evaluations once the candidate cannot beat the incumbent. Neither proof estimates the error of a fitted model. [Refit arithmetic](formal/RefitTarget.lean) · [Wolfram enumeration](formal/RefitTarget.wl) · [Pruning lemma](formal/Challenger.lean)

The [proposal-cost theorems](formal/ProposalCost.lean) check the integer break-even arithmetic and the finite-sample alternatives to the primary rule. An independent [Wolfram calculation](formal/ProposalCost.wl) solves the corresponding inequalities over real nonnegative cost ratios. A further [symbolic check](formal/IdentificationBounds.wl) covers the attainable-set result. Run the Wolfram checks from the thesis checkout with a configured Wolfram Engine.

```powershell
wolframscript -file thesis-paper/formal/RefitTarget.wl
wolframscript -file thesis-paper/formal/ProposalCost.wl
wolframscript -file thesis-paper/formal/IdentificationBounds.wl
wolframscript -file thesis-paper/formal/TranscriptCells.wl
wolframscript -file thesis-paper/formal/JointChoice.wl
```

The historical command verifies the archived R14 source against its PDF and replays its screening tables from the R13 prediction archive. The earlier transport and operating archives retain their original names and digests. [Short-paper package](../packages/thesis-minimum-vote-paper.zip) · [Frozen R13 evidence](../packages/thesis-decision-evidence.zip) · [Previous robustness archive](../packages/thesis-robustness-evidence.zip) · [Earlier transport archive](../packages/thesis-transport-evidence.zip) · [Archive checksums](evidence/revision/artifacts.json)

```powershell
.\thesis-python.ps1 thesis-paper/package_revision.py --screening-release --verify-source
```

For a fresh collection from the completed runs, see the [campaign commands](../experiments/context_review/README.md). `operating_results.py --collect` refreshes the [portable tables](evidence/operating/README.md) and rebuilds the manuscript tables.

## Earlier context-design experiments used in the reassessment

The first PMLB campaign established the original 14-task, three-fold grid. Its nested, separating and random families shared a common Compact Gain anchor. The exhaustive direct Gain comparator added later refits a fresh model at each ranked width. The archived identification study reports both comparators.

| Material | Source |
|---|---|
| Frozen six-task search proposals and a constructed mechanism | [Original diagnostic protocols](../experiments/context_diagnostics/README.md) |
| PMLB selectors and same-width controls, including the regrouped sensitivity study | [Original protocols and definitions](../experiments/context_study/README.md) · [Independent implementation audit](audit/rounds/README.md) |
| Three high-dimensional NIPS tasks assessed on the public challenge validation data | [Data and settings](../configs/context_study/nips2003.json) · [Task means](evidence/revision/challenge-by-dataset.csv) |
| Restoration order and complement-only delta debugging | [Search implementation](../src/minppi_mofs/veto_schedule.py) · [Comparators](../experiments/context_study/schedule_controls.py) · [Trace replay](audit/rounds/replay_revision.py) |
| Prior data archives | [Original PMLB runs](../packages/thesis-pmlb-evidence-r1.zip) · [Challenge and context-control runs](../packages/thesis-context-revision-evidence.zip) |

From the thesis checkout, reproduce the high-dimensional and search-control runs as follows. The first command retrieves the public archives. The high-dimensional driver applies a separate one-hour process limit to each inner split and writes its own output directory. The search controls require the completed primary PMLB run linked above.

```powershell
.\thesis-python.ps1 -m experiments.context_study.fetch_challenge
.\thesis-python.ps1 -m experiments.context_study.run_challenge
.\thesis-python.ps1 -m experiments.context_study.working_set
.\thesis-python.ps1 -m experiments.context_study.schedule_controls
.\thesis-python.ps1 -m experiments.context_study.random_families
.\thesis-python.ps1 -m experiments.context_study.fixed_margins
.\thesis-python.ps1 -m experiments.context_study.matched_selectors
.\thesis-python.ps1 thesis-paper/audit/rounds/replay_revision.py
```

## Earlier six-dataset study

[Earlier manuscript](../packages/prior-context-vote/minpi.pdf) · [Earlier submission source](../packages/prior-context-vote/minpi-source.zip) · [Earlier project review](review.md)

The study measures whether columns can be removed from an initial ranked support without a majority of panels detecting an excessive loss in any tested context. The predictors used for reporting held-out performance are retrained on the selected columns. All paths below are relative to the `MastersThesis` checkout on Windows.

## Implementation and evidence

| Component | Location |
|---|---|
| Search with a callable vote oracle | [context_vote.py](../src/minppi_mofs/context_vote.py) |
| Tree-based anchor, frozen masks and bootstrap panels | [context_vote_selector.py](../experiments/context_vote_selector.py) |
| Primary study and same-study controls, run serially on one GPU | [primary runner](../experiments/run_context_vote.py), [control runner](../experiments/run_context_vote_controls.py), [time-limit reconciliation](../experiments/reconcile_context_vote.py) |
| Stress test on all three datasets whose original width exceeds 100 | [runner and input checks](../experiments/run_context_vote_width.py), [per-arm results](evidence/width.csv) |
| Original ISOLET speaker-group holdout (secondary analysis) | [runner and group reconstruction](../experiments/run_context_vote_isolet_official.py), [four cold runs](evidence/speakers.csv), [independent checks](audit/isolet_official.py) |
| Seed-level data for the manuscript | [new study](evidence/primary.csv), [controls](evidence/controls.csv), [historical benchmarks](evidence/historical.csv), [source hashes](evidence/manifest.json) |
| Cross-predictor study | [frozen design](transfer/protocol.json), [linear-model adapter](transfer/run.py), [90 evaluations](transfer/per-seed.csv) |
| Secondary context-by-panel ablation on the same 15 linear-model splits | [design](transfer/factorial-design.md), [two additional arms](transfer/factorial.csv) |
| Independent test-score and search-trace checks | [audit script](audit/verify.py), [paired results](audit/paired-effects.csv), [feature-set overlap](audit/stability.csv) |
| Local irreducibility | [portable numerical certificates](evidence/certificates.json), [checker](certificates.py), [Lean development](formal/MinPI.lean), [Wolfram Language enumeration](formal/Ballots.wl) |

### Verify the historical six-dataset tables and local decisions

These commands use the checked-in data. They do not fit models. Lean 4.35 is pinned in the formalization directory. The PDF uses the Springer LNCS class and BibTeX style installed with TeX Live.

```powershell
.\thesis-python.ps1 thesis-paper/certificates.py
.\thesis-python.ps1 thesis-paper/make_tables.py
Push-Location thesis-paper/formal
lean MinPI.lean
Pop-Location
```

The certificate checker verifies the stated comparisons conditional on the supplied scores. Add `--sources` to match each disclosed score to the hashed estimator record and the selected support to its frozen record. The empirical certificate applies to the observed masking oracle. The kernel checks its discrete mathematical model; it does not execute the Python pipeline.

### How much did each part of voting contribute on the linear models?

Each cell below averages the same five held-out splits. The last column compares the selected feature identities with the full MinPI rule on all 15 splits. The two mixed settings were added after reviewing the initial experiment.

| Contexts tested | Panels per context | Breast cancer AUC ↑ | Wine AUC ↑ | Diabetes MAE ↓ | Identical supports |
|---|---|---:|---:|---:|---:|
| Empty only | Original panel | 0.9876 | 0.9962 | 44.49 | 9/15 |
| Empty only | Majority of four | 0.9875 | 0.9962 | 44.67 | 12/15 |
| All distinct | Original panel | 0.9876 | 0.9991 | 44.43 | 13/15 |
| All distinct | Majority of four (MinPI) | 0.9875 | 0.9991 | 44.54 | 15/15 |

### Run or resume the primary and control batches

Use the existing six-dataset, five-seed inputs under `results/thesis_benchmark/baselines-eight-preprocessed-five-seeds/`. The primary and control runners validate archived input hashes. Each logs to its output directory and prints the current dataset, seed, arm and elapsed time. Run the commands sequentially so that the timing arms have the GPU to themselves. The control study reads each MinPI width from the primary study's finalized group index.

```powershell
.\thesis-python.ps1 experiments/run_context_vote.py
.\thesis-python.ps1 experiments/reconcile_context_vote.py
.\thesis-python.ps1 experiments/run_context_vote_controls.py
.\thesis-python.ps1 experiments/reconcile_context_vote.py --controls --run results/thesis_benchmark/context-vote-controls-20261115-final
.\thesis-python.ps1 experiments/run_context_vote_width.py
.\thesis-python.ps1 experiments/run_context_vote_isolet_official.py --zip tmp/isolet-uci-official-20261115.zip
```

An additional terminal can follow either job. The primary batch writes its group retry decisions to `recovery.log`.

```powershell
Get-Content results/thesis_benchmark/context-vote-20261115/run.log -Tail 12 -Wait
```

The two batches use fixed paths pinned by their protocols. Running them again in this checkout resumes the same experiments. The reconciler selects a valid whole dataset–seed group and retains invalid attempts and their reason in `effective.json`. In a fresh checkout the original preprocessed partitions and input manifest must be present. The historical comparison grid is a separately timed campaign and its reported times are labeled accordingly.

The speaker-group runner reads the original [UCI ISOLET download](https://archive.ics.uci.edu/static/public/54/isolet.zip), SHA-256 `fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053`. Supply the ZIP with `--zip`. The runner checks row identities, class-label mapping and original file boundaries before creating inputs.

### Recompute empirical checks from raw records

```powershell
.\thesis-python.ps1 thesis-paper/audit/verify.py
.\thesis-python.ps1 thesis-paper/collect_evidence.py --with-width --with-speakers --archive packages/thesis-context-vote-evidence.zip
.\thesis-python.ps1 thesis-paper/certificates.py --export --sources
.\thesis-python.ps1 thesis-paper/make_tables.py
.\thesis-python.ps1 thesis-paper/transfer/run.py --output tmp/linear-transfer-replication.csv
.\thesis-python.ps1 thesis-paper/transfer/run.py --factorial --output tmp/linear-factorial-replication.csv
```

The [raw evidence archive](../packages/thesis-context-vote-evidence.zip) contains the effective primary, control, original-width and speaker-holdout records, watchdogs, test predictions and selection diagnostics. It uses paths relative to the thesis checkout. Historical comparison data and study inputs retain their original locations. The transfer study runs from scikit-learn's bundled datasets and uses separate train, ranking and test partitions.
