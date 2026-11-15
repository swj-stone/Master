# Choosing a deletion with one candidate refit

The [R13 paper](../supplement/decision-study.pdf), [standalone R13 LaTeX source](../supplement/decision-study-source.zip) and [decision evidence archive](../../packages/thesis-decision-evidence.zip) accompany the [response to the supplied review](../final-review/response.md). The [previous release](../../packages/thesis-robustness-evidence.zip) contains the 43-page paper on which that review was based.

| Question | Study inputs, implementation and outputs |
|---|---|
| Can one candidate beat another in the same world? | [Joint observation and regret](../../experiments/context_review/joint_choice.md) · [All same-fold, equal-width pairs and explicit common worlds](joint_choice/) · [Lean decision lemmas](../formal/JointChoice.lean) |
| How much does a score add over random choice in the same top-vote tier? | [Definition of the lottery and tie rule](../../experiments/context_review/within_tier.md) · [Guarded returns, paired effects and visible ties](within_tier/) |
| How much of the four-bit result survives preserving predicted class? | [Recoding and exchangeability checks](../../experiments/context_review/class_precision.md) · [Five conditional controls and per-source counts](class_precision/) |
| How does the cheap refit compare on common slates? | [Timing boundaries and design-check example](../../experiments/context_review/budget_comparison.md) · [Raw clocks, returned supports and predictive metrics](budget_comparison/) |
| Did the rule chosen on other sources beat a fixed single point? | [Whole-source omission protocol](../../experiments/context_review/transfer_choice.md) · [All held-source decisions and three breach penalties](transfer_choice/) |

From the parent `Paper` folder in PowerShell, these commands check the published numbers and reproduce the wide-panel design ceiling, respectively. The first command extracts the complete release to a temporary directory, removes archived comparison tables and recreates them. It uses the existing shared Python environment. No raw predictor matrices are required for this replay. Fresh training and fresh clock readings require the original data. The archive includes the exact version used for live measurement under `budget_comparison/timed-source.py`.

```powershell
.\MastersThesis\thesis-python.ps1 thesis-paper/package_revision.py --verify-transport packages/thesis-decision-evidence.zip
.\MastersThesis\thesis-python.ps1 -m experiments.context_review.plan_comparison --original-wide --target-pp 5
```

The second command has no outer-label input. For new studies, pass a JSON file describing each policy's possible guarded returns and their probabilities instead of `--original-wide`. [Input example and assumptions](../../experiments/context_review/budget_comparison.md)

The frozen R13 archive includes its own packaging program. Use the replay command above to verify that archive. The current short paper, source and overlay package are indexed in the [manuscript guide](../README.md). The [artifact ledger](revision/artifacts.json) records the checksums.
