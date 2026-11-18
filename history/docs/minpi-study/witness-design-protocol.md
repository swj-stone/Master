# Sensitivity to training-witness design

This retrospective design diagnostic fixes the first archived seed (1129) for Gisette, gas and superconductivity. It uses the matched training partition for Gisette, the first seven gas batches and the original composition-group training partition for materials. We access only training features and labels when computing the diagnostics. No downstream learner or test predictions enter this experiment.

Use the published witness builder as the reference: a 256-observation cap, 0.8 weight on supervised geometry, paired nearest and fourth-nearest hit/miss comparisons, and five reference-quantile response bands for regression. Change one factor at a time. Test caps 128 and 512, supervised weights 0 and 1, and neighbour-rank pairs (1,1) and (1,8) for each source. Test three and seven bands for materials only. A repeated rank contributes two identical comparisons. Truncate each rank separately at the number of available reference neighbours, keep the original clipping and tie handling, and use the same seed for each configuration. When the cap changes, the selection of witnesses can also change.

Use the default coalition sizes, input-index tie rule and budget ceiling of ten per cent. For each configuration retain the sorted support, ternary matrix digest, number of rows, and intersection and union with the default support. Report the Jaccard coefficient as their ratio. Before interpreting an overlap, require the default reconstructed ballot to match the production builder entry by entry, its digest to match the earlier order diagnostic, and every certified support to agree with exhaustive integer scoring of the same ballot. These are three single-seed sensitivity profiles. They do not estimate accuracy, identify an optimal setting or attribute the effect of a changed row cap to sampling and row count separately.

Run with one numerical-library thread:

```powershell
.\thesis-python.ps1 -m experiments.minpi_study_witness_design --output results/minpi-study/witness-design
```

