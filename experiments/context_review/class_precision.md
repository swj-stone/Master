# Four-bit score codes within the same predicted class

The experiment concerns a known finite dictionary of outer feature rows. It asks whether preserving the original classifier's predicted class explains the change in the number of compatible refit decisions when four-bit score codes are randomized. Training data and the fitted models are fixed. Both the reduced model and its reference move with the same reassignment of feature rows to labelled positions. Each proposal's decision uses a 3% AUC and accuracy tolerance.

The new control permutes donors among distinct input vectors with equal multiplicity and equal argmax of the unquantized frozen probability vector. Ties in argmax follow the stored class order. No outer labels or refitted predictions enter code generation. Each draw preserves the joint frequencies of predicted class, input multiplicity and quantized score vector. The observer knows the dictionary and the codebook and sees the code on each labelled position. The original predicted classes on those positions are not supplied separately. There are 24 fold-specific four-bit codes shared by more than one predicted class, covering 102 of 38,021 row occurrences.

Across the 42 folds, 37,736 of 37,760 distinct input types, accounting for 37,964 of 38,021 rows, belonged to strata with more than one type. Only 37,157 types (37,336 rows) had at least two possible four-bit codes in their stratum. Five assignments actually changed the codes of 14,817–14,926 types, or 14,940–15,054 rows (39.3–39.6%). Distinct donor types often carried the same four-bit code.

| Four-bit observation | Opposite decisions witnessed | Unresolved | Certified invariant |
|---|---:|---:|---:|
| Genuine score code alone | 338 | 9 | 357 |
| Class and multiplicity matched, code alone | 470–475 | 6–14 | 217–228 |
| Multiplicity matched, code alone | 648–656 | 0–1 | 48–56 |
| Genuine code and two point queries | 9 | 0 | 695 |
| Class and multiplicity matched, code and two point queries | 5–9 | 0 | 695–699 |
| Multiplicity matched, code and two point queries | 3–8 | 0 | 696–701 |

Each row summarizes 704 historical proposals. Ranges are extrema across the five fixed codebooks and are taken separately for each column. Every codebook has the genuine fold-level frequency of each code. Once the two point queries are appended, the joint frequencies of code and point queries can differ. These counts describe feasible worlds on the archived inputs. They are not out-of-sample pass probabilities.

In 12 of the 14 sources, the lower bound for ambiguity under every class-preserving draw exceeded the upper bound under the genuine code, counting every unresolved case as ambiguous for that upper bound. In Phoneme the bounds touched at 14. Wine quality red had 39 witnesses and no unresolved cases for each codebook. In eight sources the largest class-preserving upper bound across the five draws was also below the smallest multiplicity-only lower bound. These are paired descriptions of the fixed tasks; random codebooks do not give independent replications of the datasets.

The assigned codes and input multiplicities were independently reconstructed for all 210 fold and draw combinations. The independent check also enumerated all 24 possible assignments across six nontrivial point-plus-code orbits, including both an invariant decision and an opposite-decision witness. Every witness in the main tables is checked against the two refitted models' AUC and accuracy. Separate coordinate extrema can leave a joint decision unresolved.

Run from the thesis checkout with the original score-resolution and row-permutation evidence present.

```powershell
.\thesis-python.ps1 -m experiments.context_review.class_precision
.\thesis-python.ps1 -m experiments.context_review.class_precision_check
```

[Fixed design](../../configs/context_review/class_precision.json) · [Task counts](../../thesis-paper/evidence/class_precision/tasks.csv) · [Input mobility and codebook hashes](../../thesis-paper/evidence/class_precision/mobility.csv) · [Every proposal](../../thesis-paper/evidence/class_precision/proposals.csv)
