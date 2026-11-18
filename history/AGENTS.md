# Master's thesis workspace

- Work in this checkout on `thesis/masters`; use its own files and Git index.
- The sibling `../Master` checkout on `main` is a frozen final-version reference.
  Do not modify its files or branch, or merge thesis work into it, unless the user
  explicitly asks. Shared Git worktree bookkeeping is expected.
- The reference baseline is commit `9086ddc071c3e173f5ba3b1a0b61b123677b2605`.
- Use `./thesis-python.ps1` for Python commands. It uses the existing `../venv`
  dependencies but imports this checkout's `src` first. Plain venv Python still
  has an editable installation pointing to the original `Master/src`.
- Do not run `pip install -e .` or reinstall `requirements.txt`: that would
  retarget the shared environment. Changes to shared dependency versions affect
  both projects and require explicit user direction.
- `data/`, `results/`, and `packages/` were copied as independent files. They
  remain ignored by Git. Write new experiments under new run IDs in this
  checkout; never use output paths inside `../Master`.
- Preserve the copied conference paper as reference. Thesis writing and future
  restructuring happen here, under thesis-specific paths as needed.
- Push only the thesis branch to the existing `origin`, for example
  `git push -u origin thesis/masters`. Never force-push or push to `main` as part
  of thesis work.

## PI versions

- Use the annotated tag `thesis/piv1-frozen` as the PIv1 baseline. Its method,
  settings, and evidence are indexed in [the archive](docs/thesis/archives/piv1/README.md).
- Develop subsequent PI variants separately from PIv1. Retain voting and the
  minimum across coalition-size strata. Prioritize end-to-end efficiency and
  reuse with different downstream selectors. Measure each backend on its own
  and with the PI prefilter, including the cost of prefiltering.

## minPI manuscript revisions

Use descriptive headings in the manuscript, including run-in paragraph titles. For figures and tables, choose the comparisons that readers need, preserve the data and its units, and check legibility in the single ICOMP main PDF at publication size.

Whenever a revision of `paper/minpi/main.tex` or its included manuscript,
figures or evidence is finished, run an independent internal referee round
before delivering the manuscript. Read `paper/minpi/README.md` and the user's
latest comments. Build the candidate with
`./thesis-python.ps1 -m experiments.review_minpi_study --prepare`. Give its
PDF, source, evidence and hashes to a referee agent separate from the writer.
Require an explicit scientific rationale for experimental parameters and direct sensitivity checks on choices that drive the conclusions. Have the referee challenge the strongest practical comparator, inputs to the guarantee, all boundary and tie cases, negative results and the interpretation in Discussion. Use an independent calculation or counterexample where possible. Check the current paper against its single declared contribution. Ask the referee to test the main claims, inspect the rendered pages and record
concrete objections with their locations in `docs/minpi-study/current-review.md`.
For a substantive defect the referee records `decision: revise`. Route the
defect to the experimenter or writer, fix it, rebuild and repeat without
waiting for another user instruction. Where evidence limits a claim, narrow it
or report the unresolved limitation. An external-review recommendation is a
scientific judgment and carries no prediction of acceptance.

The referee records `decision: ready_for_external_review`, `inputs_sha256`,
`evidence_sha256` and `pdf_sha256` in YAML front matter when the audited claims are supportable.
Keep the review history and the disposition of returned issues in that report.
Validate the decision using `./thesis-python.ps1 -m experiments.review_minpi_study`
and update the local replication bundle after review.
