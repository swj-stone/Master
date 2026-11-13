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
