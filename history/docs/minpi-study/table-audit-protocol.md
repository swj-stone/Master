# Table 2 numerical audit

Start from the 42 measured training panels in the original and revised arithmetic timings. Use the three published budgets (2%, 10%, 50%) in the revised run and the original 10% operating point for comparison. The reviewed PDF has SHA-256 `068159987c74ffd73970697c462fe8d5faf8e30c9eda2d82d86930c844d0b03f`.

Read the revised run's integer-nanosecond trial records. Within each source, seed and nominal budget, take the median of each method's five training-to-support calls and divide the minPI median by the comparator median. Calculate the median and linearly interpolated 25th and 75th percentiles of these panel ratios. Pool the 42 ratios for the comparison to unchecked floating-point pruning. Show the corresponding rows for both groups of 21 panels, defined by the common signed-word bound, including their exact-anchor and verified-proposal controls. Check the subgroup statistics against the archived summary before rounding the table cells. The quantiles describe variation across panels. The native paths for minPI and the exact anchor share the same calls.

Separately reaggregate the earlier 10% run from its archived millisecond trial records, excluding artificial Madelon, and verify identical panel identities. The original and revised timings were collected in separate runs. Their difference at 10% does not isolate the speed effect of the tighter bounds on silent columns.

At the study root, run `python -m experiments.minpi_study_table_audit --output fresh-table-audit --table fresh-table.tex` with `src` on the Python path. On the thesis workstation use the study's PowerShell Python wrapper. The two source trial files remain untouched. The table output is optional.
