"""
Collect the figures the ICOMP 2026 paper needs into conference-paper/figures/.

Figures are produced by the experiment/demo pipeline under results/figures/
(gitignored scratch). This script copies the specific PDFs the paper
references into conference-paper/figures/, so the paper directory is
self-contained for building and packaging (``make pdf`` / ``make zip``).

Run:
    python conference-paper/sync_figures.py
    python conference-paper/sync_figures.py --source results/figures
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Figures referenced by paper/icomp2026_conference.tex (\includegraphics).
# All must be PDF (paper convention: never PNG/JPG).
REQUIRED_FIGURES: tuple[str, ...] = (
    "figure_1_minppi_trichotomy.pdf",
    "figure_2_prop1_detection.pdf",
    "figure_3_algorithm_b_vs_a.pdf",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    parser = argparse.ArgumentParser(
        description="Sync paper figures into conference-paper/figures/"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root / "results" / "figures",
        help="Directory holding the generated figure PDFs (default: results/figures)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=here / "figures",
        help="Destination directory (default: conference-paper/figures)",
    )
    return parser.parse_args()


def main() -> None:
    """Copy each required figure from source to dest; report and fail on gaps."""
    args = parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []
    for name in REQUIRED_FIGURES:
        src = args.source / name
        if src.is_file():
            shutil.copy2(src, args.dest / name)
            copied.append(name)
        else:
            missing.append(name)

    for name in copied:
        print(f"  copied  {name}")
    for name in missing:
        print(f"  MISSING {name}  (not found in {args.source})")

    print(f"\n{len(copied)}/{len(REQUIRED_FIGURES)} figures synced to {args.dest}")
    if missing:
        print(
            "Generate the missing figures first (see experiments/make_figures.py).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
