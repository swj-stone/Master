"""Reproduce manuscript Figure 3 from the 405-configuration experiment."""
from pathlib import Path

from research.fast_single.plot_figure3 import HERE, render


if __name__ == "__main__":
    figures = Path(__file__).resolve().parent / "figures"
    render(HERE / "evidence-extension/all-405.csv", (74, 331), figures / "matched-curves")
