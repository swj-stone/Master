"""Run a time-bounded trial of the thesis benchmark on prepared datasets."""

from experiments.run_thesis_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(test_mode=True))
