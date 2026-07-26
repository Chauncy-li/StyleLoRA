"""Evaluate post-training V6 rho sweeps from generated percentile-axis JSONL."""

from research.continuous_style.v6_cli import run


if __name__ == "__main__":
    run("sweep_evaluation")
