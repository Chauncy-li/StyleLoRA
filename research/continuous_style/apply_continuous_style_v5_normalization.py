"""Apply the frozen train V5 normalization model to validation/test data."""

from research.continuous_style.v5_cli import run


if __name__ == "__main__":
    run("apply_normalization")
