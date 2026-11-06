"""What the evals share: where the repository and its results are, and how numbers are shown."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def fmt(value: float | None) -> str:
    """A metric for a table: '-' when the metric does not exist (zero denominator)."""
    return "  -  " if value is None else f"{value:.3f}"
