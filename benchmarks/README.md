# Benchmarks

Timings of `evals/run.py` with the fake predictor, so they measure our code
and not the model. Each run is 60 dev receipts. Add one row per measurement;
the median of three runs is the value to compare against.

Command: `uv run python -m evals.run --predictor fake --latency 2.0 --limit 60`

| Date       | Change                                   | Workers | Runs (s)            | Median (s) |
|------------|------------------------------------------|---------|---------------------|------------|
| 2026-11-03 | Baseline: batches of 64, load in main    | 8       | 20.38, 20.54, 20.35 | 20.38      |
