# Benchmarks

Timings of `evals/run.py` with the fake predictor, so they measure our code
and not the model. Each run is 60 dev receipts. Add one row per measurement;
the median of three runs is the value to compare against.

Command: `uv run python -m evals.run --predictor fake --latency 2.0 --limit 60`

| Date       | Change                                   | Workers | Runs (s)            | Median (s) |
|------------|------------------------------------------|---------|---------------------|------------|
| 2026-11-03 | Baseline: batches of 64, load in main    | 8       | 20.38, 20.54, 20.35 | 20.38      |
| 2026-11-03 | Load and decode inside each worker       | 8       | 19.09, 18.72, 18.68 | 18.72      |
| 2026-11-03 | Same code, 4 workers                     | 4       | 35.87, 35.72, 36.16 | 35.87      |
| 2026-11-03 | Same code, 16 workers                    | 16      | 13.52, 14.13, 13.70 | 13.70      |

Chosen default: 16 workers. Lite model, 60 dev receipts, 22.8 s wall, no 429 errors, mean call 2.8 s. Not yet tested above 60 receipts or with 32 workers.
