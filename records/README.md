# Experiment records

These directories preserve the raw JSON outputs from the first protein
speedrun experiments. The JSON files are copied byte-for-byte from the run
directories. Model weights are not committed because each 150M checkpoint is
about 594 MB.

All runs used Docker on one node with eight H100 80GB GPUs. Times in
`result.json` are measured training time and exclude checkpoint evaluation.

| Record | Purpose | Steps | Tokens | Best Contact P@L | Final Contact P@L | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2026-07-29-tiny-gold-small-3k` | 0.895M-parameter small-data control | 3,000 | 259,497,668 | 0.04252 at step 400 | 0.03326 | 46.05 s |
| `2026-07-29-150m-throughput` | 148M-parameter batch and memory pilots | 2 each | up to 691,264 | pilot only | pilot only | up to 1.70 s after step 0 |
| `2026-07-29-150m-d0` | First representative 1,000-step run | 1,000 | 364,810,091 | 0.05205 at step 1,000 | 0.05205 | 449.38 s |
| `2026-07-29-150m-d1-1h` | One-hour constant-learning-rate control | 8,000 | 2,917,942,981 | 0.05205 at step 1,000 | 0.04434 | 3,564.89 s |

The tiny control and throughput pilots used corpus
`655432896642d8b0aae08f991904761daf769e909132ea9983fca2dcdbc5ef82`.
D0 and D1 used the one-million-sequence UniRef50 development corpus
`f4d76eabe12265179d8e4f82151440d3a9c2a154ff0873c6e667083f0cef899a`.

Each D0/D1 directory contains:

- `config.json`: the exact saved run configuration;
- `result.json`: the complete checkpoint-by-checkpoint history; and
- `final-evaluation-repeat.json`: the independent final-checkpoint re-score.

The older tiny control did not save a top-level config. Throughput pilots are
kept as raw result files because their purpose was only batch-size, memory, and
timing calibration. Smoke tests and evaluator-development repetitions are
intentionally omitted.
