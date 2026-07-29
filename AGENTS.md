# Agent working rules

- Run all project Python, training, evaluation, and tests through `run.sh` or
  an equivalent Docker invocation. Do not install a host Python, Conda, CUDA,
  or PyTorch environment.
- Keep datasets, checkpoints, run artifacts, Docker state, and temporary
  training files under `/mnt/workspace/sunweiwei_google_com/`, never under
  Home or the system disk.
- Candidate changes belong in `submission/model.py`, `submission/train.py`,
  and their configs. Do not weaken the corpus, checkpoint, metric, timing,
  hardware, seed, or target-confirmation contracts.
- The sole v0 pretraining quality metric is long-range Contact P@L. Training
  loss is diagnostic and must not become an additional leaderboard objective.
- `protocol.v0.json` has no official numeric target until the five reference
  calibration seeds finish. Do not invent or silently freeze a target.
- Post-training and FoldBench are outside v0.
