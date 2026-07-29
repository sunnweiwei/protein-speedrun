# Agent working rules

- Run all project Python, training, evaluation, and tests through `run.sh` or
  an equivalent Docker invocation. Do not install a host Python, Conda, CUDA,
  or PyTorch environment.
- Keep datasets, checkpoints, run artifacts, Docker state, and temporary
  training files under `/mnt/workspace/sunweiwei_google_com/`, never under
  Home or the system disk.
- Candidate changes belong in `candidate.py` and `config.json`. Do not weaken
  the corpus, checkpoint, metric, timing, or hardware contracts.
- The sole v0 pretraining quality metric is long-range Contact P@L. Training
  loss is diagnostic and must not become an additional leaderboard objective.
- `config.json` has no official numeric target yet. Do not invent or silently
  freeze one.
- Post-training and FoldBench are outside v0.
