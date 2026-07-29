# Protein Speedrun

A minimal modded-nanoGPT-style loop for protein sequence pretraining.

The experiment asks one question:

> How much single-H100 training time does a sequence model need to reach a
> fixed long-range Contact P@L?

The first version intentionally contains only one editable candidate, one
config, one trusted runner, and one score.

## Files

- `candidate.py`: model, optimizer, and pretraining objective. Edit this file.
- `config.json`: model size, training budget, and optional target.
- `speedrun.py`: fixed data contract, timing, checkpointing, and Contact P@L.
- `run.sh`: Docker-only entrypoint using the large workspace disk.

The checkpoint contract is:

```python
model = candidate.build_model(model_config)
embeddings = model.encode(tokens, padding_mask)  # [B, L, D]
```

Each checkpoint includes `candidate.py`, `model.json`, `weights.pt`, hashes,
and an external evaluation result. Evaluation runs in a fresh Python process
and is excluded from training time.

## Metric

The only quality number is long-range Contact P@L:

- long-range separation: at least 24 residues;
- contact: C-beta distance below 8 Å, with C-alpha fallback in the corpus;
- score: per-protein precision among the top `L` predicted contacts, averaged
  across eligible proteins; and
- probe: the same deterministic linear contact probe for every checkpoint.

Training loss is diagnostic only. MLM and causal objectives can be compared
because both must expose the same residue embeddings.

`target_contact_p_at_l` remains `null` in the initial config. A numeric target
should only be frozen after longer reference runs across multiple seeds.

## Run

All project Python runs inside Docker. Docker state, data, and checkpoints must
remain under `/mnt/workspace/sunweiwei_google_com/`.

Synthetic end-to-end smoke:

```bash
./run.sh smoke
```

The smoke trains for two steps, evaluates step 0/1/2, reconstructs the final
checkpoint, repeats its evaluation, and fails if the repeated score changes.

Run the same tiny setup on the existing real `gold_small/v1` corpus:

```bash
./run.sh run
```

The default real corpus is:

```text
/mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz
```

It contains 1,533 train sequences, 170 probe-training proteins, and 185
probe-evaluation proteins with no sequence-cluster overlap. Its SHA-256 is:

```text
655432896642d8b0aae08f991904761daf769e909132ea9983fca2dcdbc5ef82
```

Choose another physical GPU or seed:

```bash
PROTEIN_SPEEDRUN_GPU=1 ./run.sh run \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/seed-66 \
  66
```

## Scope

This is an executable research kernel, not yet a competition platform.
Multi-seed target calibration, official aggregation, post-training folding,
FoldBench, distributed training, and agent orchestration are deliberately
deferred until this small loop is useful.
