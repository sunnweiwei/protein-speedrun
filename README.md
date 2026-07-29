# Protein Speedrun

A minimal modded-nanoGPT-style loop for protein sequence pretraining.

The experiment asks one question:

> How much H100 training time does a sequence model need to reach a fixed
> long-range Contact P@L?

The first version intentionally contains only one editable candidate, one
config, one trusted runner, and one score.

## Files

- `candidate.py`: model, optimizer, and pretraining objective. Edit this file.
- `config.json`: model size, training budget, and optional target.
- `speedrun.py`: fixed data contract, timing, checkpointing, and Contact P@L.
- `plot_result.py`: dependency-light budget-curve renderer.
- `records/`: raw JSON outputs and a short experiment index.
- `run.sh`: Docker-only entrypoint using the large workspace disk.

The checkpoint contract is:

```python
model = candidate.build_model(model_config)
embeddings = model.encode(tokens, padding_mask)  # [B, L, D]
```

Each checkpoint includes `candidate.py`, `model.json`, `weights.pt`, hashes,
and an external evaluation result. Single-GPU evaluation runs in a fresh
Python process. DDP evaluation uses one persistent, isolated worker process
per GPU; workers reconstruct the checkpoint and shard both probe fitting and
scoring without entering the training processes. Evaluation is excluded from
training time.

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

That 1,533-sequence corpus is only an integration fixture. It is too small
for model or scaling conclusions. The first development pretraining corpus is
a deterministic one-million-sequence sample of UniRef50 2021_04
representatives. `prepare_uniref.py` downloads the historical release with
resume plus MD5 verification, then materializes memory-mapped token and offset
shards. The sharded loader keeps the large training split out of each DDP
rank's Python heap. Materialization removes exact probe matches. Before this
development corpus can become an official leaderboard corpus, an MMseqs
cluster audit must also exclude homologs of both the contact probe and the
frozen FoldBench targets; an exact-match-only corpus is not called
leakage-audited.

The first representative scale is `configs/esm2-150m-pilot.json`: 30 layers,
width 640, 20 attention heads, and FFN width 2560. It contains 148,376,960
parameters, matching the public ESM-2 150M size class while retaining this
repository's deliberately simple Transformer implementation. The pilot config
is only for memory, throughput, and contract checks; it is not a training
budget. `configs/esm2-150m-d0.json` is the first development budget: BF16,
global batch 2048 across eight H100s, 1,000 updates, 50 warmup updates, and
checkpoints every 100 updates. `configs/esm2-150m-d1-1h.json` extends the same
constant-learning-rate setup to 8,000 updates, approximately one hour of
measured training, with checkpoints every 500 updates.

Choose another physical GPU or seed:

```bash
PROTEIN_SPEEDRUN_GPU=1 ./run.sh run \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/seed-66 \
  66
```

The trusted runner also supports synchronous DDP. `batch_size` is the global
batch and must be divisible by `WORLD_SIZE`; `tokens_seen` is reduced across
all ranks. Every rank trains, only global rank 0 writes checkpoints, and all
isolated evaluator workers participate in the fixed Contact P@L evaluation:

```bash
torchrun --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29500 \
  speedrun.py run \
  --config config.json \
  --corpus /data/gold_small_v1/corpus.npz \
  --output /runs/ddp-example
```

Launch that command inside the same Docker image with all eight GPUs visible.
Standard multi-node `torchrun` environment variables are supported; all nodes
must see the same corpus and output filesystem.

After the D0 corpus exists, the equivalent single-node launch is:

```bash
./run.sh ddp
```

Render the result as Contact P@L versus millions of training tokens:

```bash
python plot_result.py \
  --result /runs/esm2-150m-d0/RUN_ID/result.json \
  --output /runs/esm2-150m-d0/RUN_ID/budget-curve.png
```

Training enables deterministic CUDA, cuBLAS, and NCCL paths. Repeated
same-seed jobs must produce tensor-identical checkpoints. This costs some
throughput, but prevents small distributed floating-point differences from
turning into different budget curves.

## D0 reference run

Seed 42 on eight H100 80GB GPUs completed 1,000 updates over corpus
`f4d76eabe12265179d8e4f82151440d3a9c2a154ff0873c6e667083f0cef899a`.
It processed 364,810,091 tokens in 449.38 seconds of measured training time.
Contact P@L increased from 0.04252 at initialization to 0.05205, a 22.4%
relative gain. The final checkpoint scored identically in a fresh standalone
evaluator process. This is a development calibration result, not a
leakage-audited FoldBench claim.

The one-hour D1 constant-learning-rate control processed 2,917,942,981 tokens
in 3,564.89 seconds. It reproduced the D0 trajectory exactly through step
1,000, where Contact P@L peaked at 0.05205, then declined to 0.04434 by step
8,000. Its purpose is to record that extending the `4e-4` constant learning
rate does not produce monotonic scaling; longer-budget candidates should add a
decay schedule rather than use D1 as a recommended recipe.

## Scope

This is an executable research kernel, not yet a competition platform.
Multi-seed target calibration, official aggregation, post-training folding,
FoldBench, and agent orchestration are deliberately deferred until this small
loop is useful.
