# Protein Speedrun handover

Status snapshot: **2026-08-01**. This is the onboarding document for an Agent
continuing the project on another machine. Read this file before launching a
job. The shorter [`README.md`](README.md) is the executable interface; this
document records the research context, evidence, storage layout, and migration
boundary.

## The program in one page

The long-term goal is an autonomous protein-model research loop:

```text
Agent proposes a model / objective / optimizer / data change
    -> bounded pre-training
    -> one fixed representation score
    -> fixed folding post-training
    -> FoldBench development score
    -> Agent revises the design
    -> promising designs move to larger compute tiers and final evaluation
```

There are two standalone repositories with deliberately different jobs:

| Repository | Stage | Current fixed starting point | Primary metric |
| --- | --- | --- | --- |
| [`sunnweiwei/protein-speedrun`](https://github.com/sunnweiwei/protein-speedrun) | Protein sequence pre-training | Small ESM-2-style bidirectional Transformer trained from scratch | Long-range Contact P@L |
| [`sunnweiwei/protein-folding`](https://github.com/sunnweiwei/protein-folding) | Structure-model post-training | Official ESMC-6B embeddings plus ESMFold2-Fast | Native FoldBench metric for one frozen track |

They are **not connected yet**. Protein Folding v1 intentionally starts from
official ESMC-6B embeddings. A Speedrun checkpoint only guarantees
`model.encode(tokens, padding_mask) -> [B, L, D]`; no adapter has yet replaced
ESMC-6B inside ESMFold2, and no downstream folding score has been measured for
a Speedrun checkpoint.

The broader orchestration and scientific design live in
[`tangxiangru/ai4ai`](https://github.com/tangxiangru/ai4ai), especially
`docs/PLM_FOLDING_PROGRAM.md`. The two standalone repositories remain the
primary development locations. AI4AI is periodically synchronized; do not
assume its snapshot is newer than the standalone repository.

## Non-negotiable operating rules

1. Run every project Python command, test, training job, evaluator, and plot in
   Docker. Do not create a host Conda/venv/PyTorch/CUDA environment.
2. Put repositories, datasets, model caches, checkpoints, run artifacts,
   temporary training files, and Docker state on the large disk, never under
   Home.
3. Keep training and evaluation isolated. Evaluation time is not training
   time, and a candidate must not edit the evaluator or probe data.
4. Long-range Contact P@L is the sole v0 quality number. Training loss is a
   diagnostic, not a second leaderboard metric.
5. `target_contact_p_at_l` is still `null`. Do not invent a target before the
   positive-control and multi-seed calibration experiments are complete.
6. The current 1M-sequence corpus is a development corpus, not a
   FoldBench-leakage-audited training set.

## What the code freezes

- `candidate.py`: editable model, objective, and optimizer.
- `config.json` and `configs/*.json`: editable model/training recipes within a
  declared experiment.
- `speedrun.py`: data contract, timing, checkpointing, distributed evaluator,
  and score. Treat it as trusted infrastructure.
- `prepare_uniref.py`: deterministic UniRef download/extraction/materialization.
- `run.sh`: Docker-only single-GPU and 8-GPU entrypoint.
- `records/`: committed raw JSON from the reference experiments. Weights are
  intentionally not committed.

The checkpoint interface is:

```python
model = candidate.build_model(model_config)
embeddings = model.encode(tokens, padding_mask)  # [B, L, D]
```

Contact P@L uses residue pairs separated by at least 24 positions, a C-beta
distance below 8 Å (C-alpha fallback), the top `L` predicted pairs for each
protein, and a deterministic linear contact probe. The final score is the mean
per-protein precision across eligible probe proteins.

Training supports synchronous `torchrun`: every rank trains, global batch size
is divided over ranks, rank 0 writes checkpoints, and one isolated evaluator
worker per GPU shards both probe fitting and scoring. Deterministic CUDA,
cuBLAS, and NCCL settings are enabled so repeated same-seed jobs should produce
tensor-identical checkpoints.

## Verified state on the old machine

The snapshot below distinguishes committed evidence from large-disk state.

### Hardware and Docker

- 8 × NVIDIA H100 80GB HBM3.
- Docker socket:
  `/mnt/workspace/sunweiwei_google_com/ai4ai-docker.sock`.
- Docker root:
  `/mnt/workspace/sunweiwei_google_com/ai4ai-docker-data`.
- Built image: `protein-speedrun:dev`, approximately 9.78GB.
- Repository was clean at standalone revision
  `f08a7a8c4963392cca7f4b66e8d51b9d0f7fd51a` before this handover was added.

Docker image IDs are machine-local evidence, not portable protocol IDs. Rebuild
the image from the committed Dockerfile on the new machine.

### Data inventory

| Artifact | Old-machine path | Size | Identity / role |
| --- | --- | ---: | --- |
| Tiny real fixture | `protein-pretrain-speedrun-data/gold_small_v1/` | 868KB | `corpus.npz` SHA-256 `655432896642d8b0aae08f991904761daf769e909132ea9983fca2dcdbc5ef82` |
| 1M development corpus | `protein-pretrain-speedrun-data/uniref50_2021_04_d0/` | 265MB | `corpus.json` content SHA-256 `f4d76eabe12265179d8e4f82151440d3a9c2a154ff0873c6e667083f0cef899a` |
| UniRef source/intermediates | `protein-pretrain-speedrun-data/sources/` | about 205GB | Rebuildable; do not transfer by default |
| All run artifacts | `protein-pretrain-speedrun-runs/` | about 23GB | Includes checkpoints, results, smokes, and evaluator calibration |
| D0 run | `protein-pretrain-speedrun-runs/scale-v1/esm2-150m-d0-seed42/` | 6.1GB | 11 checkpoints including step 0 |
| D1 one-hour run | `protein-pretrain-speedrun-runs/scale-v1/esm2-150m-d1-1h-seed42/` | 9.4GB | 17 checkpoints including step 0 |

The tiny fixture contains 1,533 training sequences, 170 probe-training
proteins, and 185 probe-evaluation proteins with no sequence-cluster overlap.
It is an integration fixture only and is too small for scaling conclusions.
Its materializer is not in this standalone repository, so transfer
`gold_small_v1/` even if all large UniRef sources are rebuilt.

The development corpus is a deterministic uniform-reservoir sample of
1,000,000 UniRef50 2021_04 representative sequences:

- 248,385,020 residues;
- canonical amino acids only;
- lengths 32–1,024;
- 20 shards of 50,000 sequences;
- sample seed `20260729`;
- exact probe matches removed (zero were found);
- probe SHA-256
  `655432896642d8b0aae08f991904761daf769e909132ea9983fca2dcdbc5ef82`.

Exact-match removal is not sufficient for a scientific leaderboard. Before
calling this corpus leakage-audited, run an MMseqs2 homology exclusion against
both the contact probe and every frozen FoldBench target.

The rebuildable source files on the old machine are:

| File | Bytes | Verification |
| --- | ---: | --- |
| `uniref2021_04.tar.gz` | 169,188,630,923 | MD5 `444f0a7062a65a988ba1d5949d1d6419` |
| `uniref50.xml.gz` | 25,107,210,941 | SHA-256 `ebe8563d804333bec9f8e68fa103277946b648d68146f97067e07bc42b2ef1e6` |
| `uniref50.tar` | 25,107,230,720 | Redundant intermediate; not required by `prepare_uniref.py` |

### Reference experiments

All committed rows below used Docker on one node with eight H100 80GB GPUs.
Times are measured training time and exclude checkpoint evaluation.

| Run | Model / purpose | Steps | Tokens | Best Contact P@L | Final Contact P@L | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiny control | 0.895M params, tiny fixture | 3,000 | 259,497,668 | 0.04252 at step 400 | 0.03326 | 46.05s |
| throughput pilots | 148.4M params, batches up to 2,048 | 2 | up to 691,264 | calibration only | calibration only | up to 1.70s after step 0 |
| D0 | 148,376,960 params, seed 42 | 1,000 | 364,810,091 | 0.05205 at step 1,000 | 0.05205 | 449.38s |
| D1 | Same model and seed, constant-LR control | 8,000 | 2,917,942,981 | 0.05205 at step 1,000 | 0.04434 | 3,564.89s |

D0 used 30 layers, width 640, 20 heads, FFN multiplier 4, MLM masking 0.15,
BF16, global batch 2,048, crop length 256, AdamW at `4e-4`, 50 warmup
steps, and no decay after warmup. It moved Contact P@L from 0.042524 at
initialization to 0.052051. A fresh final-checkpoint evaluator reproduced the
same score exactly.

D1 reproduced the D0 trajectory exactly through step 1,000 and then declined
to 0.044337. This is **not evidence against scaling laws**: token budget was
increased while holding an unsuitable constant `4e-4` learning rate. The
result says the longer recipe needs decay and that Contact P@L may be operating
near its signal floor at this training scale.

Canonical raw results are committed under [`records/`](records/). Do not infer
numbers from console logs when a `result.json` exists.

## Migration to a new machine

Choose one large-disk root. The old-machine value was:

```bash
export PS_WORKSPACE_ROOT=/mnt/workspace/sunweiwei_google_com
export DOCKER_HOST=unix://$PS_WORKSPACE_ROOT/ai4ai-docker.sock
```

Clone on that disk:

```bash
git clone https://github.com/sunnweiwei/protein-speedrun.git \
  "$PS_WORKSPACE_ROOT/protein-speedrun"
mkdir -p \
  "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data" \
  "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-runs"
```

`run.sh` currently hardcodes
`/mnt/workspace/sunweiwei_google_com`. If the new large-disk path differs,
change its `workspace_root=` line in a reviewable commit before running, or
mount the large disk at the same absolute path. Do not point it at Home.

Configure the new Docker daemon so its `DockerRootDir` is also under
`$PS_WORKSPACE_ROOT`, then verify it before any image build:

```bash
sudo docker --host "$DOCKER_HOST" info --format '{{.DockerRootDir}}'
```

### What to transfer

For the minimum reproducible continuation, transfer:

1. `protein-pretrain-speedrun-data/gold_small_v1/`;
2. `protein-pretrain-speedrun-data/uniref50_2021_04_d0/`;
3. the D0 and D1 run directories if later work needs their actual weights;
4. any uncommitted future run directory and its exact config.

The committed `records/` are enough to inspect historical scores, but not to
load historical weights. Transfer all 23GB of runs only if storage and network
make that cheaper than selecting the two reference directories above.

Do **not** copy the 205GB `sources/` tree by default. Rebuild it deterministically
inside Docker. The only non-rebuildable prerequisite is the small
`gold_small_v1/corpus.npz` probe/fixture.

After transfer, verify at least:

```bash
sha256sum \
  "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz" \
  "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data/uniref50_2021_04_d0/probe.npz"
sed -n '1,220p' \
  "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data/uniref50_2021_04_d0/corpus.json"
```

Both NPZ files must hash to
`655432896642d8b0aae08f991904761daf769e909132ea9983fca2dcdbc5ef82`,
and `corpus.json` must record content SHA-256
`f4d76eabe12265179d8e4f82151440d3a9c2a154ff0873c6e667083f0cef899a`.

### Rebuild the large UniRef data

Build once:

```bash
cd "$PS_WORKSPACE_ROOT/protein-speedrun"
sudo docker --host "$DOCKER_HOST" build -t protein-speedrun:dev .
```

Download with network access, writing only to the mounted large disk:

```bash
sudo docker --host "$DOCKER_HOST" run --rm --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PS_WORKSPACE_ROOT/protein-speedrun:/workspace:ro" \
  -v "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data:/data" \
  -w /workspace protein-speedrun:dev \
  python prepare_uniref.py download \
    --output /data/sources/uniref2021_04/uniref2021_04.tar.gz
```

Extract and materialize offline:

```bash
sudo docker --host "$DOCKER_HOST" run --rm --network none \
  --user "$(id -u):$(id -g)" \
  -v "$PS_WORKSPACE_ROOT/protein-speedrun:/workspace:ro" \
  -v "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data:/data" \
  -w /workspace protein-speedrun:dev \
  python prepare_uniref.py extract-uniref50 \
    --archive /data/sources/uniref2021_04/uniref2021_04.tar.gz \
    --output /data/sources/uniref2021_04/uniref50.xml.gz

sudo docker --host "$DOCKER_HOST" run --rm --network none \
  --user "$(id -u):$(id -g)" \
  -v "$PS_WORKSPACE_ROOT/protein-speedrun:/workspace:ro" \
  -v "$PS_WORKSPACE_ROOT/protein-pretrain-speedrun-data:/data" \
  -w /workspace protein-speedrun:dev \
  python prepare_uniref.py materialize \
    --uniref-xml /data/sources/uniref2021_04/uniref50.xml.gz \
    --probe-corpus /data/gold_small_v1/corpus.npz \
    --output /data/uniref50_2021_04_d0 \
    --sequences 1000000 --shard-sequences 50000 \
    --seed 20260729 --min-length 32 --max-length 1024
```

`materialize` refuses to overwrite an existing output. Move a suspect partial
directory aside for audit rather than deleting or overwriting it.

## First-day verification

From the standalone repository on the large disk:

1. Confirm `git status --short --branch` is clean and read `AGENTS.md`.
2. Confirm Docker root and data/run roots are on the large disk.
3. Confirm all eight GPUs are visible to Docker.
4. Run `./run.sh smoke`; it trains two steps and repeats the final evaluation.
5. Run `./run.sh run` on `gold_small_v1` only as an integration check.
6. Do not launch D0 until the materialized corpus identity matches this file.
7. Launch the frozen reference with `./run.sh ddp`, using a new output path.
8. Compare `result.json` with [`records/2026-07-29-150m-d0/`](records/2026-07-29-150m-d0/).

`./run.sh ddp` is the intended 8-GPU single-node path. Do not run eight
independent fits when the experiment calls for one distributed job.

## Most important next experiments

The current metric has only moved 0.0095 above the untrained model, so the
instrument must be calibrated before architecture search.

1. **Positive control:** score a public converged ESM2-150M checkpoint with the
   same Contact P@L evaluator. It must land clearly above 0.052; otherwise the
   probe or checkpoint adapter is suspect.
2. **Noise floor:** run at least three seeds for the chosen default recipe.
3. **Long-budget repair:** add a defensible learning-rate decay, then compare
   against D1 at matched tokens/FLOPs. Do not silently relabel D1 as a baseline
   recipe.
4. **Proxy validity:** test whether a cheap budget preserves candidate ranking
   at a 10× budget; a working proposal is Spearman `rho >= 0.7`.
5. **Leakage audit:** cluster the training corpus against the probe and frozen
   FoldBench targets before making scientific claims.
6. **PLM-to-folding bridge:** hold the folding trunk fixed, substitute a smaller
   public protein LM, and measure the lDDT change. This is the first experiment
   that connects pre-training quality to the actual downstream objective.

Until these gates pass, this repository is a development speedrun kernel, not
an official protein-model leaderboard.

## Related completed work outside this repository

These artifacts explain later decisions but are not Speedrun scores:

- A full Docker-only OpenDDE reproduction on FoldBench protein-protein produced
  200 successes over 276 officially scored interfaces (`0.72464`), or
  `0.71685` under a strict 279-interface denominator, versus the paper's
  reported `0.7689`. An audited scorer sensitivity gives `0.72401` strict.
- A cost-aware FoldBench PPI validation coreset selected 183 assemblies / 220
  interfaces at 30% of the five-seed full compute. Across six held-out model
  families, nested leave-one-family-out error was 0.61 percentage points MAE,
  1.15 points P90, and 1.72 points worst case. Those are empirical errors, not
  a guarantee for a new architecture.
- Those old-machine artifacts live under
  `/mnt/workspace/sunweiwei_google_com/opendde-foldbench-repro` and
  `/mnt/workspace/sunweiwei_google_com/foldbench-validation`. Their protocols
  and results belong with the Folding/FoldBench stage, not this metric.

## Evidence vocabulary

Use these labels in future records:

- **verified:** the command ran and the artifact/result exists;
- **recorded:** a repository or paper states the number, but it was not rerun in
  this environment;
- **predicted/planned:** inferred from code or proposed, not measured.

Never turn an import smoke test, official example evaluation, or planned data
recipe into a model-quality claim.
