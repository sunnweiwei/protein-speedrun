# Protein Pretraining Speedrun

This track asks one question:

> How quickly can a protein sequence model reach a fixed, transferable
> structural-representation quality?

It follows the format of
[`modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt), but replaces
objective-specific language-model loss with one objective-neutral score:

`long_range_contact_p_at_l`

The score is the macro mean of per-protein precision among the top `L`
predicted long-range contacts. A long-range pair has sequence separation at
least 24 residues and is a contact when its C-beta distance is below 8
Angstrom (C-alpha is used when C-beta is unavailable).

Training loss is an implementation detail. A submission may use MLM, causal
next-token prediction, span corruption, a hybrid objective, or a new
objective. The external evaluator only requires full-sequence residue
embeddings through the checkpoint contract.

## One number per stage

The pretraining leaderboard exposes exactly one number:

`median_confirmed_seconds_to_target`

A run reaches the target only after two consecutive scheduled checkpoints
score at or above the frozen Contact P@L target. The confirmation time is the
training time at the second checkpoint. An official candidate uses seeds
`42, 66, 101, 2024, 8888`; at least four must confirm, and the record time is
the median confirmation time among successful seeds.

The trusted schedule always evaluates step 0 and then every configured
`checkpoint_every` steps, plus the final step when needed. A candidate cannot
add free evaluations, skip a scheduled evaluation, or continue training after
the callback confirms the target.

Evaluation time is excluded from training time. Compilation and warmup policy
will be frozen before the first official record. The v0 smoke runner measures
only model-training sections and records evaluation wall time separately.
Each checkpoint is evaluated in a fresh Python process, so evaluator CUDA
initialization cannot hide candidate initialization or kernel warmup.
Every run sees exactly one NVIDIA H100 80GB GPU. Different official seeds may
run concurrently on different physical GPUs, but each seed is still an
independent one-GPU record.
The PyTorch base image is pinned by registry digest; PyTorch, CUDA, and cuDNN
versions are checked against `protocol.v0.json` before timing begins.

The future post-training leaderboard is deliberately outside v0. It will
freeze the encoder, train the same folding/diffusion model with the same
structural-data and compute budget, and report one FoldBench proxy score.

## Fixed data

The initial real-data profile is derived from the existing immutable
`gold_small/v1` manifest:

- 3,725 high-quality experimental PINDER complexes;
- structural cutoff `2021-09-30`;
- interface and sequence-cluster decontamination against FoldBench;
- X-ray resolution at most 2.5 A or cryo-EM resolution at most 3.0 A; and
- strict interface-completeness and biological-interface filters.

Sequence clusters are deterministically assigned to:

- pretraining train;
- contact-probe train; or
- contact-probe evaluation.

No sequence cluster appears in more than one split. The materialized speedrun
corpus contains packed amino-acid tokens and fixed C-beta coordinates in NPZ;
raw structures remain in the large-disk operator data store.
The current immutable materialization contains 1,533 train sequences, 170
probe-training proteins, and 185 probe-evaluation proteins; 137 of the latter
meet the frozen minimum-contact eligibility rule.

`gold_small/v1` was originally selected for PPI structure training rather
than billion-sequence language-model pretraining. It is intentionally a
small, high-quality v0 speedrun corpus, not a claim that this is the final
large-scale ESMC corpus.

## Submission contract

A candidate owns `submission/model.py` and `submission/train.py`.

`submission/model.py` must export:

```python
def build_model(model_config: dict) -> torch.nn.Module: ...
```

The returned model must export:

```python
def encode(
    tokens: torch.Tensor,          # [B, L], amino-acid vocabulary
    padding_mask: torch.Tensor,    # [B, L], True for real residues
) -> torch.Tensor:                 # [B, L, D]
    ...
```

`model.py` must be self-contained apart from the Python/PyTorch runtime,
because its exact bytes are copied into the checkpoint. Editing the candidate
workspace later therefore cannot change how an old checkpoint is reconstructed.

Every checkpoint is a directory containing:

- `checkpoint.json`: immutable metadata and accounting;
- `model_config.json`: constructor input; and
- `model.py`: the immutable `build_model()` and `encode()` implementation
  snapshot; and
- `weights.pt`: a plain `state_dict`, loaded with `weights_only=True`.

The evaluator rejects missing fields, non-finite accounting, data-manifest
mismatches, and embedding outputs that violate shape or finiteness rules.

## Docker-only usage

All commands run project Python inside Docker. The host only invokes Docker
and edits files. The isolated Docker daemon stores both Docker and containerd
state on `/mnt/workspace`.

```bash
cd /mnt/workspace/sunweiwei_google_com/protein-speedrun

./run.sh smoke
```

Set `PROTEIN_SPEEDRUN_GPU=1` (or another single index) to choose which physical
GPU is exposed to the container.

The smoke command:

1. builds the image using the large-disk Docker daemon;
2. creates a deterministic synthetic corpus under the large disk;
3. trains the tiny reference encoder;
4. externally evaluates every checkpoint;
5. tests two-consecutive-checkpoint target confirmation; and
6. runs the unit tests in the same image.

Prepare the real `gold_small/v1` corpus with the already available Protenix
runtime image:

```bash
./run.sh prepare-gold-small
```

Run one calibration seed:

```bash
./run.sh run \
  configs/baseline_50m.json \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/baseline_50m/seed-42
```

Pass a fifth argument to override the config seed. This lets the five official
seeds run concurrently on five physical GPUs without duplicating configs:

```bash
PROTEIN_SPEEDRUN_GPU=1 ./run.sh run \
  configs/baseline_50m.json \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/baseline_50m/seed-66 \
  66
```

The real config has no official target until baseline calibration is
complete. Calibration freezes a target that is:

1. comfortably above the untrained score;
2. reached by all reference seeds;
3. below the reference final score by a predeclared safety margin; and
4. stable under repeated evaluator runs.

After all five calibration seeds finish, ask the tool for the highest
quantized target that satisfies predeclared margins and two-checkpoint
confirmation for every seed:

```bash
./run.sh calibrate \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/baseline_50m \
  0.002 \
  0.002
```

The output is explicitly a recommendation. It never edits
`protocol.v0.json`; freezing the target remains a reviewed decision.

Verify that repeated external evaluation of an unchanged checkpoint has no
measurable jitter:

```bash
./run.sh verify \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-runs/baseline_50m/seed-42/checkpoints/step-00000500 \
  /mnt/workspace/sunweiwei_google_com/protein-pretrain-speedrun-data/gold_small_v1/corpus.npz
```

## Files

- `protocol.v0.json`: frozen metric, split, and stability semantics.
- `speedrun/prepare_corpus.py`: immutable real/synthetic corpus builder.
- `speedrun/checkpoint.py`: checkpoint contract.
- `speedrun/contact_eval.py`: the sole pretraining metric.
- `speedrun/evaluate_once.py`: process-isolated trusted checkpoint evaluation.
- `speedrun/evaluate.py`: repeated evaluator-jitter check.
- `speedrun/calibrate.py`: all-seed target recommendation without mutation.
- `speedrun/run.py`: timed training/evaluation loop.
- `speedrun/summarize.py`: multi-seed official record aggregation.
- `submission/`: editable reference implementation.
- `configs/`: smoke and 50M calibration configurations.
