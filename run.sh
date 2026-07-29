#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace_root=/mnt/workspace/sunweiwei_google_com
docker_host=${DOCKER_HOST:-unix://$workspace_root/ai4ai-docker.sock}
image=protein-speedrun:dev
data_root=${PROTEIN_SPEEDRUN_DATA_ROOT:-$workspace_root/protein-pretrain-speedrun-data}
run_root=${PROTEIN_SPEEDRUN_RUN_ROOT:-$workspace_root/protein-pretrain-speedrun-runs}
gpu=${PROTEIN_SPEEDRUN_GPU:-0}

case "$data_root:$run_root" in
  "$workspace_root"/*:"$workspace_root"/*) ;;
  *) echo "data and runs must be on $workspace_root" >&2; exit 2 ;;
esac
case "$gpu" in
  ""|*[!0-9]*) echo "GPU must be one numeric index" >&2; exit 2 ;;
esac

docker_root=$(sudo docker --host "$docker_host" info --format '{{.DockerRootDir}}')
case "$docker_root" in
  "$workspace_root"/*) ;;
  *) echo "Docker root must be on the large workspace disk" >&2; exit 2 ;;
esac

mkdir -p "$data_root" "$run_root"
sudo docker --host "$docker_host" build \
  --tag "$image" \
  --file "$project_root/Dockerfile" \
  "$project_root"

docker_python() {
  sudo docker --host "$docker_host" run --rm \
    --network none \
    --gpus "device=$gpu" \
    --user "$(id -u):$(id -g)" \
    --env PYTHONPATH=/workspace \
    --volume "$project_root:/workspace:ro" \
    --volume "$data_root:/data" \
    --volume "$run_root:/runs" \
    --workdir /workspace \
    "$image" \
    python "$@"
}

docker_ddp() {
  sudo docker --host "$docker_host" run --rm \
    --network none \
    --gpus all \
    --user "$(id -u):$(id -g)" \
    --env PYTHONPATH=/workspace \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env NCCL_ALGO=Ring \
    --env NCCL_PROTO=Simple \
    --volume "$project_root:/workspace:ro" \
    --volume "$data_root:/data:ro" \
    --volume "$run_root:/runs" \
    --workdir /workspace \
    "$image" \
    torchrun --nnodes=1 --nproc-per-node=8 \
      --master-addr=127.0.0.1 --master-port=29500 "$@"
}

command=${1:-}
run_id=$(date -u +%Y%m%dT%H%M%SZ)
case "$command" in
  smoke)
    docker_python speedrun.py smoke \
      --config config.json \
      --corpus /data/smoke/corpus.npz \
      --output "/runs/smoke/$run_id"
    ;;
  run)
    corpus=${2:-$data_root/gold_small_v1/corpus.npz}
    output=${3:-$run_root/gold-small/$run_id}
    seed=${4:-}
    case "$corpus:$output" in
      "$data_root"/*:"$run_root"/*) ;;
      *) echo "corpus/output must stay under the configured large-disk roots" >&2; exit 2 ;;
    esac
    if [ ! -f "$corpus" ]; then
      echo "missing corpus: $corpus" >&2
      exit 2
    fi
    corpus_in_container=/data/${corpus#"$data_root/"}
    output_in_container=/runs/${output#"$run_root/"}
    if [ -n "$seed" ]; then
      docker_python speedrun.py run \
        --config config.json \
        --corpus "$corpus_in_container" \
        --output "$output_in_container" \
        --seed "$seed"
    else
      docker_python speedrun.py run \
        --config config.json \
        --corpus "$corpus_in_container" \
        --output "$output_in_container"
    fi
    ;;
  ddp)
    config=${2:-$project_root/configs/esm2-150m-d0.json}
    corpus=${3:-$data_root/uniref50_2021_04_d0/corpus.json}
    output=${4:-$run_root/esm2-150m-d0/$run_id}
    case "$config:$corpus:$output" in
      "$project_root"/*:"$data_root"/*:"$run_root"/*) ;;
      *) echo "config/corpus/output must stay under their configured roots" >&2; exit 2 ;;
    esac
    if [ ! -f "$config" ] || [ ! -f "$corpus" ]; then
      echo "missing config or corpus: $config $corpus" >&2
      exit 2
    fi
    config_in_container=/workspace/${config#"$project_root/"}
    corpus_in_container=/data/${corpus#"$data_root/"}
    output_in_container=/runs/${output#"$run_root/"}
    docker_ddp speedrun.py run \
      --config "$config_in_container" \
      --corpus "$corpus_in_container" \
      --output "$output_in_container"
    ;;
  *)
    echo "usage: $0 {smoke|run [CORPUS OUTPUT SEED]|ddp [CONFIG CORPUS OUTPUT]}" >&2
    exit 2
    ;;
esac
