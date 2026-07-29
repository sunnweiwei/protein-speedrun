#!/bin/sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
workspace_root=/mnt/workspace/sunweiwei_google_com
default_socket=unix://$workspace_root/ai4ai-docker.sock
docker_host=${DOCKER_HOST:-$default_socket}
image=${PROTEIN_SPEEDRUN_IMAGE:-ai4ai-protein-pretrain-speedrun:dev}
protenix_image=ai4s-share-public-cn-beijing.cr.volces.com/release/protenix@sha256:5e91abe0ef9ec8fc34581080d85cad5cd4c27dcaead4f1920b14c01cc0642434
data_root=${PROTEIN_SPEEDRUN_DATA_ROOT:-$workspace_root/protein-pretrain-speedrun-data}
run_root=${PROTEIN_SPEEDRUN_RUN_ROOT:-$workspace_root/protein-pretrain-speedrun-runs}
speedrun_gpu=${PROTEIN_SPEEDRUN_GPU:-0}

case "$speedrun_gpu" in
  ""|*[!0-9]*) echo "PROTEIN_SPEEDRUN_GPU must be one GPU index" >&2; exit 2 ;;
esac

case "$data_root" in
  "$workspace_root"/*) ;;
  *) echo "data root must be under $workspace_root" >&2; exit 2 ;;
esac

case "$run_root" in
  "$workspace_root"/*) ;;
  *) echo "run root must be under $workspace_root" >&2; exit 2 ;;
esac

docker_root=$(
  sudo docker --host "$docker_host" info --format '{{.DockerRootDir}}'
)
case "$docker_root" in
  "$workspace_root"/*) ;;
  *) echo "Docker root is not on the large workspace disk: $docker_root" >&2; exit 2 ;;
esac

mkdir -p "$data_root" "$run_root"

build_image() {
  sudo docker --host "$docker_host" build \
    --tag "$image" \
    --file "$project_root/Dockerfile" \
    "$project_root"
}

run_project_python() {
  speedrun_image_id=$(
    sudo docker --host "$docker_host" image inspect "$image" --format '{{.Id}}'
  )
  sudo docker --host "$docker_host" run --rm \
    --network none \
    --gpus "device=$speedrun_gpu" \
    --user "$(id -u):$(id -g)" \
    --env "SPEEDRUN_IMAGE_ID=$speedrun_image_id" \
    --env "PYTHONPATH=/workspace" \
    --volume "$project_root:/workspace:ro" \
    --volume "$data_root:/speedrun-data" \
    --volume "$run_root:/speedrun-runs" \
    --workdir /workspace \
    "$image" \
    python "$@"
}

command=${1:-}
case "$command" in
  smoke)
    build_image
    if [ ! -f "$data_root/smoke/corpus.npz" ]; then
      run_project_python \
        "speedrun/prepare_corpus.py" synthetic \
        --output /speedrun-data/smoke/corpus.npz
    fi
    speedrun_run_id=$(date -u +%Y%m%dT%H%M%SZ)
    run_project_python \
      "speedrun/run.py" \
      --config "configs/smoke.json" \
      --protocol "protocol.v0.json" \
      --corpus /speedrun-data/smoke/corpus.npz \
      --output "/speedrun-runs/smoke/$speedrun_run_id/seed-42"
    run_project_python \
      -m unittest discover \
      -s "tests" \
      -v
    ;;
  prepare-gold-small)
    output_dir=$data_root/gold_small_v1
    mkdir -p "$output_dir"
    if [ -e "$output_dir/corpus.npz" ] && [ -e "$output_dir/manifest.json" ]; then
      echo "immutable gold_small_v1 corpus already exists at $output_dir"
      exit 0
    fi
    if [ -e "$output_dir/corpus.npz" ] || [ -e "$output_dir/manifest.json" ]; then
      echo "refusing incomplete gold_small_v1 output at $output_dir" >&2
      exit 2
    fi
    sudo docker --host "$docker_host" run --rm \
      --network none \
      --env PYTHONPATH=/protenix-src \
      --volume "$project_root:/workspace:ro" \
      --volume "$workspace_root/Protenix-training:/protenix-src:ro" \
      --volume "$workspace_root/protenix-data-20240522/mmcif_bioassembly:/structures:ro" \
      --volume "$workspace_root/foldgym-data/manifests/gold_small/v1:/source-manifest:ro" \
      --volume "$output_dir:/output" \
      --workdir /workspace \
      "$protenix_image" \
      python \
      "speedrun/prepare_corpus.py" gold-small \
      --source-manifest /source-manifest/manifest.jsonl.gz \
      --structure-dir /structures \
      --materializer-image "$protenix_image" \
      --output /output/corpus.npz \
      --metadata-output /output/manifest.json
    sudo docker --host "$docker_host" run --rm \
      --network none \
      --volume "$output_dir:/output" \
      "$protenix_image" \
      chown "$(id -u):$(id -g)" /output/corpus.npz /output/manifest.json
    ;;
  run)
    if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
      echo "usage: $0 run CONFIG CORPUS OUTPUT [SEED]" >&2
      exit 2
    fi
    build_image
    config=$2
    corpus=$3
    output=$4
    case "$corpus" in
      "$data_root"/*) ;;
      *) echo "corpus must be under $data_root" >&2; exit 2 ;;
    esac
    case "$output" in
      "$run_root"/*) ;;
      *) echo "output must be under $run_root" >&2; exit 2 ;;
    esac
    corpus_in_container=/speedrun-data/${corpus#"$data_root/"}
    output_in_container=/speedrun-runs/${output#"$run_root/"}
    if [ "$#" -eq 5 ]; then
      run_project_python \
        "speedrun/run.py" \
        --config "$config" \
        --protocol "protocol.v0.json" \
        --corpus "$corpus_in_container" \
        --output "$output_in_container" \
        --seed "$5"
    else
      run_project_python \
        "speedrun/run.py" \
        --config "$config" \
        --protocol "protocol.v0.json" \
        --corpus "$corpus_in_container" \
        --output "$output_in_container"
    fi
    ;;
  verify)
    if [ "$#" -ne 3 ]; then
      echo "usage: $0 verify CHECKPOINT CORPUS" >&2
      exit 2
    fi
    build_image
    checkpoint=$2
    corpus=$3
    case "$checkpoint" in
      "$run_root"/*) ;;
      *) echo "checkpoint must be under $run_root" >&2; exit 2 ;;
    esac
    case "$corpus" in
      "$data_root"/*) ;;
      *) echo "corpus must be under $data_root" >&2; exit 2 ;;
    esac
    checkpoint_in_container=/speedrun-runs/${checkpoint#"$run_root/"}
    corpus_in_container=/speedrun-data/${corpus#"$data_root/"}
    run_project_python \
      "speedrun/evaluate.py" \
      --checkpoint "$checkpoint_in_container" \
      --corpus "$corpus_in_container" \
      --protocol "protocol.v0.json" \
      --device cuda \
      --repeat 3 \
      --output "$checkpoint_in_container/verification.json"
    ;;
  calibrate)
    if [ "$#" -ne 4 ]; then
      echo "usage: $0 calibrate RUNS INITIAL_MARGIN FINAL_MARGIN" >&2
      exit 2
    fi
    build_image
    calibration_runs=$2
    case "$calibration_runs" in
      "$run_root"/*) ;;
      *) echo "calibration runs must be under $run_root" >&2; exit 2 ;;
    esac
    calibration_runs_in_container=/speedrun-runs/${calibration_runs#"$run_root/"}
    run_project_python \
      "speedrun/calibrate.py" \
      --runs "$calibration_runs_in_container" \
      --protocol "protocol.v0.json" \
      --initial-margin "$3" \
      --final-margin "$4"
    ;;
  *)
    echo "usage: $0 {smoke|prepare-gold-small|run CONFIG CORPUS OUTPUT [SEED]|verify CHECKPOINT CORPUS|calibrate RUNS INITIAL_MARGIN FINAL_MARGIN}" >&2
    exit 2
    ;;
esac
