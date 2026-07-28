#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${FAST_PLATE_OCR_DATASET_ROOT:-/zxk/plate_ocr/plate/CBLPRD-330K/fast-plate-ocr}"
CLI_BIN="${FAST_PLATE_OCR_BIN:-fast-plate-ocr}"
MODEL_CONFIG="${FAST_PLATE_OCR_MODEL_CONFIG:-${PROJECT_ROOT}/models/cct_s_v2.yaml}"
PLATE_CONFIG="${FAST_PLATE_OCR_PLATE_CONFIG:-${PROJECT_ROOT}/config/cn_plate_config.yaml}"
OUTPUT_DIR="${FAST_PLATE_OCR_OUTPUT_DIR:-${PROJECT_ROOT}/trained_models/cblprd_cct_s_v2_torch}"
GPU_ID="${FAST_PLATE_OCR_GPU:-}"

usage() {
    cat <<'EOF'
Usage: scripts/train_cblprd_torch.sh [--quick] [--gpu ID] [train options]

Run CBLPRD training with the Keras PyTorch backend. Extra options are appended
to the CLI command, so they override the defaults below.

  --quick     Run 3 epochs and write to trained_models/quick_test
  --gpu ID    Select one physical GPU through CUDA_VISIBLE_DEVICES
  -h, --help  Show this help

Environment overrides:
  FAST_PLATE_OCR_GPU
  FAST_PLATE_OCR_DATASET_ROOT
  FAST_PLATE_OCR_MODEL_CONFIG
  FAST_PLATE_OCR_PLATE_CONFIG
  FAST_PLATE_OCR_OUTPUT_DIR
  FAST_PLATE_OCR_BIN
EOF
}

QUICK_MODE=0
FORWARDED_ARGS=()
while (($#)); do
    case "$1" in
        --quick)
            QUICK_MODE=1
            ;;
        --gpu)
            if (($# < 2)) || [[ "$2" == -* ]]; then
                echo "--gpu requires a numeric GPU ID." >&2
                exit 2
            fi
            GPU_ID="$2"
            shift
            ;;
        --gpu=*)
            GPU_ID="${1#*=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            ;;
    esac
    shift
done

if [[ -n "${GPU_ID}" && ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID '${GPU_ID}'; expected a non-negative integer." >&2
    exit 2
fi

EPOCHS="${FAST_PLATE_OCR_EPOCHS:-200}"
BATCH_SIZE="${FAST_PLATE_OCR_BATCH_SIZE:-1024}"
if ((QUICK_MODE)); then
    EPOCHS="${FAST_PLATE_OCR_QUICK_EPOCHS:-3}"
    BATCH_SIZE="${FAST_PLATE_OCR_QUICK_BATCH_SIZE:-1024}"
    OUTPUT_DIR="${FAST_PLATE_OCR_QUICK_OUTPUT_DIR:-${PROJECT_ROOT}/trained_models/quick_test}"
fi

TRAIN_ANNOTATIONS="${DATASET_ROOT}/train/annotations.csv"
VAL_ANNOTATIONS="${DATASET_ROOT}/val/annotations.csv"
for required_path in \
    "${MODEL_CONFIG}" \
    "${PLATE_CONFIG}" \
    "${TRAIN_ANNOTATIONS}" \
    "${VAL_ANNOTATIONS}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required file not found: ${required_path}" >&2
        exit 1
    fi
done

if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
    echo "Command not found: ${CLI_BIN}. Run scripts/setup_pytorch_env.sh first." >&2
    exit 1
fi

export KERAS_BACKEND=torch
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export NO_ALBUMENTATIONS_UPDATE=1
if [[ -n "${GPU_ID}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
mkdir -p "${MPLCONFIGDIR}"

echo "Starting PyTorch training: model=$(basename "${MODEL_CONFIG}"), epochs=${EPOCHS}, batch_size=${BATCH_SIZE}"
echo "Dataset: ${DATASET_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

exec "${CLI_BIN}" train \
    --model-config-file "${MODEL_CONFIG}" \
    --plate-config-file "${PLATE_CONFIG}" \
    --annotations "${TRAIN_ANNOTATIONS}" \
    --val-annotations "${VAL_ANNOTATIONS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${FAST_PLATE_OCR_LR:-0.001}" \
    --early-stopping-patience "${FAST_PLATE_OCR_EARLY_STOPPING_PATIENCE:-20}" \
    --early-stopping-metric "${FAST_PLATE_OCR_EARLY_STOPPING_METRIC:-val_plate_acc}" \
    --validate-dataset "${FAST_PLATE_OCR_VALIDATE_DATASET:-warn}" \
    --workers "${FAST_PLATE_OCR_WORKERS:-16}" \
    --output-dir "${OUTPUT_DIR}" \
    "${FORWARDED_ARGS[@]}"
