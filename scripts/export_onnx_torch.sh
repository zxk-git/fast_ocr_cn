#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLI_BIN="${FAST_PLATE_OCR_BIN:-fast-plate-ocr}"
TRAIN_OUTPUT_DIR="${FAST_PLATE_OCR_TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/trained_models/cblprd_cct_s_v2_torch}"

usage() {
    cat <<'EOF'
Usage: scripts/export_onnx_torch.sh [MODEL.keras] [fast-plate-ocr export options]
  ### 3. ONNX 导出脚本

  scripts/export_onnx_torch.sh:1

  自动查找正式训练目录下最新的 best.keras：

  ./scripts/export_onnx_torch.sh

  指定模型导出：

  ./scripts/export_onnx_torch.sh \
    trained_models/cblprd_cct_s_v2_torch/训练时间目录/best.keras

  导出快速测试模型：

  ./scripts/export_onnx_torch.sh \
    trained_models/quick_test/训练时间目录/best.keras

  指定 ONNX 输出目录：

  FAST_PLATE_OCR_ONNX_OUTPUT_DIR=exported_models \
    ./scripts/export_onnx_torch.sh

  禁用模型简化：

  ./scripts/export_onnx_torch.sh --no-simplify

  导出 NCHW、float32 输入模型：

  ./scripts/export_onnx_torch.sh \
    --onnx-data-format channels_first \
    --onnx-input-dtype float32

    
Export a trained Keras model to ONNX with the PyTorch backend. When MODEL.keras
is omitted, the newest best.keras under the training output directory is used.

Environment overrides:
  FAST_PLATE_OCR_MODEL_PATH
  FAST_PLATE_OCR_PLATE_CONFIG
  FAST_PLATE_OCR_ONNX_OUTPUT_DIR
  FAST_PLATE_OCR_TRAIN_OUTPUT_DIR
  FAST_PLATE_OCR_BIN
  PYTHON_BIN

Default ONNX interface: channels_last, uint8 input, dynamic batch, simplified.
Extra CLI options are appended and can override these defaults.
EOF
}

find_latest_model() {
    local search_root="$1"
    local latest=""
    local candidate
    shopt -s nullglob globstar
    for candidate in "${search_root}"/**/best.keras; do
        if [[ -z "${latest}" || "${candidate}" -nt "${latest}" ]]; then
            latest="${candidate}"
        fi
    done
    printf '%s' "${latest}"
}

if (($#)) && [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

MODEL_PATH="${FAST_PLATE_OCR_MODEL_PATH:-}"
if (($#)) && [[ "$1" != -* ]]; then
    MODEL_PATH="$1"
    shift
fi
if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH="$(find_latest_model "${TRAIN_OUTPUT_DIR}")"
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}" ]]; then
    echo "Trained model not found. Pass MODEL.keras or run scripts/train_cblprd_torch.sh first." >&2
    exit 1
fi

MODEL_DIR="$(cd -- "$(dirname -- "${MODEL_PATH}")" && pwd)"
MODEL_PATH="${MODEL_DIR}/$(basename -- "${MODEL_PATH}")"
PLATE_CONFIG="${FAST_PLATE_OCR_PLATE_CONFIG:-}"
if [[ -z "${PLATE_CONFIG}" && -f "${MODEL_DIR}/plate_config.yaml" ]]; then
    PLATE_CONFIG="${MODEL_DIR}/plate_config.yaml"
fi
PLATE_CONFIG="${PLATE_CONFIG:-${PROJECT_ROOT}/config/cn_plate_config.yaml}"
if [[ ! -f "${PLATE_CONFIG}" ]]; then
    echo "Plate configuration not found: ${PLATE_CONFIG}" >&2
    exit 1
fi

OUTPUT_DIR="${FAST_PLATE_OCR_ONNX_OUTPUT_DIR:-${MODEL_DIR}}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! command -v "${CLI_BIN}" >/dev/null 2>&1; then
    echo "Command not found: ${CLI_BIN}. Run scripts/setup_pytorch_env.sh first." >&2
    exit 1
fi

export KERAS_BACKEND=torch
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "${MPLCONFIGDIR}"

if ! "${PYTHON_BIN}" -c 'import onnx, onnxruntime, onnxscript, onnxslim'; then
    echo "ONNX export dependencies are missing. Run scripts/setup_pytorch_env.sh first." >&2
    exit 1
fi

echo "Exporting ONNX model: ${MODEL_PATH}"
echo "Output directory: ${OUTPUT_DIR}"

exec "${CLI_BIN}" export \
    --model "${MODEL_PATH}" \
    --plate-config-file "${PLATE_CONFIG}" \
    --format onnx \
    --save-dir "${OUTPUT_DIR}" \
    --dynamic-batch \
    --simplify \
    --onnx-input-dtype uint8 \
    --onnx-data-format channels_last \
    "$@"
