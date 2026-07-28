#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_VERIFY=0

usage() {
    cat <<'EOF'
Usage: scripts/setup_pytorch_env.sh [--skip-verify]

  ### 1. 环境搭建脚本

  scripts/setup_pytorch_env.sh:1

  安装训练及 ONNX 依赖，并执行 GPU、数据集和模型验证：

  ./scripts/setup_pytorch_env.sh

  跳过完整验证，只安装和检查依赖：

  ./scripts/setup_pytorch_env.sh --skip-verify

  指定 Python：

  PYTHON_BIN=/usr/local/bin/python3 ./scripts/setup_pytorch_env.sh


Install fast-plate-ocr training and ONNX export dependencies without installing
or replacing PyTorch and without installing TensorFlow.

  --skip-verify  Skip the GPU and dataset verification step
  -h, --help     Show this help

Set PYTHON_BIN to select the Python interpreter.
EOF
}

while (($#)); do
    case "$1" in
        --skip-verify)
            SKIP_VERIFY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

read_torch_versions() {
    "${PYTHON_BIN}" -c \
        'import torch, torchvision; print(f"{torch.__version__}|{torchvision.__version__}")'
}

assert_no_tensorflow() {
    if "${PYTHON_BIN}" -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)'; then
        echo "TensorFlow is installed in this Python environment; use a clean PyTorch-only environment." >&2
        exit 1
    fi
}

verify_imports() {
    KERAS_BACKEND=torch "${PYTHON_BIN}" -c \
        'import keras, onnx, onnxruntime, onnxscript, onnxslim, torch; assert keras.backend.backend() == "torch"; assert torch.cuda.is_available()'
}

assert_no_tensorflow
if ! TORCH_VERSIONS_BEFORE="$(read_torch_versions)"; then
    echo "PyTorch and torchvision must already be installed before running this script." >&2
    exit 1
fi

echo "Keeping installed PyTorch versions: ${TORCH_VERSIONS_BEFORE}"
"${PYTHON_BIN}" -m pip install \
    --upgrade-strategy only-if-needed \
    -r "${PROJECT_ROOT}/requirements/train-torch.txt"
"${PYTHON_BIN}" -m pip install \
    --upgrade-strategy only-if-needed \
    "onnx==1.17.0" \
    "onnxruntime==1.23.2" \
    "onnxscript==0.1.0" \
    "onnxslim==0.1.82"
"${PYTHON_BIN}" -m pip install --no-deps -e "${PROJECT_ROOT}"

TORCH_VERSIONS_AFTER="$(read_torch_versions)"
if [[ "${TORCH_VERSIONS_AFTER}" != "${TORCH_VERSIONS_BEFORE}" ]]; then
    echo "PyTorch was unexpectedly changed: ${TORCH_VERSIONS_BEFORE} -> ${TORCH_VERSIONS_AFTER}" >&2
    exit 1
fi
assert_no_tensorflow
verify_imports

export KERAS_BACKEND=torch
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_ROOT}/.cache/matplotlib}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "${MPLCONFIGDIR}"

if ((SKIP_VERIFY == 0)); then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/verify_pytorch_training_env.py"
fi

echo "PyTorch-only training and ONNX export environment is ready."
