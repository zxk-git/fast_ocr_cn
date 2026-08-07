#!/usr/bin/env python3
"""Unified setup, training, fine-tuning, and ONNX export workflow for this repository."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ONNX_PACKAGES = ("onnx==1.17.0", "onnxruntime==1.23.2", "onnxscript==0.1.0", "onnxslim==0.1.82")


class WorkflowError(RuntimeError):
    """Raised when a workflow precondition is not satisfied."""


class WorkflowUsageError(WorkflowError):
    """Raised when an option is invalid."""


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise WorkflowError(f"Command not found: {command}")
    return resolved


def runtime_env(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    pathlib.Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def run_python(interpreter: str, code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [interpreter, "-c", code],
        cwd=PROJECT_ROOT, env=env, check=False,
        capture_output=True, text=True,
    )


# ---- setup ----

def assert_tensorflow_absent(interpreter: str) -> None:
    code = 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)'
    result = run_python(interpreter, code)
    if result.returncode == 0:
        raise WorkflowError("TensorFlow is installed in this Python environment; use a clean PyTorch-only environment.")
    if result.returncode != 1:
        raise WorkflowError(f"Unable to inspect TensorFlow installation: {result.stderr.strip()}")


def read_torch_versions(interpreter: str) -> str:
    result = run_python(interpreter, 'import torch, torchvision; print(f"{torch.__version__}|{torchvision.__version__}")')
    if result.returncode != 0:
        raise WorkflowError("PyTorch and torchvision must already be installed before running setup.")
    return result.stdout.strip()


def verify_imports(interpreter: str) -> None:
    code = (
        "import keras, onnx, onnxruntime, onnxscript, onnxslim, torch; "
        'assert keras.backend.backend() == "torch"; assert torch.cuda.is_available()'
    )
    result = run_python(interpreter, code, env=runtime_env("torch"))
    if result.returncode != 0:
        raise WorkflowError(f"Training environment verification failed: {result.stderr.strip()}")


def run_setup(skip_verify: bool) -> None:
    interpreter = require_command(os.environ.get("PYTHON_BIN", sys.executable))
    assert_tensorflow_absent(interpreter)
    versions_before = read_torch_versions(interpreter)
    print(f"Keeping installed PyTorch versions: {versions_before}")
    run([interpreter, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
         "-r", str(PROJECT_ROOT / "requirements" / "train-torch.txt")])
    run([interpreter, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *ONNX_PACKAGES])
    run([interpreter, "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])
    versions_after = read_torch_versions(interpreter)
    if versions_after != versions_before:
        raise WorkflowError(f"PyTorch was unexpectedly changed: {versions_before} -> {versions_after}")
    assert_tensorflow_absent(interpreter)
    verify_imports(interpreter)
    if not skip_verify:
        run([interpreter, str(PROJECT_ROOT / "scripts" / "verify_pytorch_training_env.py")],
            env=runtime_env("torch"))
    print("PyTorch-only training environment is ready.")


# ---- shared helpers ----

def required_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise WorkflowError(f"Required file not found: {path}")
    return path


def validate_gpu_id(gpu: str | None) -> None:
    if gpu and not gpu.isdecimal():
        raise WorkflowUsageError(f"Invalid GPU ID '{gpu}'; expected a non-negative integer.")


def find_latest_model(search_root: pathlib.Path) -> pathlib.Path | None:
    candidates = (path for path in search_root.rglob("best.keras") if path.is_file())
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def _build_command_options(options: dict[str, object]) -> list[str]:
    return [str(v) for pair in options.items() for v in pair]


def _gpu_env(gpu: str | None) -> dict[str, str]:
    if gpu:
        validate_gpu_id(gpu)
        return {"CUDA_VISIBLE_DEVICES": gpu}
    return {}


# ---- train ----

def _common_train_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "--model-config-file": required_file(pathlib.Path(args.model_config)),
        "--plate-config-file": required_file(pathlib.Path(args.plate_config)),
        "--annotations": required_file(pathlib.Path(args.dataset_root) / "train" / "annotations.csv"),
        "--val-annotations": required_file(pathlib.Path(args.dataset_root) / "val" / "annotations.csv"),
        "--epochs": str(args.epochs),
        "--batch-size": str(args.batch_size),
        "--lr": str(args.lr),
        "--early-stopping-patience": str(args.early_stopping_patience),
        "--early-stopping-metric": str(args.early_stopping_metric),
        "--workers": str(args.workers),
    }


def build_train_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    output = pathlib.Path(args.output_dir)
    options = _common_train_options(args)
    options["--output-dir"] = output

    env = runtime_env("torch")
    env.update(_gpu_env(args.gpu))

    command = [
        require_command(os.environ.get("FAST_PLATE_OCR_BIN", "fast-plate-ocr")),
        "train",
        *_build_command_options(options),
    ]
    return command, env


def run_train(args: argparse.Namespace) -> None:
    command, env = build_train_command(args)
    print(f"Starting PyTorch training: epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"Output: {args.output_dir}")
    run(command, env=env)


# ---- fine-tune ----

def resolve_fine_tune_model(model_arg: str | None, search_root: str) -> pathlib.Path:
    if model_arg:
        model_path = pathlib.Path(model_arg)
    else:
        latest = find_latest_model(pathlib.Path(search_root))
        if latest is None:
            raise WorkflowError(f"Trained model not found below: {search_root}")
        model_path = latest
    if not model_path.is_file():
        raise WorkflowError(f"Trained model not found: {model_path}")
    return model_path.resolve()


def build_fine_tune_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model_path = resolve_fine_tune_model(args.model, args.search_root)
    output = pathlib.Path(args.output_dir)

    options = _common_train_options(args)
    options["--output-dir"] = output
    options["--weights-path"] = model_path

    env = runtime_env("torch")
    env.update(_gpu_env(args.gpu))

    command = [
        require_command(os.environ.get("FAST_PLATE_OCR_BIN", "fast-plate-ocr")),
        "train",
        *_build_command_options(options),
    ]
    return command, env


def run_fine_tune(args: argparse.Namespace) -> None:
    command, env = build_fine_tune_command(args)
    print(f"Fine-tuning model: {command[command.index('--weights-path') + 1]}")
    print(f"Learning rate: {args.lr}, epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    run(command, env=env)


# ---- export ----

def resolve_export_model(model_arg: str | None, search_root: str) -> pathlib.Path:
    return resolve_fine_tune_model(model_arg, search_root)


def build_export_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model_path = resolve_export_model(args.model, args.search_root)
    if args.plate_config:
        plate_config = pathlib.Path(args.plate_config)
    else:
        plate_config = model_path.with_name("plate_config.yaml")
        if not plate_config.is_file():
            plate_config = PROJECT_ROOT / "config" / "cn_plate_config.yaml"
    required_file(plate_config)

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()

    interpreter = require_command(os.environ.get("PYTHON_BIN", sys.executable))
    env = runtime_env("tensorflow")
    run([interpreter, "-c", "import onnx, onnxruntime, onnxscript, onnxslim"], env=env)

    command = [
        require_command(os.environ.get("FAST_PLATE_OCR_BIN", "fast-plate-ocr")),
        "export",
        "--model", str(model_path),
        "--plate-config-file", str(plate_config.resolve()),
        "--format", "onnx",
        "--save-dir", str(output_dir),
        "--dynamic-batch",
        "--no-skip-validation",
        "--onnx-opset-version", "13",
        "--simplify",
        "--onnx-input-dtype", "float32",
        "--onnx-data-format", "channels_last",
    ]
    return command, env


def run_export(args: argparse.Namespace) -> None:
    command, env = build_export_command(args)
    print(f"Exporting ONNX model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    run(command, env=env)


# ---- CLI ----

def _add_train_args(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument("--dataset-root", default=str(PROJECT_ROOT.parent.parent / "asserts" / "FastOCRData"),
                        help="训练数据根目录 (含 train/val)")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "models" / "cct_xs_v2.yaml"),
                        help="模型结构配置 YAML")
    parser.add_argument("--plate-config", default=str(PROJECT_ROOT / "config" / "cn_plate_config.yaml"),
                        help="车牌字符集配置 YAML")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="初始学习率")
    parser.add_argument("--early-stopping-patience", type=int, help="早停耐心值")
    parser.add_argument("--early-stopping-metric", default="val_plate_acc",
                        help="早停监控指标 (默认 val_plate_acc)")
    parser.add_argument("--workers", type=int, help="数据加载线程数")
    parser.add_argument("--quick", action="store_true", help="快速验证模式 (3 epochs)")
    parser.add_argument("--gpu", help="GPU 设备编号")


def _resolve_train_defaults(args: argparse.Namespace, prefix: str) -> None:
    """Apply default values for training parameters that were not specified."""
    if args.epochs is None:
        args.epochs = 3 if args.quick else 200
    if args.batch_size is None:
        args.batch_size = 2048
    if args.lr is None:
        args.lr = 0.001
    if args.early_stopping_patience is None:
        args.early_stopping_patience = 10
    if args.workers is None:
        args.workers = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    setup = subparsers.add_parser("setup", help="Install and verify the PyTorch training environment")
    setup.add_argument("--skip-verify", action="store_true", help="Skip GPU and dataset verification")
    setup.set_defaults(func=lambda args: run_setup(args.skip_verify))

    # train
    train = subparsers.add_parser("train", help="Train the OCR model with Keras Torch backend")
    _add_train_args(train, "train")
    train.add_argument("--output-dir", default=str(PROJECT_ROOT / "trained_models" / "cblprd_cct_s_v2_torch"),
                       help="模型输出目录")
    train.set_defaults(func=lambda args: _resolve_train_defaults(args, "train") or run_train(args))

    # fine-tune
    ft = subparsers.add_parser("fine-tune", help="Fine-tune a pre-trained model on new data")
    ft.add_argument("model", nargs="?", help="Pretrained model .keras path")
    ft.add_argument("--search-root", default=str(PROJECT_ROOT / "trained_models"),
                    help="查找最新 best.keras 的根目录 (未指定 model 时使用)")
    _add_train_args(ft, "fine-tune")
    ft.add_argument("--output-dir", default=str(PROJECT_ROOT / "trained_models" / "fine_tuned" / "cct_xs_v2_torch"),
                    help="微调模型输出目录")
    ft.set_defaults(func=lambda args: _resolve_fine_tune_defaults(args) or run_fine_tune(args))

    # export
    exp = subparsers.add_parser("export", help="Export a trained Keras model to ONNX")
    exp.add_argument("model", nargs="?", help="Keras model path; defaults to newest best.keras")
    exp.add_argument("--search-root", default=str(PROJECT_ROOT / "trained_models"),
                     help="查找最新 best.keras 的根目录")
    exp.add_argument("--plate-config", help="plate_config.yaml 路径")
    exp.add_argument("--output-dir", default="",
                     help="ONNX 输出目录 (默认模型同目录)")
    exp.set_defaults(func=lambda args: run_export(args))

    return parser


def _resolve_fine_tune_defaults(args: argparse.Namespace) -> None:
    if args.epochs is None:
        args.epochs = 3 if args.quick else 50
    if args.batch_size is None:
        args.batch_size = 2048
    if args.lr is None:
        args.lr = 0.0001
    if args.early_stopping_patience is None:
        args.early_stopping_patience = 20
    if args.workers is None:
        args.workers = 16


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except WorkflowUsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except WorkflowError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"error: command failed with exit code {error.returncode}: {error.cmd}", file=sys.stderr)
        return error.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
