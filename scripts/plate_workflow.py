#!/usr/bin/env python3
"""Unified setup, training, and ONNX export workflow for this repository."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = pathlib.Path("/zxk/plate_ocr/plate/FastOCRData")
DEFAULT_TRAIN_OUTPUT = PROJECT_ROOT / "trained_models" / "cblprd_cct_s_v2_torch"
ONNX_PACKAGES = ("onnx==1.17.0", "onnxruntime==1.23.2", "onnxscript==0.1.0", "onnxslim==0.1.82")


class WorkflowError(RuntimeError):
    """Raised when a workflow precondition is not satisfied."""


class WorkflowUsageError(WorkflowError):
    """Raised when an environment-provided option is invalid."""


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
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_tensorflow_absent(interpreter: str) -> None:
    code = 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("tensorflow") else 1)'
    result = run_python(interpreter, code)
    if result.returncode == 0:
        raise WorkflowError("TensorFlow is installed in this Python environment; use a clean PyTorch-only environment.")
    if result.returncode != 1:
        raise WorkflowError(f"Unable to inspect TensorFlow installation: {result.stderr.strip()}")


def read_torch_versions(interpreter: str) -> str:
    result = run_python(
        interpreter,
        'import torch, torchvision; print(f"{torch.__version__}|{torchvision.__version__}")',
    )
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
    run(
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "--upgrade-strategy",
            "only-if-needed",
            "-r",
            str(PROJECT_ROOT / "requirements" / "train-torch.txt"),
        ]
    )
    run([interpreter, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *ONNX_PACKAGES])
    run([interpreter, "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])
    versions_after = read_torch_versions(interpreter)
    if versions_after != versions_before:
        raise WorkflowError(f"PyTorch was unexpectedly changed: {versions_before} -> {versions_after}")
    assert_tensorflow_absent(interpreter)
    verify_imports(interpreter)
    if not skip_verify:
        run(
            [interpreter, str(PROJECT_ROOT / "scripts" / "verify_pytorch_training_env.py")],
            env=runtime_env("torch"),
        )
    print("PyTorch-only training environment is ready.")


def required_file(path: pathlib.Path) -> pathlib.Path:
    if not path.is_file():
        raise WorkflowError(f"Required file not found: {path}")
    return path


def validate_gpu_id(gpu: str | None) -> None:
    if gpu and not gpu.isdecimal():
        raise WorkflowUsageError(f"Invalid GPU ID '{gpu}'; expected a non-negative integer.")


def build_train_command(quick: bool, gpu: str | None, forwarded: list[str]) -> tuple[list[str], dict[str, str]]:
    selected_gpu = gpu if gpu is not None else os.environ.get("FAST_PLATE_OCR_GPU", "")
    validate_gpu_id(selected_gpu)
    dataset_root = pathlib.Path(os.environ.get("FAST_PLATE_OCR_DATASET_ROOT", DEFAULT_DATASET_ROOT))
    model_config = required_file(
        pathlib.Path(os.environ.get("FAST_PLATE_OCR_MODEL_CONFIG", PROJECT_ROOT / "models" / "cct_s_v2.yaml"))
    )
    plate_config = required_file(
        pathlib.Path(os.environ.get("FAST_PLATE_OCR_PLATE_CONFIG", PROJECT_ROOT / "config" / "cn_plate_config.yaml"))
    )
    train_annotations = required_file(dataset_root / "train" / "annotations.csv")
    val_annotations = required_file(dataset_root / "val" / "annotations.csv")

    epochs = os.environ.get("FAST_PLATE_OCR_EPOCHS", "200")
    batch_size = os.environ.get("FAST_PLATE_OCR_BATCH_SIZE", "1024")
    output_dir = pathlib.Path(os.environ.get("FAST_PLATE_OCR_OUTPUT_DIR", DEFAULT_TRAIN_OUTPUT))
    if quick:
        epochs = os.environ.get("FAST_PLATE_OCR_QUICK_EPOCHS", "3")
        batch_size = os.environ.get("FAST_PLATE_OCR_QUICK_BATCH_SIZE", "1024")
        output_dir = pathlib.Path(
            os.environ.get("FAST_PLATE_OCR_QUICK_OUTPUT_DIR", PROJECT_ROOT / "trained_models" / "quick_test")
        )

    env = runtime_env("torch")
    if selected_gpu:
        env["CUDA_VISIBLE_DEVICES"] = selected_gpu
    options = {
        "--model-config-file": model_config,
        "--plate-config-file": plate_config,
        "--annotations": train_annotations,
        "--val-annotations": val_annotations,
        "--epochs": epochs,
        "--batch-size": batch_size,
        "--lr": os.environ.get("FAST_PLATE_OCR_LR", "0.001"),
        "--early-stopping-patience": os.environ.get("FAST_PLATE_OCR_EARLY_STOPPING_PATIENCE", "20"),
        "--early-stopping-metric": os.environ.get("FAST_PLATE_OCR_EARLY_STOPPING_METRIC", "val_plate_acc"),
        "--validate-dataset": os.environ.get("FAST_PLATE_OCR_VALIDATE_DATASET", "warn"),
        "--workers": os.environ.get("FAST_PLATE_OCR_WORKERS", "16"),
        "--output-dir": output_dir,
    }
    cli_options = (str(value) for option in options.items() for value in option)
    command = [
        require_command(os.environ.get("FAST_PLATE_OCR_BIN", "fast-plate-ocr")),
        "train",
        *cli_options,
        *forwarded,
    ]
    return command, env


def run_train(quick: bool, gpu: str | None, forwarded: list[str]) -> None:
    command, env = build_train_command(quick, gpu, forwarded)
    print(
        f"Starting PyTorch training: epochs={command[command.index('--epochs') + 1]}, "
        f"batch_size={command[command.index('--batch-size') + 1]}"
    )
    print(f"Output: {command[command.index('--output-dir') + 1]}")
    run(command, env=env)


def find_latest_model(search_root: pathlib.Path) -> pathlib.Path | None:
    candidates = (path for path in search_root.rglob("best.keras") if path.is_file())
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def resolve_export_model(model_arg: str | None) -> pathlib.Path:
    configured = model_arg or os.environ.get("FAST_PLATE_OCR_MODEL_PATH")
    if configured:
        model_path = pathlib.Path(configured)
    else:
        search_root = pathlib.Path(os.environ.get("FAST_PLATE_OCR_TRAIN_OUTPUT_DIR", DEFAULT_TRAIN_OUTPUT))
        latest = find_latest_model(search_root)
        if latest is None:
            raise WorkflowError(f"Trained model not found below: {search_root}")
        model_path = latest
    if not model_path.is_file():
        raise WorkflowError(f"Trained model not found: {model_path}")
    return model_path.resolve()


def build_export_command(model_arg: str | None, forwarded: list[str]) -> tuple[list[str], dict[str, str]]:
    model_path = resolve_export_model(model_arg)
    configured_plate = os.environ.get("FAST_PLATE_OCR_PLATE_CONFIG")
    adjacent_plate = model_path.with_name("plate_config.yaml")
    plate_config = pathlib.Path(configured_plate) if configured_plate else adjacent_plate
    if not plate_config.is_file() and not configured_plate:
        plate_config = PROJECT_ROOT / "config" / "cn_plate_config.yaml"
    required_file(plate_config)
    output_dir = pathlib.Path(os.environ.get("FAST_PLATE_OCR_ONNX_OUTPUT_DIR", model_path.parent))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    interpreter = require_command(os.environ.get("PYTHON_BIN", sys.executable))
    cli = require_command(os.environ.get("FAST_PLATE_OCR_BIN", "fast-plate-ocr"))
    env = runtime_env("tensorflow")
    run(
        [interpreter, "-c", "import onnx, onnxruntime, onnxscript, onnxslim"],
        env=env,
    )
    command = [
        cli,
        "export",
        "--model",
        str(model_path),
        "--plate-config-file",
        str(plate_config.resolve()),
        "--format",
        "onnx",
        "--save-dir",
        str(output_dir),
        "--dynamic-batch",
        "--no-skip-validation",
        "--onnx-opset-version",
        "13",
        "--simplify",
        "--onnx-input-dtype",
        "float32",
        "--onnx-data-format",
        "channels_last",
        *forwarded,
    ]
    return command, env


def run_export(model_arg: str | None, forwarded: list[str]) -> None:
    command, env = build_export_command(model_arg, forwarded)
    print(f"Exporting ONNX model: {command[command.index('--model') + 1]}")
    print(f"Output directory: {command[command.index('--save-dir') + 1]}")
    run(command, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="Install and verify the PyTorch training environment")
    setup.add_argument("--skip-verify", action="store_true", help="Skip GPU and dataset verification")
    train = subparsers.add_parser("train", help="Train the CBLPRD model with the Keras Torch backend")
    train.add_argument("--quick", action="store_true", help="Use quick-test epochs and output directory")
    train.add_argument("--gpu", help="Physical GPU ID exposed through CUDA_VISIBLE_DEVICES")
    export = subparsers.add_parser("export", help="Export a trained Keras model to ONNX")
    export.add_argument("model", nargs="?", help="Keras model path; defaults to the newest best.keras")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    try:
        if args.command == "setup":
            if forwarded:
                raise WorkflowUsageError(f"Unknown setup arguments: {' '.join(forwarded)}")
            run_setup(args.skip_verify)
        elif args.command == "train":
            run_train(args.quick, args.gpu, forwarded)
        else:
            run_export(args.model, forwarded)
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
