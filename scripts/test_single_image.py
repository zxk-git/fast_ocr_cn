#!/usr/bin/env python3
"""Run one license-plate image through a local ONNX or Keras model."""

import argparse
import importlib
import os
import pathlib
import sys
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np


ModelType = Literal["onnx", "keras"]
Device = Literal["auto", "cpu", "cuda"]
ONNX_DTYPES = {"tensor(uint8)": np.uint8, "tensor(float)": np.float32}


def model_type(model_path: pathlib.Path) -> ModelType:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".keras":
        return "keras"
    raise ValueError(f"Unsupported model format: {model_path.suffix or '<none>'}")


def resolve_plate_config(model_path: pathlib.Path, requested: pathlib.Path | None) -> pathlib.Path:
    config_path = requested if requested is not None else model_path.with_name("plate_config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Plate configuration not found: {config_path}")
    return config_path.resolve()


def cast_onnx_input(batch: np.ndarray, input_type: str) -> np.ndarray:
    dtype = ONNX_DTYPES.get(input_type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input type: {input_type}")
    return batch.astype(dtype, copy=False)


def plate_output(outputs: Any) -> np.ndarray:
    if isinstance(outputs, Mapping):
        if not outputs:
            raise ValueError("Model returned no outputs.")
        selected = outputs.get("plate", next(iter(outputs.values())))
    elif isinstance(outputs, (list, tuple)):
        if not outputs:
            raise ValueError("Model returned no outputs.")
        selected = outputs[0]
    else:
        selected = outputs
    if hasattr(selected, "detach"):
        selected = selected.detach()
    if hasattr(selected, "cpu"):
        selected = selected.cpu()
    return np.asarray(selected)


def select_onnx_providers(device: Device, available: list[str]) -> list[str]:
    if device == "auto":
        if not available:
            raise RuntimeError("ONNX Runtime reported no available execution providers.")
        return available
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if provider not in available:
        raise RuntimeError(f"{provider} is not available; available providers: {available}")
    return [provider]


def load_batch(image_path: pathlib.Path, config_path: pathlib.Path) -> tuple[np.ndarray, Any]:
    config_module = importlib.import_module("fast_plate_ocr.inference.config")
    process_module = importlib.import_module("fast_plate_ocr.core.process")
    config = config_module.PlateConfig.from_yaml(config_path)
    frame = process_module.read_and_resize_plate_image(
        image_path,
        img_height=config.img_height,
        img_width=config.img_width,
        image_color_mode=config.image_color_mode,
        keep_aspect_ratio=config.keep_aspect_ratio,
        interpolation_method=config.interpolation,
        padding_color=config.padding_color,
    )
    return process_module.preprocess_image(frame), config


def decode_plate(raw_output: np.ndarray, config: Any) -> str:
    process_module = importlib.import_module("fast_plate_ocr.core.process")
    predictions = process_module.postprocess_output(
        model_output=raw_output,
        max_plate_slots=config.max_plate_slots,
        model_alphabet=config.alphabet,
        pad_char=config.pad_char,
    )
    if len(predictions) != 1:
        raise ValueError(f"Expected exactly one prediction, got {len(predictions)}.")
    return predictions[0].plate


def run_onnx(batch: np.ndarray, model_path: pathlib.Path, device: Device) -> np.ndarray:
    ort = importlib.import_module("onnxruntime")
    available = ort.get_available_providers()
    providers = select_onnx_providers(device, available)
    session = ort.InferenceSession(str(model_path), providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if not inputs:
        raise ValueError("ONNX model has no inputs.")
    if not outputs:
        raise ValueError("ONNX model has no outputs.")
    model_input = inputs[0]
    output_names = [output.name for output in outputs]
    fetch_names = ["plate"] if "plate" in output_names else [output_names[0]]
    result = session.run(fetch_names, {model_input.name: cast_onnx_input(batch, model_input.type)})
    return plate_output(result)


def run_keras(batch: np.ndarray, model_path: pathlib.Path, backend: str) -> np.ndarray:
    os.environ["KERAS_BACKEND"] = backend
    keras = __import__("keras")
    importlib.import_module("fast_plate_ocr.train.model.layers")
    model = keras.models.load_model(model_path, compile=False)
    return plate_output(model(batch, training=False))


def existing_file(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=pathlib.Path, default='/home/zxk/ai_lab_project/Dataset/plate/fast-plate-ocr/data/image.png', help="Path to one cropped plate image")
    parser.add_argument("--model", type=pathlib.Path, help="Path to a .onnx or .keras model")
    parser.add_argument("--plate-config", type=pathlib.Path, default='config/cn_plate_config.yaml', help="Plate config; defaults beside the model")
    parser.add_argument(
        "--keras-backend",
        choices=("torch", "tensorflow", "jax"),
        default=os.environ.get("KERAS_BACKEND", "torch"),
        help="Keras backend used for .keras models (default: KERAS_BACKEND or torch)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="ONNX execution device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image_path = existing_file(args.image, "Image")
        model_path = existing_file(args.model, "Model")
        selected_type = model_type(model_path)
        config_path = resolve_plate_config(model_path, args.plate_config)
        batch, config = load_batch(image_path, config_path)
        if selected_type == "onnx":
            raw_output = run_onnx(batch, model_path, args.device)
        else:
            raw_output = run_keras(batch, model_path, args.keras_backend)
        plate = decode_plate(raw_output, config)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"model_type={selected_type}")
    print(f"plate={plate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
