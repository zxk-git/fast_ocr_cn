#!/usr/bin/env python3
"""Compare one Keras model output with its exported ONNX model output."""

import argparse
import os
import pathlib
import sys

import numpy as np


if __package__:
    from scripts import test_single_image as inference
else:
    import test_single_image as inference


def compare_outputs(
    keras_output: np.ndarray,
    onnx_output: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, str]:
    if keras_output.size == 0 or onnx_output.size == 0:
        raise ValueError("Model outputs must not be empty.")

    keras_shape = "x".join(str(dimension) for dimension in keras_output.shape)
    onnx_shape = "x".join(str(dimension) for dimension in onnx_output.shape)
    report = {
        "keras_shape": keras_shape,
        "onnx_shape": onnx_shape,
        "comparable": str(keras_output.shape == onnx_output.shape).lower(),
    }
    if keras_output.shape != onnx_output.shape:
        return report

    keras_values = keras_output.astype(np.float64, copy=False).ravel()
    onnx_values = onnx_output.astype(np.float64, copy=False).ravel()
    absolute_diff = np.abs(keras_values - onnx_values)
    norm_product = np.linalg.norm(keras_values) * np.linalg.norm(onnx_values)
    cosine_similarity = (
        float(np.dot(keras_values, onnx_values) / norm_product)
        if norm_product
        else float(np.array_equal(keras_values, onnx_values))
    )
    report.update(
        {
            "max_abs_diff": f"{np.max(absolute_diff):.10g}",
            "mean_abs_diff": f"{np.mean(absolute_diff):.10g}",
            "rmse": f"{np.sqrt(np.mean(np.square(absolute_diff))):.10g}",
            "cosine_similarity": f"{cosine_similarity:.10g}",
            "allclose": str(np.allclose(keras_values, onnx_values, atol=atol, rtol=rtol)).lower(),
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=pathlib.Path, default=pathlib.Path("data/image.png"))
    parser.add_argument("--keras-model", type=pathlib.Path, default=pathlib.Path("trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/best.keras"))
    parser.add_argument("--onnx-model", type=pathlib.Path, default=pathlib.Path("trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/best.onnx"))
    parser.add_argument(
        "--plate-config",
        type=pathlib.Path,
        default=pathlib.Path("config/cn_plate_config.yaml"),
    )
    parser.add_argument(
        "--keras-backend",
        choices=("torch", "tensorflow", "jax"),
        default=os.environ.get("KERAS_BACKEND", "torch"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.atol < 0 or args.rtol < 0:
            raise ValueError("Comparison tolerances must be non-negative.")
        image_path = inference.existing_file(args.image, "Image")
        keras_path = inference.existing_file(args.keras_model, "Keras model")
        onnx_path = inference.existing_file(args.onnx_model, "ONNX model")
        config_path = inference.existing_file(args.plate_config, "Plate configuration")
        batch, config = inference.load_batch(image_path, config_path)
        keras_output = inference.run_keras(batch, keras_path, args.keras_backend)
        onnx_output = inference.run_onnx(batch, onnx_path, args.device)
        report = {
            "keras_plate": inference.decode_plate(keras_output, config),
            "onnx_plate": inference.decode_plate(onnx_output, config),
            **compare_outputs(keras_output, onnx_output, atol=args.atol, rtol=args.rtol),
        }
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for name, value in report.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
