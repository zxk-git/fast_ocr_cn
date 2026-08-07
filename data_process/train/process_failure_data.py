#!/usr/bin/env python3
"""Evaluate a fast-plate-ocr model on one or more labeled datasets and save
prediction failures in fastplateocr format for model improvement.

This script supports three dataset modes:
    --dataset path/to/annotations.csv        single CSV file
    --dataset path/to/dataset_dir            directory with annotations.csv
    --dataset path/to/grouped_root           root containing subdirs each with annotations.csv

For each dataset, the script runs batch prediction, computes OCR metrics,
identifies cases where the prediction does not match ground truth, and
appends all failure images + labels to the challenge_data output directory.

Output structure (fastplateocr format):
    challenge_data/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── annotations.csv    (columns: image_path, plate_text)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

from evaluation_metrics import MetricAccumulator, format_report_chinese

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import test_single_image as inference  # noqa: E402


Device = Literal["auto", "cpu", "cuda"]
LOGGER = logging.getLogger("process_failure_data")


@dataclass(frozen=True)
class FailureRecord:
    """A single prediction failure with image path, ground truth and prediction."""

    image_path: pathlib.Path
    plate_text: str
    predicted_text: str


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def resolve_datasets(dataset: pathlib.Path) -> dict[str, pathlib.Path]:
    """Resolve a dataset path to {group_name: annotations.csv} mapping.

    Accepts:
        - A single CSV file → {stem: path}
        - A directory with annotations.csv → {dirname: path}
        - A root containing subdirs with annotations.csv → grouped
    """
    path = dataset.resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Dataset file must be CSV: {path}")
        name = path.parent.name if path.name == "annotations.csv" else path.stem
        return {name: path}
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {path}")
    direct = path / "annotations.csv"
    if direct.is_file():
        return {path.name: direct.resolve()}
    # Level 1: */annotations.csv  (e.g. CCPD groups with flat structure)
    grouped = {
        item.parent.name: item.resolve()
        for item in sorted(path.glob("*/annotations.csv"))
    }
    if grouped:
        return grouped
    # Level 2: */*/annotations.csv  (e.g. CCPD groups with train/val subdirs)
    grouped = {
        f"{item.parent.parent.name}/{item.parent.name}": item.resolve()
        for item in sorted(path.glob("*/*/annotations.csv"))
    }
    if not grouped:
        raise FileNotFoundError(f"No annotations.csv found under: {path}")
    return grouped


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_annotations(annotations_path: pathlib.Path) -> tuple[list[pathlib.Path], list[str]]:
    """Load image paths and ground truth labels from a fastplateocr annotations CSV."""
    image_paths: list[pathlib.Path] = []
    plate_texts: list[str] = []
    with annotations_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "plate_text"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing image_path or plate_text column: {annotations_path}")
        for line_number, row in enumerate(reader, 2):
            image_path = pathlib.Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = annotations_path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"{annotations_path}:{line_number}: image not found: {image_path}")
            image_paths.append(image_path.resolve())
            plate_texts.append(row["plate_text"])
    if not image_paths:
        raise ValueError(f"No annotations found in: {annotations_path}")
    return image_paths, plate_texts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_keras_model(model_path: pathlib.Path, device: Device, gpu: int) -> Any:
    """Load a Keras model for inference."""
    os.environ["KERAS_BACKEND"] = "torch"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch = importlib.import_module("torch")
    use_cuda = device != "cpu" and bool(torch.cuda.is_available())
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA requested but PyTorch reports it unavailable.")
    keras = importlib.import_module("keras")
    importlib.import_module("fast_plate_ocr.train.model.layers")
    return keras.models.load_model(model_path, compile=False)


def load_onnx_model(model_path: pathlib.Path, device: Device, gpu: int) -> Any:
    """Load an ONNX model for inference."""
    ort = importlib.import_module("onnxruntime")
    if device == "cpu":
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CPUExecutionProvider unavailable for ONNX Runtime.")
        providers = ["CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in ort.get_available_providers():
        providers = [("CUDAExecutionProvider", {"device_id": gpu}), "CPUExecutionProvider"]
    else:
        if device == "cuda":
            raise RuntimeError("CUDAExecutionProvider unavailable for ONNX Runtime.")
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    return session


# ---------------------------------------------------------------------------
# Image preprocessing & prediction
# ---------------------------------------------------------------------------


def load_image_batch(image_paths: list[pathlib.Path], config: Any) -> np.ndarray:
    """Load and preprocess a batch of images into a numpy array."""
    process = importlib.import_module("fast_plate_ocr.core.process")
    frames = []
    for img_path in image_paths:
        frame = process.read_and_resize_plate_image(
            img_path,
            img_height=config.img_height,
            img_width=config.img_width,
            image_color_mode=config.image_color_mode,
            keep_aspect_ratio=config.keep_aspect_ratio,
            interpolation_method=config.interpolation,
            padding_color=config.padding_color,
        )
        frames.append(frame)
    return process.preprocess_image(np.stack(frames))


def predict_batch(
    model: Any, model_type: str, batch: np.ndarray, config: Any
) -> tuple[np.ndarray, list[str]]:
    """Run inference on a batch.

    Returns:
        (raw_logits, decoded_strings)
        raw_logits shape: (batch_size, max_plate_slots, len(alphabet))
    """
    process = importlib.import_module("fast_plate_ocr.core.process")

    if model_type == "onnx":
        inputs_meta = model.get_inputs()
        if not inputs_meta:
            raise ValueError("ONNX model has no inputs.")
        outputs_meta = model.get_outputs()
        if not outputs_meta:
            raise ValueError("ONNX model has no outputs.")
        output_names = [o.name for o in outputs_meta]
        fetch_names = ["plate"] if "plate" in output_names else [output_names[0]]
        casted = inference.cast_onnx_input(batch, inputs_meta[0].type)
        raw = model.run(fetch_names, {inputs_meta[0].name: casted})
        raw_output = inference.plate_output(raw)
    else:
        raw_output = inference.plate_output(model(batch, training=False))

    predictions = process.postprocess_output(
        model_output=raw_output,
        max_plate_slots=config.max_plate_slots,
        model_alphabet=config.alphabet,
        pad_char=config.pad_char,
    )
    return raw_output, [pred.plate for pred in predictions]


def collect_failures(
    image_paths: list[pathlib.Path],
    truths: list[str],
    predictions: list[str],
) -> list[FailureRecord]:
    """Return records where predicted text does not match ground truth."""
    failures: list[FailureRecord] = []
    for img, truth, pred in zip(image_paths, truths, predictions):
        if pred != truth:
            failures.append(FailureRecord(img, truth, pred))
    return failures


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_failures(failures: list[FailureRecord], output_dir: pathlib.Path) -> int:
    """Append failure data in fastplateocr format under output_dir.

    Images are copied into output_dir/images/ (skipped if already present).
    Annotations are appended to output_dir/annotations.csv.
    Existing entries are never removed.

    Returns the number of failures written in this call.
    """
    if not failures:
        return 0

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = output_dir / "annotations.csv"
    write_header = not annotations_path.is_file()

    written = 0
    with annotations_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        if write_header:
            writer.writeheader()
        for record in failures:
            stem = record.image_path.stem
            ext = record.image_path.suffix
            target_name = f"{stem}{ext}"
            target_path = images_dir / target_name
            # Copy only once — subsequent runs with overlapping images skip the copy
            if not target_path.is_file():
                shutil.copy2(record.image_path, target_path)
            writer.writerow({
                "image_path": f"images/{target_name}",
                "plate_text": record.plate_text,
            })
            written += 1

    return written


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_single_dataset(
    *,
    annotation_path: pathlib.Path,
    group_name: str,
    config: Any,
    loaded_model: Any,
    model_type: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[FailureRecord]]:
    """Evaluate one dataset and return (metric_report, failure_records)."""
    image_paths, plate_texts = load_annotations(annotation_path)
    LOGGER.info("[%s] loaded %d images from %s", group_name, len(image_paths), annotation_path)

    metrics = MetricAccumulator(config)
    failures: list[FailureRecord] = []
    metrics_start = time.perf_counter()

    progress = tqdm(
        total=len(image_paths), desc=group_name, unit="img",
        dynamic_ncols=True,
    )
    for start in range(0, len(image_paths), batch_size):
        end = start + batch_size
        batch_paths = image_paths[start:end]
        batch_truths = plate_texts[start:end]
        batch = load_image_batch(batch_paths, config)
        t0 = time.perf_counter()
        raw_output, predictions = predict_batch(loaded_model, model_type, batch, config)
        elapsed = time.perf_counter() - t0
        metrics.update(raw_output, batch_truths, elapsed)
        batch_failures = collect_failures(batch_paths, batch_truths, predictions)
        failures.extend(batch_failures)
        progress.update(len(batch_paths))
    progress.close()

    elapsed = time.perf_counter() - metrics_start  # approximate
    return metrics.report(elapsed), failures


def run_processing(
    *,
    model: pathlib.Path,
    datasets: list[pathlib.Path],
    plate_config: pathlib.Path | None,
    output_dir: pathlib.Path,
    device: Device,
    gpu: int,
    batch_size: int,
) -> dict[str, Any]:
    """Main pipeline: resolve datasets → predict → collect failures → save."""
    model_path = inference.existing_file(model, "Model")
    model_type = inference.model_type(model_path)
    config_path = inference.resolve_plate_config(model_path, plate_config)
    config_module = importlib.import_module("fast_plate_ocr.inference.config")
    config = config_module.PlateConfig.from_yaml(config_path)

    # Discover datasets — merge results from all provided paths.
    # When multiple roots are given, prefix group names with the root directory
    # to avoid collisions (e.g. both CBLPRD-330K and CCPD2020 have "train"/"val").
    annotation_files: dict[str, pathlib.Path] = {}
    for ds_path in datasets:
        resolved = resolve_datasets(ds_path)
        root_label = ds_path.resolve().name  # use the directory itself, not its parent
        for name, csv_path in resolved.items():
            display_name = f"{root_label}/{name}" if len(datasets) > 1 else name
            if display_name in annotation_files:
                LOGGER.warning("Duplicate dataset name '%s', overwriting", display_name)
            annotation_files[display_name] = csv_path
        LOGGER.info("Found %d datasets under %s", len(resolved), ds_path)
    total_datasets = len(annotation_files)

    # Load model once
    if model_type == "onnx":
        loaded_model = load_onnx_model(model_path, device, gpu)
    else:
        loaded_model = load_keras_model(model_path, device, gpu)
    LOGGER.info("Loaded %s model from %s", model_type.upper(), model_path)

    overall_metrics = MetricAccumulator(config)
    all_failures: list[FailureRecord] = []
    group_reports: dict[str, Any] = {}
    total_images = 0
    overall_started = time.perf_counter()

    for group_name, anno_path in annotation_files.items():
        group_metrics, group_failures = process_single_dataset(
            annotation_path=anno_path,
            group_name=group_name,
            config=config,
            loaded_model=loaded_model,
            model_type=model_type,
            batch_size=batch_size,
        )
        group_reports[group_name] = group_metrics
        all_failures.extend(group_failures)
        total_images += group_metrics["samples"]

    overall_seconds = time.perf_counter() - overall_started

    # Compute overall metrics from group reports
    total_correct_plates = sum(g["correct_plates"] for g in group_reports.values())
    total_correct_chars = sum(
        int(g.get("character_accuracy", 0) * g["samples"] * config.max_plate_slots)
        for g in group_reports.values()
    )
    total_correct_lengths = sum(
        int(g.get("plate_length_accuracy", 0) * g["samples"])
        for g in group_reports.values()
    )
    character_slots = total_images * config.max_plate_slots
    total_edit_errors = sum(
        g.get("character_error_rate", 0) * g["samples"] * config.max_plate_slots
        for g in group_reports.values()
    )
    overall_report = {
        "samples": total_images,
        "correct_plates": total_correct_plates,
        "plate_accuracy": total_correct_plates / total_images,
        "character_accuracy": total_correct_chars / character_slots,
        "plate_length_accuracy": total_correct_lengths / total_images,
        "character_error_rate": total_edit_errors / character_slots,
    }

    # Save all failures
    saved_count = save_failures(all_failures, output_dir)

    return {
        "model": str(model_path.resolve()),
        "model_type": model_type,
        "runtime_device": f"cuda:{gpu}" if device == "cuda" else device,
        "plate_config": str(config_path.resolve()),
        "dataset": [str(ds.resolve()) for ds in datasets],
        "output_dir": str(output_dir.resolve()),
        "total_datasets": total_datasets,
        "total_images": total_images,
        "failure_count": saved_count,
        "failure_rate": saved_count / total_images if total_images else 0,
        "elapsed_seconds": overall_seconds,
        "overall": overall_report,
        "groups": group_reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="/zxk/plate_ocr/plate_ocr/fast-plate-ocr/trained_models/fine_tuned/cct_xs_v2_torch/2026-08-07_11-52-11/last.keras",
        type=pathlib.Path,
        help="Path to a .keras or .onnx model",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[pathlib.Path("/zxk/plate_ocr/plate_ocr/asserts/CBLPRD-330K/separate"), 
                pathlib.Path("/zxk/plate_ocr/plate_ocr/asserts/CCPD/fast-plate-ocr"), 
                pathlib.Path("/zxk/plate_ocr/plate_ocr/asserts/CCPD2020/fast-plate-ocr"),
                pathlib.Path("/zxk/plate_ocr/plate_ocr/asserts/challenge_data")],
        type=pathlib.Path,
        help="One or more datasets: CSV files, directories with annotations.csv, "
             "or grouped roots with */annotations.csv",
    )
    parser.add_argument(
        "--plate-config",
        default="/zxk/plate_ocr/plate_ocr/fast-plate-ocr/config/cn_plate_config.yaml",
        type=pathlib.Path,
        help="Path to plate_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="/zxk/plate_ocr/plate_ocr/asserts/fine_challenge_data_xs",
        type=pathlib.Path,
        help="Output directory for failure data in fastplateocr format",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Inference batch size (default: 64)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.gpu < 0 or args.batch_size <= 0:
            raise ValueError("gpu must be non-negative; batch-size must be positive.")

        report = run_processing(
            model=args.model,
            datasets=args.dataset,
            plate_config=args.plate_config,
            output_dir=args.output_dir,
            device=args.device,
            gpu=args.gpu,
            batch_size=args.batch_size,
        )

        # Print summary
        print(f"\n模型: {report['model']}")
        print(f"模型类型: {report['model_type'].upper()}")
        print(f"设备: {report['runtime_device']}")
        print(f"配置: {report['plate_config']}")
        print(f"数据集: {', '.join(report['dataset'])}")
        print(f"数据集数量: {report['total_datasets']}")
        print(f"总图像数: {report['total_images']}")
        print(f"预测失败数: {report['failure_count']}")
        print(f"失败率: {report['failure_rate'] * 100:.2f}%")
        print(f"耗时: {report['elapsed_seconds']:.2f} 秒")

        if report["failure_count"] == 0:
            print("\n没有发现预测失败的数据。")
        else:
            print(f"\n预测失败数据已保存到: {report['output_dir']}")

        # Chinese metrics report
        eval_report = {
            "model": report["model"],
            "model_type": report["model_type"],
            "runtime_device": report["runtime_device"],
            "plate_config": report["plate_config"],
            "overall": report["overall"],
            "groups": report["groups"],
        }
        print(f"\n{format_report_chinese(eval_report)}")

    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
