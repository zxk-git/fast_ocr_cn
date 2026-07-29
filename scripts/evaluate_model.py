#!/usr/bin/env python3
"""Evaluate Keras(torch) or ONNX fast-plate-ocr models on labeled datasets."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import pathlib
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from tqdm.auto import tqdm

if __package__:
    from scripts import test_single_image as inference
    from scripts.evaluation_metrics import MetricAccumulator, format_report_chinese
else:
    import test_single_image as inference
    from evaluation_metrics import MetricAccumulator, format_report_chinese


Device = Literal["auto", "cpu", "cuda"]
LOGGER = logging.getLogger("evaluate_model")


@dataclass(frozen=True)
class Record:
    image_path: pathlib.Path
    plate_text: str
    group: str


@dataclass(frozen=True)
class Runtime:
    model_type: str
    device: str
    predict: Callable[[np.ndarray], np.ndarray]
    synchronize: Callable[[], None]


def _noop() -> None:
    pass


def resolve_datasets(dataset: pathlib.Path) -> dict[str, pathlib.Path]:
    """Resolve one CSV, one dataset directory, or immediate grouped datasets."""
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
    grouped = {item.parent.name: item.resolve() for item in sorted(path.glob("*/annotations.csv"))}
    if not grouped:
        raise FileNotFoundError(f"No annotations.csv found in dataset: {path}")
    return grouped


def load_records(annotations: pathlib.Path, group: str, config: Any) -> list[Record]:
    records: list[Record] = []
    with annotations.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image_path", "plate_text"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing image_path or plate_text column: {annotations}")
        for line_number, row in enumerate(reader, 2):
            image_path = pathlib.Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = annotations.parent / image_path
            plate_text = row["plate_text"]
            if not image_path.is_file():
                raise ValueError(f"{annotations}:{line_number}: image not found: {image_path}")
            if len(plate_text) > config.max_plate_slots:
                raise ValueError(f"{annotations}:{line_number}: plate exceeds max_plate_slots")
            invalid = sorted(set(plate_text) - set(config.alphabet))
            if invalid:
                raise ValueError(f"{annotations}:{line_number}: characters outside plate alphabet: {invalid}")
            records.append(Record(image_path.resolve(), plate_text, group))
    if not records:
        raise ValueError(f"Dataset contains no annotation rows: {annotations}")
    return records


def onnx_providers(device: Device, gpu: int, available: list[str]) -> list[Any]:
    if device == "cpu":
        if "CPUExecutionProvider" not in available:
            raise RuntimeError(f"CPUExecutionProvider is unavailable: {available}")
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": gpu}), "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError(f"CUDAExecutionProvider is unavailable: {available}")
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(f"No supported ONNX execution provider: {available}")
    return ["CPUExecutionProvider"]


def load_onnx_runtime(model_path: pathlib.Path, device: Device, gpu: int) -> Runtime:
    ort = importlib.import_module("onnxruntime")
    providers = onnx_providers(device, gpu, ort.get_available_providers())
    session = ort.InferenceSession(str(model_path), providers=providers)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if not inputs or not outputs:
        raise ValueError("ONNX model must expose at least one input and output.")
    model_input = inputs[0]
    output_names = [output.name for output in outputs]
    fetch_names = ["plate"] if "plate" in output_names else [output_names[0]]

    def predict(batch: np.ndarray) -> np.ndarray:
        values = inference.cast_onnx_input(batch, model_input.type)
        return inference.plate_output(session.run(fetch_names, {model_input.name: values}))

    active = session.get_providers()[0]
    return Runtime("onnx", f"{active}:gpu={gpu}" if "CUDA" in active else active, predict, _noop)


def load_keras_runtime(model_path: pathlib.Path, device: Device, gpu: int) -> Runtime:
    os.environ["KERAS_BACKEND"] = "torch"
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch = importlib.import_module("torch")
    use_cuda = device != "cpu" and bool(torch.cuda.is_available())
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA was requested but PyTorch reports it unavailable.")
    keras = importlib.import_module("keras")
    importlib.import_module("fast_plate_ocr.train.model.layers")
    model = keras.models.load_model(model_path, compile=False)

    def predict(batch: np.ndarray) -> np.ndarray:
        return inference.plate_output(model(batch, training=False))

    synchronize = torch.cuda.synchronize if use_cuda else _noop
    return Runtime("keras", f"cuda:{gpu}" if use_cuda else "cpu", predict, synchronize)


def load_runtime(model_path: pathlib.Path, device: Device, gpu: int) -> Runtime:
    selected = inference.model_type(model_path)
    if selected == "keras":
        return load_keras_runtime(model_path, device, gpu)
    return load_onnx_runtime(model_path, device, gpu)


def _load_batch(records: list[Record], config: Any, pool: ThreadPoolExecutor) -> np.ndarray:
    process = importlib.import_module("fast_plate_ocr.core.process")

    def load(record: Record) -> np.ndarray:
        return process.read_and_resize_plate_image(
            record.image_path,
            img_height=config.img_height,
            img_width=config.img_width,
            image_color_mode=config.image_color_mode,
            keep_aspect_ratio=config.keep_aspect_ratio,
            interpolation_method=config.interpolation,
            padding_color=config.padding_color,
        )

    return process.preprocess_image(np.stack(list(pool.map(load, records))))


def _predict(runtime: Runtime, batch: np.ndarray) -> tuple[np.ndarray, float]:
    runtime.synchronize()
    started = time.perf_counter()
    output = runtime.predict(batch)
    runtime.synchronize()
    return output, time.perf_counter() - started


def evaluate_groups(
    groups: dict[str, list[Record]],
    config: Any,
    runtime: Runtime,
    *,
    batch_size: int,
    workers: int,
    show_progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overall = MetricAccumulator(config)
    reports: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        first_records = next(iter(groups.values()))[:batch_size]
        warmup = _load_batch(first_records, config, pool)
        runtime.predict(warmup)
        runtime.synchronize()
        overall_started = time.perf_counter()
        for group, records in groups.items():
            group_metrics = MetricAccumulator(config)
            group_started = time.perf_counter()
            progress = tqdm(
                total=len(records), desc=group, unit="img",
                disable=not show_progress, dynamic_ncols=True,
            )
            for start in range(0, len(records), batch_size):
                selected = records[start : start + batch_size]
                batch = _load_batch(selected, config, pool)
                output, elapsed = _predict(runtime, batch)
                truths = [record.plate_text for record in selected]
                group_metrics.update(output, truths, elapsed)
                overall.update(output, truths, elapsed)
                progress.update(len(selected))
            progress.close()
            reports[group] = group_metrics.report(time.perf_counter() - group_started)
        overall_seconds = time.perf_counter() - overall_started
    return overall.report(overall_seconds), reports


def run_evaluation(
    *, model: pathlib.Path, dataset: pathlib.Path, plate_config: pathlib.Path | None,
    device: Device, gpu: int, batch_size: int, workers: int, show_progress: bool,
) -> dict[str, Any]:
    model_path = inference.existing_file(model, "Model")
    config_path = inference.resolve_plate_config(model_path, plate_config)
    config_module = importlib.import_module("fast_plate_ocr.inference.config")
    config = config_module.PlateConfig.from_yaml(config_path)
    annotation_files = resolve_datasets(dataset)
    groups: dict[str, list[Record]] = {}
    for name, path in annotation_files.items():
        groups[name] = load_records(path, name, config)
        LOGGER.info("loaded %d annotations from %s", len(groups[name]), path)
    runtime = load_runtime(model_path, device, gpu)
    overall, group_reports = evaluate_groups(
        groups, config, runtime, batch_size=batch_size, workers=workers,
        show_progress=show_progress,
    )
    return {
        "model": str(model_path),
        "model_type": runtime.model_type,
        "runtime_device": runtime.device,
        "plate_config": str(config_path),
        "datasets": {name: str(path) for name, path in annotation_files.items()},
        "settings": {
            "batch_size": batch_size, "workers": workers,
            "gpu": gpu, "show_progress": show_progress,
        },
        "overall": overall,
        "groups": group_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/best.keras",
        type=pathlib.Path, help="Path to a .keras or .onnx model",
    )
    parser.add_argument(
        "--dataset", default='/zxk/plate_ocr/plate/CCPD/fast-plate-ocr', type=pathlib.Path,
        help="Annotations CSV, its directory, or a root containing grouped datasets",
    )
    parser.add_argument(
        "--plate-config", type=pathlib.Path, default="config/cn_plate_config.yaml",
        help="Defaults to plate_config.yaml beside the model",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index used by torch or ONNX Runtime")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--workers", type=int, default=8, help="Parallel image loading threads")
    parser.add_argument("--output", type=pathlib.Path,default='plate/fast-plate-ocr/trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/ccpd_evaluate_result.json', help="Optional JSON report path")
    parser.add_argument("--no-progress", action="store_true", help="Disable image progress bars")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.gpu < 0 or args.batch_size <= 0 or args.workers <= 0:
            raise ValueError("gpu must be non-negative; batch-size and workers must be positive.")
        report = run_evaluation(
            model=args.model, dataset=args.dataset, plate_config=args.plate_config,
            device=args.device, gpu=args.gpu, batch_size=args.batch_size, workers=args.workers,
            show_progress=not args.no_progress,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(format_report_chinese(report))
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
