#!/usr/bin/env python3
"""Convert CCPD2020 (green plate / new energy vehicle) dataset to fast-plate-ocr format.

CCPD2020 holds 8-character green license plates in a train/test/val split under
``ccpd_green/``.  This script:

1. Parses filename-embedded annotations (polygon + plate text)
2. Perspective-crops each plate to a rectified image
3. Writes per-split ``images/`` + ``annotations.csv``
4. Copies the existing ``cn_plate_config.yaml`` (max_slots=8) into the output
5. Publishes atomically via a temp directory

Usage::

    python data_convert/CCPD20202Fastocr.py
    python data_convert/CCPD20202Fastocr.py --workers 8
    python data_convert/CCPD20202Fastocr.py --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Ensure the project root is importable regardless of how this script is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from data_convert.CCPD2Fastocr import (
    ConversionError,
    PROVINCES,
    ALPHABETS,
    ADS,
    rectify_plate,
)

DEFAULT_DATASET_ROOT = Path(
    "/zxk/plate_ocr/plate_ocr/asserts/CCPD/CCPD2020"
)
DEFAULT_PLATE_CONFIG = _PROJECT_ROOT / "config" / "cn_plate_config.yaml"
DEFAULT_WORKERS = 16
JPEG_QUALITY = 95
LOGGER = logging.getLogger("ccpd20202fastocr")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Annotation:
    plate_text: str
    polygon: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ImageRow:
    basename: str
    source_path: Path
    annotation: Annotation


def _parse_point(value: str, filename: str) -> tuple[int, int]:
    try:
        parts = value.split("&")
        if len(parts) != 2:
            raise ValueError
        point = (int(parts[0]), int(parts[1]))
    except ValueError as error:
        raise ConversionError(f"invalid point {value!r} in {filename}") from error
    if point[0] < 0 or point[1] < 0:
        raise ConversionError(f"negative point {value!r} in {filename}")
    return point


def parse_ccpd2020_filename(filename: str) -> Annotation:
    """Decode a CCPD2020 green-plate filename into plate text and ordered polygon.

    CCPD2020 green plates have **8** characters (province + letter + 6 alphanumeric).
    The CCPD polygon order is RB / LB / LT / RT; we reorder to LT / RT / RB / LB
    for perspective warping.
    """
    path = Path(filename)
    if path.name != filename or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ConversionError(f"invalid CCPD image name: {filename}")
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ConversionError(f"expected 7 CCPD fields in {filename}")

    raw_points = tuple(_parse_point(item, filename) for item in parts[3].split("_"))
    if len(raw_points) != 4:
        raise ConversionError(f"expected 4 polygon points in {filename}")

    try:
        indexes = tuple(int(item) for item in parts[4].split("_"))
        if len(indexes) != 8:
            raise ConversionError(
                f"expected 8 character codes for green plate, got {len(indexes)}: {filename}"
            )
        plate = (
            PROVINCES[indexes[0]]
            + ALPHABETS[indexes[1]]
            + "".join(ADS[index] for index in indexes[2:])
        )
    except (IndexError, ValueError) as error:
        raise ConversionError(f"invalid plate indexes in {filename}") from error

    # CCPD order: right-bottom, left-bottom, left-top, right-top.
    # Reorder to: left-top, right-top, right-bottom, left-bottom.
    polygon = (raw_points[2], raw_points[3], raw_points[0], raw_points[1])
    if cv2.contourArea(np.asarray(polygon, dtype=np.float32)) <= 0:
        raise ConversionError(f"degenerate polygon in {filename}")

    return Annotation(plate_text=plate, polygon=polygon)


def _discover_source_root(dataset_root: Path) -> Path:
    """Resolve the directory holding train/test/val splits.

    Supports ``ccpd_green/`` nesting (the standard CCPD2020 layout).
    """
    green = dataset_root / "ccpd_green"
    if green.is_dir():
        return green
    return dataset_root


def _read_split(root: Path, split: str) -> list[ImageRow]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise ConversionError(f"missing CCPD2020 split: {split_dir}")
    paths: set[Path] = set()
    for suffix in IMAGE_SUFFIXES:
        paths.update(split_dir.glob(f"*{suffix}"))
        paths.update(split_dir.glob(f"*{suffix.upper()}"))
    if not paths:
        raise ConversionError(f"CCPD2020 split contains no images: {split_dir}")
    sorted_paths = sorted(paths, key=lambda p: p.name)
    rows = [ImageRow(p.name, p, parse_ccpd2020_filename(p.name)) for p in sorted_paths]
    if len({row.basename for row in rows}) != len(rows):
        raise ConversionError(f"duplicate image name in split: {split}")
    return rows


def _convert_image(row: ImageRow, images_dir: Path) -> None:
    image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConversionError(f"could not decode image: {row.source_path}")
    try:
        height, width = image.shape[:2]
        clamped = tuple(
            (min(max(0, x), width - 1), min(max(0, y), height - 1))
            for x, y in row.annotation.polygon
        )
        if clamped != row.annotation.polygon:
            LOGGER.warning(
                "clamping out-of-bounds coordinate in %s: %s -> %s",
                row.source_path, row.annotation.polygon, clamped,
            )
        crop = rectify_plate(image, clamped)
        written = cv2.imwrite(
            str(images_dir / row.basename),
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
    except cv2.error as error:
        raise ConversionError(
            f"OpenCV failed for {row.source_path}: {error}"
        ) from error
    if not written:
        raise ConversionError(f"could not write image: {images_dir / row.basename}")


def _write_annotations(rows: list[ImageRow], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": f"images/{row.basename}",
                    "plate_text": row.annotation.plate_text,
                }
            )


def _materialize_split(rows: list[ImageRow], split_dir: Path, workers: int) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    completed = 0
    batch_size = workers * 4
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            futures = {
                executor.submit(_convert_image, row, images_dir): row
                for row in batch
            }
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 1000 == 0 or completed == len(rows):
                    LOGGER.info(
                        "converted %d/%d images in %s",
                        completed, len(rows), split_dir.name,
                    )
    _write_annotations(rows, split_dir / "annotations.csv")


def _write_report(
    report_path: Path,
    root: Path,
    output_dir: Path,
    split_stats: dict[str, dict[str, object]],
    plate_texts: list[str],
    workers: int,
) -> None:
    total = len(plate_texts)
    report: dict[str, object] = {
        "dataset_root": str(root),
        "output_dir": str(output_dir),
        "total_images": total,
        "success_count": total,
        "failure_count": 0,
        "errors": [],
        "max_plate_slots": 8,
        "included_splits": sorted(split_stats.keys()),
        "splits": split_stats,
        "conversion": {
            "crop_mode": "four_point_perspective",
            "output_size": "natural",
            "jpeg_quality": JPEG_QUALITY,
            "workers": workers,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def convert_dataset(
    dataset_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
    plate_config: Path | str = DEFAULT_PLATE_CONFIG,
) -> Path:
    """Convert CCPD2020 to fast-plate-ocr format, publishing atomically.

    Returns the path to the output directory.
    """
    root = Path(dataset_root)
    output_dir = Path(out_dir) if out_dir is not None else root / "fast-plate-ocr"
    config_path = Path(plate_config)

    if workers <= 0:
        raise ConversionError("workers must be greater than zero")
    if not root.is_dir():
        raise ConversionError(f"dataset root does not exist: {root}")
    if not config_path.is_file():
        raise ConversionError(f"plate config not found: {config_path}")

    source_root = _discover_source_root(root)
    _ensure_output_available(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=".fast-plate-ocr-", dir=output_dir.parent)
    )

    split_stats: dict[str, dict[str, object]] = {}
    plate_texts: list[str] = []

    try:
        # Merge train + test into a single train split.
        train_rows = _read_split(source_root, "train") + _read_split(source_root, "test")
        val_rows = _read_split(source_root, "val")

        for split_name, rows in [("train", train_rows), ("val", val_rows)]:
            LOGGER.info("converting %s (%d images)", split_name, len(rows))
            _materialize_split(rows, work_dir / split_name, workers)
            plate_texts.extend(row.annotation.plate_text for row in rows)
            split_stats[split_name] = {
                "source_dir": str(source_root),
                "total_images": len(rows),
                "success_count": len(rows),
                "failure_count": 0,
            }

        # Copy plate config into the output.
        shutil.copyfile(config_path, work_dir / "plate_config.yaml")

        # Write conversion report into work_dir (survives the rename).
        _write_report(
            work_dir / "conversion_report.json",
            root, output_dir, split_stats, plate_texts, workers,
        )

        # Atomic publish.
        work_dir.replace(output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to CCPD2020 dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: {dataset-root}/ccpd_green-fast-plate-ocr)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel threads (default: %(default)s)",
    )
    parser.add_argument(
        "--plate-config",
        type=Path,
        default=DEFAULT_PLATE_CONFIG,
        help="Path to plate_config.yaml (default: config/cn_plate_config.yaml)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_dir = convert_dataset(
            args.dataset_root,
            args.out_dir,
            workers=args.workers,
            plate_config=args.plate_config,
        )
    except (ConversionError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"Converted dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
