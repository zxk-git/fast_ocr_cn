#!/usr/bin/env python3
"""Convert CCPD2019 images to grouped fast-plate-ocr datasets."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


PROVINCES = tuple("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新")
ALPHABETS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
ADS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
CCPD_GROUPS = (
    "ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db",
    "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather",
)
DEFAULT_DATASET_ROOT = Path("/zxk/plate_ocr/plate/CCPD/OpenDataLab___CCPD/raw/CCPD2019")
DEFAULT_OUTPUT_DIR = Path("/zxk/plate_ocr/plate/CCPD/fast-plate-ocr")
DEFAULT_WORKERS = 16
JPEG_QUALITY = 95
LOGGER = logging.getLogger("ccpd2fastocr")


class ConversionError(ValueError):
    """Raised when CCPD input cannot be converted safely."""


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


def parse_ccpd_filename(filename: str) -> Annotation:
    """Decode a plate label and ordered polygon from a CCPD image name."""
    path = Path(filename)
    if path.name != filename or path.suffix.lower() != ".jpg":
        raise ConversionError(f"invalid CCPD image name: {filename}")
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ConversionError(f"expected 7 CCPD fields in {filename}")
    raw_points = tuple(_parse_point(item, filename) for item in parts[3].split("_"))
    if len(raw_points) != 4:
        raise ConversionError(f"expected 4 polygon points in {filename}")
    try:
        indexes = tuple(int(item) for item in parts[4].split("_"))
        if len(indexes) != 7:
            raise ValueError
        plate = (
            PROVINCES[indexes[0]]
            + ALPHABETS[indexes[1]]
            + "".join(ADS[index] for index in indexes[2:])
        )
    except (IndexError, ValueError) as error:
        raise ConversionError(f"invalid plate indexes in {filename}") from error
    # CCPD stores right-bottom, left-bottom, left-top, right-top.
    polygon = (raw_points[2], raw_points[3], raw_points[0], raw_points[1])
    if cv2.contourArea(np.asarray(polygon, dtype=np.float32)) <= 0:
        raise ConversionError(f"degenerate polygon in {filename}")
    return Annotation(plate_text=plate, polygon=polygon)


def rectify_plate(
    image: np.ndarray, polygon: tuple[tuple[int, int], ...]
) -> np.ndarray:
    """Perspective-warp one ordered LT/RT/RB/LB polygon at natural size."""
    if image is None or image.ndim < 2 or image.size == 0:
        raise ConversionError("cannot rectify an empty image")
    height, width = image.shape[:2]
    if any(x > width or y > height for x, y in polygon):
        raise ConversionError(f"polygon is outside image bounds {width}x{height}")
    clipped = tuple((min(x, width - 1), min(y, height - 1)) for x, y in polygon)
    points = np.asarray(clipped, dtype=np.float32)
    top, right = np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[1])
    bottom, left = np.linalg.norm(points[2] - points[3]), np.linalg.norm(points[3] - points[0])
    target_width = int(round(max(top, bottom)))
    target_height = int(round(max(left, right)))
    if target_width < 1 or target_height < 1:
        raise ConversionError("polygon produces an empty perspective crop")
    target = np.asarray(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, target)
    crop = cv2.warpPerspective(image, matrix, (target_width, target_height))
    if crop.size == 0:
        raise ConversionError("perspective transform produced an empty crop")
    return crop


def _ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _read_group(dataset_root: Path, group: str) -> list[ImageRow]:
    group_dir = dataset_root / group
    if not group_dir.is_dir():
        raise ConversionError(f"missing CCPD group: {group_dir}")
    paths = sorted(group_dir.glob("*.jpg"), key=lambda path: path.name)
    if not paths:
        raise ConversionError(f"CCPD group contains no JPG images: {group_dir}")
    rows = [ImageRow(path.name, path, parse_ccpd_filename(path.name)) for path in paths]
    if len({row.basename for row in rows}) != len(rows):
        raise ConversionError(f"duplicate image name in CCPD group: {group}")
    return rows


def _convert_image(row: ImageRow, images_dir: Path) -> None:
    image = cv2.imread(str(row.source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConversionError(f"could not decode image: {row.source_path}")
    try:
        if any(x == image.shape[1] or y == image.shape[0] for x, y in row.annotation.polygon):
            LOGGER.warning("clamping inclusive boundary coordinate in %s", row.source_path)
        crop = rectify_plate(image, row.annotation.polygon)
        written = cv2.imwrite(
            str(images_dir / row.basename), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
    except cv2.error as error:
        raise ConversionError(f"OpenCV failed for {row.source_path}: {error}") from error
    if not written:
        raise ConversionError(f"could not write image: {images_dir / row.basename}")


def _write_annotations(rows: Iterable[ImageRow], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"image_path": f"images/{row.basename}", "plate_text": row.annotation.plate_text}
            )


def _materialize_group(rows: list[ImageRow], group_dir: Path, workers: int) -> None:
    images_dir = group_dir / "images"
    images_dir.mkdir(parents=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), workers * 4):
            batch = rows[start : start + workers * 4]
            futures = [executor.submit(_convert_image, row, images_dir) for row in batch]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 1000 == 0 or completed == len(rows):
                    LOGGER.info("converted %d/%d images in %s", completed, len(rows), group_dir.name)
    _write_annotations(rows, group_dir / "annotations.csv")


def _write_config(plate_texts: Iterable[str], output_file: Path) -> tuple[str, list[str]]:
    characters = sorted({character for text in plate_texts for character in text})
    if "_" in characters:
        raise ConversionError("plate alphabet conflicts with padding character '_'")
    alphabet = "".join(characters) + "_"
    lines = [
        "max_plate_slots: 7",
        f'alphabet: "{alphabet}"',
        'pad_char: "_"',
        "img_height: 64",
        "img_width: 128",
        "keep_aspect_ratio: true",
        "interpolation: linear",
        "image_color_mode: rgb",
        "padding_color: 114",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return alphabet, characters


def _publish(work_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        output_dir.rmdir()
    work_dir.replace(output_dir)


def _build_report(
    root: Path, output_dir: Path,
    group_stats: dict[str, dict[str, object]], plate_texts: list[str],
    alphabet: str, characters: list[str], workers: int,
) -> dict[str, object]:
    total = len(plate_texts)
    return {
        "dataset_root": str(root),
        "output_dir": str(output_dir),
        "total_images": total,
        "success_count": total,
        "failure_count": 0,
        "errors": [],
        "included_groups": list(CCPD_GROUPS),
        "excluded_groups": ["ccpd_np"],
        "groups": group_stats,
        "alphabet": alphabet,
        "characters": characters,
        "max_plate_slots": 7,
        "conversion": {
            "crop_mode": "four_point_perspective",
            "output_size": "natural",
            "jpeg_quality": JPEG_QUALITY,
            "workers": workers,
        },
    }


def convert_dataset(
    dataset_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Convert all labelled CCPD2019 groups and atomically publish the result."""
    root = Path(dataset_root)
    output_dir = Path(out_dir) if out_dir is not None else root / "fast-plate-ocr"
    if workers <= 0:
        raise ConversionError("workers must be greater than zero")
    if not root.is_dir():
        raise ConversionError(f"dataset root does not exist: {root}")
    _ensure_output_available(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".fast-plate-ocr-", dir=output_dir.parent))
    group_stats: dict[str, dict[str, object]] = {}
    plate_texts: list[str] = []
    try:
        for group in CCPD_GROUPS:
            rows = _read_group(root, group)
            LOGGER.info("converting %s (%d images)", group, len(rows))
            _materialize_group(rows, work_dir / group, workers)
            plate_texts.extend(row.annotation.plate_text for row in rows)
            group_stats[group] = {
                "source_dir": str(root / group),
                "total_images": len(rows),
                "success_count": len(rows),
                "failure_count": 0,
            }
        alphabet, characters = _write_config(plate_texts, work_dir / "plate_config.yaml")
        report = _build_report(
            root, output_dir, group_stats, plate_texts, alphabet, characters, workers
        )
        (work_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _publish(work_dir, output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_dir = convert_dataset(args.dataset_root, args.out_dir, workers=args.workers)
    except (ConversionError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"Converted dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
