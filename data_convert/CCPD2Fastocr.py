#!/usr/bin/env python3
"""Convert CCPD2019 images to grouped fast-plate-ocr datasets with train/val split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import tempfile
from collections import Counter
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
DEFAULT_WORKERS = 16
DEFAULT_VAL_RATIO = 0.2
JPEG_QUALITY = 95
LOGGER = logging.getLogger("ccpd2fastocr")

# Province abbreviation -> full name mapping
PROVINCE_MAP: dict[str, str] = {
    "京": "北京", "津": "天津", "沪": "上海", "渝": "重庆",
    "冀": "河北", "晋": "山西", "蒙": "内蒙古", "辽": "辽宁",
    "吉": "吉林", "黑": "黑龙江", "苏": "江苏", "浙": "浙江",
    "皖": "安徽", "闽": "福建", "赣": "江西", "鲁": "山东",
    "豫": "河南", "鄂": "湖北", "湘": "湖南", "粤": "广东",
    "桂": "广西", "琼": "海南", "川": "四川", "贵": "贵州",
    "云": "云南", "藏": "西藏", "陕": "陕西", "甘": "甘肃",
    "青": "青海", "宁": "宁夏", "新": "新疆",
}


def _province_from_plate(plate_text: str) -> str:
    if not plate_text:
        return "未知"
    return PROVINCE_MAP.get(plate_text[0], "其他")


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


# ---------------------------------------------------------------------------
# CCPD filename parsing
# ---------------------------------------------------------------------------


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
    top = np.linalg.norm(points[1] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
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


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def _split_train_val(rows: list[ImageRow], val_ratio: float) -> tuple[list[ImageRow], list[ImageRow]]:
    """Split rows into train/val deterministically using basename SHA-256 hash."""
    if val_ratio <= 0:
        return rows, []
    sorted_rows = sorted(rows, key=lambda r: hashlib.sha256(r.basename.encode()).hexdigest())
    val_count = max(1, int(len(sorted_rows) * val_ratio))
    return sorted_rows[val_count:], sorted_rows[:val_count]


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------


def _ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _read_group(dataset_root: Path, group: str) -> list[ImageRow]:
    group_dir = dataset_root / group
    if not group_dir.is_dir():
        raise ConversionError(f"missing CCPD group: {group_dir}")
    paths = sorted(group_dir.glob("*.jpg"), key=lambda p: p.name)
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


def _materialize_group(
    rows: list[ImageRow], group_dir: Path, workers: int, val_ratio: float
) -> tuple[dict[str, int], list[ImageRow], list[ImageRow]]:
    """Perspective-crop images for one CCPD group, splitting into train/val subdirectories.

    Returns:
        (counts, train_rows, val_rows) — the actual rows assigned to each split.
    """
    train, val = _split_train_val(rows, val_ratio)
    counts: dict[str, int] = {}

    for split_name, split_rows in (("train", train), ("val", val)):
        if not split_rows:
            continue
        images_dir = group_dir / split_name / "images"
        images_dir.mkdir(parents=True)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(split_rows), workers * 4):
                batch = split_rows[start : start + workers * 4]
                futures = [executor.submit(_convert_image, r, images_dir) for r in batch]
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed % 1000 == 0 or completed == len(split_rows):
                        LOGGER.info("  %s/%s converted %d/%d images",
                                    group_dir.name, split_name, completed, len(split_rows))
        _write_annotations(split_rows, group_dir / split_name / "annotations.csv")
        counts[split_name] = len(split_rows)

    return counts, train, val


# ---------------------------------------------------------------------------
# Config and report writing
# ---------------------------------------------------------------------------


def _write_config(plate_texts: Iterable[str], output_file: Path) -> tuple[str, list[str], int]:
    characters = sorted({c for text in plate_texts for c in text})
    if "_" in characters:
        raise ConversionError("plate alphabet conflicts with padding character '_'")
    alphabet = "".join(characters) + "_"
    max_slots = max(len(t) for t in plate_texts)
    lines = [
        f"max_plate_slots: {max_slots}",
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
    return alphabet, characters, max_slots


def _publish(work_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        output_dir.rmdir()
    work_dir.replace(output_dir)


def _distribution(items: Iterable[str], key=None) -> dict[str, int]:
    if key:
        items = (key(item) for item in items)
    return dict(sorted(Counter(items).items()))


def _build_report(
    root: Path,
    output_dir: Path,
    group_stats: dict[str, dict[str, object]],
    all_plates: list[str],
    train_plates: list[str],
    val_plates: list[str],
    alphabet: str,
    characters: list[str],
    max_slots: int,
    workers: int,
    val_ratio: float,
) -> dict[str, object]:
    total = len(all_plates)
    return {
        "dataset_root": str(root),
        "output_dir": str(output_dir),
        "total_images": total,
        "train_count": len(train_plates),
        "val_count": len(val_plates),
        "val_ratio": val_ratio,
        "split_method": "sha256_deterministic",
        "success_count": total,
        "failure_count": 0,
        "errors": [],
        "included_groups": list(CCPD_GROUPS),
        "excluded_groups": ["ccpd_np"],
        "groups": group_stats,
        "alphabet": alphabet,
        "characters": characters,
        "max_plate_slots": max_slots,
        "plate_length_distribution": _distribution(all_plates, key=len),
        "province_distribution": _distribution(all_plates, key=_province_from_plate),
        "group_distribution": {
            g: s.get("source_count", 0) for g, s in group_stats.items()
        },
        "conversion": {
            "crop_mode": "four_point_perspective",
            "output_size": "natural_perspective",
            "jpeg_quality": JPEG_QUALITY,
            "workers": workers,
        },
    }


def _print_summary(output_dir: Path) -> None:
    report_path = output_dir / "conversion_report.json"
    if not report_path.is_file():
        return
    with report_path.open("r", encoding="utf-8") as f:
        r = json.load(f)

    total = r.get("total_images", 0)
    train_cnt = r.get("train_count", 0)
    val_cnt = r.get("val_count", 0)
    val_ratio_report = r.get("val_ratio", 0)

    print()
    print("=" * 60)
    print("  CCPD 数据转换统计报告")
    print("=" * 60)
    print(f"\n  📊 总体统计")
    print(f"     输入图片总数        : {total}")
    print(f"     训练集              : {train_cnt} ({train_cnt/total*100:.1f}%)" if total else "")
    print(f"     验证集              : {val_cnt} ({val_cnt/total*100:.1f}%)" if total else "")
    print(f"     验证集比例          : {val_ratio_report}")

    # Group distribution
    groups = r.get("group_distribution", {})
    if groups:
        print(f"\n  📁 CCPD 分组统计")
        for g in CCPD_GROUPS:
            if g in groups:
                gs = r["groups"].get(g, {})
                t = gs.get("train_count", 0)
                v = gs.get("val_count", 0)
                print(f"     {g:<18}: 总计={groups[g]}, train={t}, val={v}")

    _print_distribution_table(r, "plate_length_distribution", "📏 车牌位数分布", "位数")
    _print_distribution_table(r, "province_distribution", "🗺️  省份分布统计", "省份")

    print("=" * 60)


def _print_distribution_table(report: dict, key: str, title: str, label: str, top_n: int | None = None) -> None:
    data = report.get(key, {})
    if not data:
        return
    items = list(data.items())
    if top_n and len(items) > top_n:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n  {title}")
    total = sum(v for _, v in items)
    for name, count in items:
        pct = count / total * 100 if total > 0 else 0.0
        print(f"     {str(name):<12} : {count:>7}  ({pct:5.1f}%)")
    if top_n and len(data) > top_n:
        print(f"     ... 共 {len(data)} 种，以上为 Top {top_n}")


# ---------------------------------------------------------------------------
# Main conversion pipeline
# ---------------------------------------------------------------------------


def convert_dataset(
    dataset_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    workers: int = DEFAULT_WORKERS,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Path:
    """Convert all labelled CCPD2019 groups with train/val split."""
    root = Path(dataset_root)
    output_dir = Path(out_dir) if out_dir is not None else root / "fast-plate-ocr"
    if workers <= 0:
        raise ConversionError("workers must be greater than zero")
    if not 0 <= val_ratio < 1:
        raise ConversionError("val_ratio must be in [0, 1)")
    if not root.is_dir():
        raise ConversionError(f"dataset root does not exist: {root}")
    _ensure_output_available(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".fast-plate-ocr-", dir=output_dir.parent))

    group_stats: dict[str, dict[str, object]] = {}
    all_plates: list[str] = []
    train_plates: list[str] = []
    val_plates: list[str] = []

    try:
        for group in CCPD_GROUPS:
            rows = _read_group(root, group)
            LOGGER.info("converting %s (%d images)", group, len(rows))
            counts, train_rows, val_rows = _materialize_group(rows, work_dir / group, workers, val_ratio)

            all_plates.extend(r.annotation.plate_text for r in rows)
            train_plates.extend(r.annotation.plate_text for r in train_rows)
            val_plates.extend(r.annotation.plate_text for r in val_rows)

            group_stats[group] = {
                "source_dir": str(root / group),
                "source_count": len(rows),
                "train_count": counts.get("train", 0),
                "val_count": counts.get("val", 0),
            }
        report = _build_report(
            root, output_dir, group_stats, all_plates, train_plates, val_plates,
            alphabet, characters, max_slots, workers, val_ratio,
        )
        (work_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _publish(work_dir, output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    _print_summary(output_dir)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="CCPD2019 数据集根目录")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数 (默认 {DEFAULT_WORKERS})")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help=f"验证集比例 (默认 {DEFAULT_VAL_RATIO})")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_dir = convert_dataset(args.dataset_root, args.out_dir, workers=args.workers, val_ratio=args.val_ratio)
    except (ConversionError, FileExistsError, OSError) as error:
        parser.error(str(error))
    print(f"Converted dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
