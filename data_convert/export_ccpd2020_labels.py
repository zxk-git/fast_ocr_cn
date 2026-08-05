#!/usr/bin/env python3
"""Export labels from CCPD2020 dataset by parsing filename-embedded annotations.

CCPD2020 green plates have 8 characters (province + letter + 6 alphanumeric),
while standard CCPD2019 plates have 7.  This script handles both formats.

Outputs per-split CSV files with fields:
    image_path, plate_text, x1..y4 (four LT/RT/RB/LB corners),
    bbox_lt_x..bbox_rb_y (axis-aligned detection box),
    tilt_h, tilt_v, brightness, blur

Usage:
    python data_convert/export_ccpd2020_labels.py
    python data_convert/export_ccpd2020_labels.py --dataset-root path/to/CCPD2020
    python data_convert/export_ccpd2020_labels.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Character mapping tables (CCPD convention).
PROVINCES = tuple(
    "皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O"
)
ALPHABETS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZO")
ADS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O")

CSV_FIELDS = [
    "image_path", "plate_text",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "bbox_lt_x", "bbox_lt_y", "bbox_rb_x", "bbox_rb_y",
    "tilt_h", "tilt_v", "brightness", "blur",
]

DEFAULT_DATASET_ROOT = Path(
    "/zxk/plate_ocr/plate_ocr/asserts/CCPD/CCPD2020"
)

LOGGER = logging.getLogger("export_ccpd2020_labels")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class LabelError(ValueError):
    """Raised when a CCPD filename cannot be parsed."""


def parse_ccpd_filename(filename: str) -> dict[str, object]:
    """Decode a CCPD/CCPD2020 filename into label fields.

    Filename fields (separated by ``-``):
        0 – area ratio
        1 – tilt_h_tilt_v
        2 – bbox (left-up & right-bottom)
        3 – four vertices (RB, LB, LT, RT)
        4 – character codes (7 for standard, 8 for green plates)
        5 – brightness
        6 – blur

    Four corners are reordered from CCPD's RB/LB/LT/RT to LT/RT/RB/LB
    for output (standard clockwise from top-left).
    """
    path = Path(filename)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise LabelError(f"not a supported image: {filename}")

    parts = path.stem.split("-")
    if len(parts) != 7:
        raise LabelError(f"expected 7 dash-separated fields, got {len(parts)}: {filename}")

    # --- tilt ---
    try:
        tilt_h, tilt_v = map(int, parts[1].split("_"))
    except ValueError as exc:
        raise LabelError(f"bad tilt field {parts[1]!r} in {filename}") from exc

    # --- detection bbox (left-up, right-bottom) ---
    try:
        bbox_lt, bbox_rb = parts[2].split("_")
        bbox_lt_x, bbox_lt_y = map(int, bbox_lt.split("&"))
        bbox_rb_x, bbox_rb_y = map(int, bbox_rb.split("&"))
    except ValueError as exc:
        raise LabelError(f"bad bbox field {parts[2]!r} in {filename}") from exc

    # --- four vertices (CCPD order: RB, LB, LT, RT) ---
    try:
        raw = [p.split("&") for p in parts[3].split("_")]
        if len(raw) != 4:
            raise ValueError
        rb = (int(raw[0][0]), int(raw[0][1]))
        lb = (int(raw[1][0]), int(raw[1][1]))
        lt = (int(raw[2][0]), int(raw[2][1]))
        rt = (int(raw[3][0]), int(raw[3][1]))
    except (ValueError, IndexError) as exc:
        raise LabelError(f"bad vertices field {parts[3]!r} in {filename}") from exc

    # --- plate number ---
    try:
        char_codes = [int(c) for c in parts[4].split("_")]
        if len(char_codes) not in (7, 8):
            raise LabelError(
                f"expected 7 or 8 character codes, got {len(char_codes)}: {filename}"
            )
        plate = (
            PROVINCES[char_codes[0]]
            + ALPHABETS[char_codes[1]]
            + "".join(ADS[i] for i in char_codes[2:])
        )
    except (ValueError, IndexError) as exc:
        raise LabelError(f"bad character code field in {filename}") from exc

    # --- brightness & blur ---
    try:
        brightness = int(parts[5])
        blurriness = int(parts[6])
    except ValueError as exc:
        raise LabelError(f"bad brightness/blur in {filename}") from exc

    return {
        "plate_text": plate,
        "x1": lt[0], "y1": lt[1],
        "x2": rt[0], "y2": rt[1],
        "x3": rb[0], "y3": rb[1],
        "x4": lb[0], "y4": lb[1],
        "bbox_lt_x": bbox_lt_x, "bbox_lt_y": bbox_lt_y,
        "bbox_rb_x": bbox_rb_x, "bbox_rb_y": bbox_rb_y,
        "tilt_h": tilt_h, "tilt_v": tilt_v,
        "brightness": brightness, "blur": blurriness,
    }


def _process_one(image_path: Path) -> dict[str, object]:
    """Parse a single image filename and return a CSV row dict."""
    row = parse_ccpd_filename(image_path.name)
    row["image_path"] = image_path.name  # filename only, relative to split dir
    return row


def _discover_splits(dataset_root: Path) -> dict[str, Path]:
    """Find train/test/val split directories under the dataset root.

    Supports both flat (ccpd_green/train/) and nested structures.
    Returns ``{split_name: split_dir}``.
    """
    # First try: flat split dirs under root (ccpd2020/ccpd_green/train/)
    if (dataset_root / "ccpd_green").is_dir():
        root = dataset_root / "ccpd_green"
    else:
        root = dataset_root

    splits: dict[str, Path] = {}
    for name in ("train", "test", "val"):
        candidate = root / name
        if candidate.is_dir():
            splits[name] = candidate
    return splits


def _collect_images(split_dir: Path) -> list[Path]:
    """Return sorted list of image paths under *split_dir*."""
    paths: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        paths.extend(sorted(split_dir.glob(f"*{suffix}")))
        paths.extend(sorted(split_dir.glob(f"*{suffix.upper()}")))
    # Deduplicate and sort.
    return sorted(set(paths), key=lambda p: p.name)


def export_split(
    split_dir: Path,
    output_csv: Path,
    workers: int = 1,
) -> tuple[int, int]:
    """Export labels for all images in *split_dir* to *output_csv*.

    Returns ``(success_count, error_count)``.
    """
    images = _collect_images(split_dir)
    if not images:
        LOGGER.warning("no images found in %s", split_dir)
        return 0, 0

    rows: list[dict[str, object]] = [{}] * len(images)
    success = 0
    errors = 0

    if workers <= 1:
        for i, img in enumerate(images):
            try:
                rows[i] = _process_one(img)
                success += 1
            except LabelError as exc:
                LOGGER.warning("skipping %s: %s", img.name, exc)
                errors += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_process_one, img): i
                for i, img in enumerate(images)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rows[idx] = future.result()
                    success += 1
                except LabelError as exc:
                    LOGGER.warning("skipping %s: %s", images[idx].name, exc)
                    errors += 1

    # Write only successful rows, in original order.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            if row.get("plate_text"):  # skip failed rows (empty dict)
                writer.writerow(row)

    return success, errors


def export_dataset(
    dataset_root: Path | str,
    workers: int = 1,
) -> dict[str, tuple[int, int]]:
    """Export all CCPD2020 splits to per-split CSV files.

    Output files are written next to each split directory:
        ``<split_dir>/labels.csv``
    """
    root = Path(dataset_root)
    splits = _discover_splits(root)
    if not splits:
        raise LabelError(f"no train/test/val splits found under {root}")

    results: dict[str, tuple[int, int]] = {}
    for split_name, split_dir in sorted(splits.items()):
        output_csv = split_dir / "labels.csv"
        LOGGER.info("exporting %s (%s) -> %s", split_name, split_dir, output_csv)
        success, errors = export_split(split_dir, output_csv, workers=workers)
        results[split_name] = (success, errors)
        LOGGER.info(
            "  %s: %d labels exported, %d skipped",
            split_name, success, errors,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
        help="Path to the CCPD2020 dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers for parsing (default: 1)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        results = export_dataset(args.dataset_root, workers=args.workers)
    except LabelError as exc:
        parser.error(str(exc))

    total_success = sum(s for s, _ in results.values())
    total_errors = sum(e for _, e in results.values())
    LOGGER.info(
        "done: %d labels exported, %d skipped across %d splits",
        total_success, total_errors, len(results),
    )
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
