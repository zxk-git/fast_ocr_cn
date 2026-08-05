#!/usr/bin/env python3
"""Merge CBLPRD, CCPD2019, and CCPD2020 datasets for PyTorch training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CBLPRD_ROOT = Path("/zxk/plate_ocr/plate/CBLPRD-330K/fast-plate-ocr")
DEFAULT_CCPD_ROOT = Path("/zxk/plate_ocr/plate/CCPD/fast-plate-ocr")
DEFAULT_CCPD2020_ROOT = Path("/zxk/plate_ocr/plate_ocr/asserts/CCPD/CCPD2020/fast-plate-ocr")
DEFAULT_PLATE_CONFIG = PROJECT_ROOT / "config" / "cn_plate_config.yaml"
DEFAULT_DATASET_RATIO = 1.0
DEFAULT_CCPD_VAL_RATIO = 0.2
CCPD_GROUPS = (
    "ccpd_base", "ccpd_challenge", "ccpd_db", "ccpd_fn",
    "ccpd_rotate", "ccpd_tilt", "ccpd_weather",
)
EXCLUDED_CCPD_GROUPS = ("ccpd_blur",)
LOGGER = logging.getLogger("merge_cblprd_ccpd")


class ConversionError(ValueError):
    """Raised when converted source data cannot be safely merged."""


@dataclass(frozen=True)
class SourceRow:
    basename: str
    source_path: Path
    plate_text: str


@dataclass(frozen=True)
class MergedRow:
    output_name: str
    source_path: Path
    plate_text: str
    source_group: str


def _ensure_output_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _image_basename(value: str, annotations_file: Path, line_number: int) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "images":
        raise ConversionError(f"{annotations_file}:{line_number}: invalid image_path {value!r}")
    basename = path.name
    if basename in {"", ".", ".."} or any(part in {".", ".."} for part in path.parts):
        raise ConversionError(f"{annotations_file}:{line_number}: unsafe image_path {value!r}")
    return basename


def _read_annotations(dataset_dir: Path) -> list[SourceRow]:
    annotations_file = dataset_dir / "annotations.csv"
    if not annotations_file.is_file():
        raise ConversionError(f"missing annotations file: {annotations_file}")
    rows: list[SourceRow] = []
    with annotations_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_path", "plate_text"]:
            raise ConversionError(f"{annotations_file}: expected image_path,plate_text header")
        for line_number, row in enumerate(reader, start=2):
            basename = _image_basename(row.get("image_path") or "", annotations_file, line_number)
            plate_text = row.get("plate_text") or ""
            source_path = dataset_dir / "images" / basename
            if not plate_text or "\n" in plate_text or "\r" in plate_text:
                raise ConversionError(f"{annotations_file}:{line_number}: invalid plate_text")
            if not source_path.is_file():
                raise ConversionError(f"{annotations_file}:{line_number}: missing image {source_path}")
            rows.append(SourceRow(basename, source_path, plate_text))
    if not rows:
        raise ConversionError(f"annotations file is empty: {annotations_file}")
    if len({row.basename for row in rows}) != len(rows):
        raise ConversionError(f"duplicate image_path entries in {annotations_file}")
    return rows


def _prefixed_rows(rows: list[SourceRow], source_group: str) -> list[MergedRow]:
    prefix = source_group.replace("/", "_")
    return [MergedRow(f"{prefix}__{row.basename}", row.source_path, row.plate_text, source_group) for row in rows]


def _rank_rows(rows: list[SourceRow], namespace: str) -> list[SourceRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{namespace}/{row.basename}".encode()).digest())


def _select_rows(rows: list[SourceRow], source_group: str, ratio: float) -> list[SourceRow]:
    if ratio == 0:
        return []
    selected_count = max(1, min(len(rows), round(len(rows) * ratio)))
    selected_names = {row.basename for row in _rank_rows(rows, f"selection/{source_group}")[:selected_count]}
    return [row for row in rows if row.basename in selected_names]


def _partition_group_rows(
    rows: list[SourceRow], group: str, val_ratio: float
) -> tuple[list[SourceRow], list[SourceRow]]:
    if len(rows) < 2:
        raise ConversionError(f"{group} needs at least two samples for train and val")
    val_count = max(1, min(len(rows) - 1, round(len(rows) * val_ratio)))
    ranked = _rank_rows(rows, group)
    val_names = {row.basename for row in ranked[:val_count]}
    train_rows = [row for row in rows if row.basename not in val_names]
    val_rows = [row for row in rows if row.basename in val_names]
    return train_rows, val_rows


def _read_split_dataset(root: Path, dataset: str, ratio: float) -> tuple[list[MergedRow], list[MergedRow]]:
    selected_splits = []
    for split in ("train", "val"):
        source_group = f"{dataset}__{split}"
        source_rows = _read_annotations(root / split)
        selected = _select_rows(source_rows, source_group, ratio)
        LOGGER.info("selected %s: %d/%d", source_group, len(selected), len(source_rows))
        selected_splits.append(_prefixed_rows(selected, source_group))
    return selected_splits[0], selected_splits[1]


def _read_ccpd_rows(root: Path, ratio: float, val_ratio: float) -> tuple[list[MergedRow], list[MergedRow]]:
    train_rows: list[MergedRow] = []
    val_rows: list[MergedRow] = []
    for group in CCPD_GROUPS:
        source_group = f"ccpd__{group}"
        source_rows = _read_annotations(root / group)
        selected = _select_rows(source_rows, source_group, ratio)
        group_train, group_val = _partition_group_rows(selected, group, val_ratio)
        LOGGER.info(
            "selected %s: %d/%d; train=%d val=%d",
            group, len(selected), len(source_rows), len(group_train), len(group_val),
        )
        train_rows.extend(_prefixed_rows(group_train, source_group))
        val_rows.extend(_prefixed_rows(group_val, source_group))
    return train_rows, val_rows


def _validate_rows(rows: list[MergedRow], split: str) -> None:
    names = [row.output_name for row in rows]
    if len(names) != len(set(names)):
        raise ConversionError(f"output name collision in {split} split")
    if not rows:
        raise ConversionError(f"{split} split is empty")


def _validate_plate_config(config_path: Path, rows: list[MergedRow]) -> None:
    if not config_path.is_file():
        raise ConversionError(f"missing plate configuration: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ConversionError(f"invalid plate configuration: {config_path}")
    alphabet = config.get("alphabet")
    max_slots = config.get("max_plate_slots")
    if not isinstance(alphabet, str) or not isinstance(max_slots, int):
        raise ConversionError(f"invalid alphabet or max_plate_slots in {config_path}")
    invalid_chars = sorted({char for row in rows for char in row.plate_text} - set(alphabet))
    if invalid_chars:
        raise ConversionError(f"labels contain characters absent from config: {''.join(invalid_chars)}")
    if any(len(row.plate_text) > max_slots for row in rows):
        raise ConversionError(f"labels exceed max_plate_slots={max_slots}")


def _write_annotations(rows: list[MergedRow], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"image_path": f"images/{row.output_name}", "plate_text": row.plate_text})


def _materialize_split(rows: list[MergedRow], split_dir: Path) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    LOGGER.info("hard-linking %d images into %s", len(rows), split_dir.name)
    for row in rows:
        try:
            os.link(row.source_path, images_dir / row.output_name)
        except OSError as error:
            raise ConversionError(f"could not hard-link {row.source_path}: {error}") from error
    _write_annotations(rows, split_dir / "annotations.csv")


def _split_report(rows: list[MergedRow]) -> dict[str, object]:
    counts = Counter(row.source_group for row in rows)
    return {"total_rows": len(rows), "source_group_counts": dict(sorted(counts.items()))}


def _write_report(
    output_file: Path, *, roots: dict[str, Path], ratios: dict[str, float],
    train_rows: list[MergedRow], val_rows: list[MergedRow], val_ratio: float,
) -> None:
    report = {
        "cblprd_root": str(roots["cblprd"]),
        "ccpd_root": str(roots["ccpd2019"]),
        "ccpd2020_root": str(roots["ccpd2020"]),
        "dataset_inclusion_ratios": ratios,
        "dataset_selection_strategy": "sha256_ranked_by_source_group",
        "excluded_ccpd_groups": list(EXCLUDED_CCPD_GROUPS),
        "ccpd_validation_ratio": val_ratio,
        "ccpd_split_strategy": "sha256_ranked_stratified_by_group",
        "hard_link_count": len(train_rows) + len(val_rows),
        "splits": {"train": _split_report(train_rows), "val": _split_report(val_rows)},
    }
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_datasets(
    cblprd_root: Path | str, ccpd_root: Path | str,
    ccpd2020_root: Path | str = DEFAULT_CCPD2020_ROOT, *, out_dir: Path | str,
    cblprd_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd2020_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd_val_ratio: float = DEFAULT_CCPD_VAL_RATIO,
    plate_config: Path | str = DEFAULT_PLATE_CONFIG,
) -> Path:
    """Build one hard-linked dataset from converted CBLPRD and CCPD datasets."""
    option_ratios = {"cblprd_ratio": cblprd_ratio, "ccpd_ratio": ccpd_ratio, "ccpd2020_ratio": ccpd2020_ratio}
    for name, ratio in option_ratios.items():
        if not 0 <= ratio <= 1:
            raise ConversionError(f"{name} must be between zero and one")
    if not 0 < ccpd_val_ratio < 1:
        raise ConversionError("ccpd_val_ratio must be greater than zero and less than one")
    roots = {"cblprd": Path(cblprd_root), "ccpd2019": Path(ccpd_root), "ccpd2020": Path(ccpd2020_root)}
    ratios = {"cblprd": cblprd_ratio, "ccpd2019": ccpd_ratio, "ccpd2020": ccpd2020_ratio}
    output_dir, config_path = Path(out_dir), Path(plate_config)
    _ensure_output_available(output_dir)
    cbl_rows = _read_split_dataset(roots["cblprd"], "cblprd", cblprd_ratio) if cblprd_ratio else ([], [])
    ccpd_rows = _read_ccpd_rows(roots["ccpd2019"], ccpd_ratio, ccpd_val_ratio) if ccpd_ratio else ([], [])
    ccpd2020_rows = (_read_split_dataset(roots["ccpd2020"], "ccpd2020", ccpd2020_ratio)
                     if ccpd2020_ratio else ([], []))
    train_rows = cbl_rows[0] + ccpd_rows[0] + ccpd2020_rows[0]
    val_rows = cbl_rows[1] + ccpd_rows[1] + ccpd2020_rows[1]
    _validate_rows(train_rows, "train")
    _validate_rows(val_rows, "val")
    _validate_plate_config(config_path, train_rows + val_rows)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".merge-cblprd-ccpd-", dir=output_dir.parent))
    try:
        _materialize_split(train_rows, work_dir / "train")
        _materialize_split(val_rows, work_dir / "val")
        shutil.copyfile(config_path, work_dir / "plate_config.yaml")
        _write_report(
            work_dir / "merge_report.json", roots=roots, ratios=ratios,
            train_rows=train_rows, val_rows=val_rows, val_ratio=ccpd_val_ratio,
        )
        work_dir.replace(output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cblprd-root", type=Path, default=DEFAULT_CBLPRD_ROOT)
    parser.add_argument("--ccpd-root", type=Path, default=DEFAULT_CCPD_ROOT)
    parser.add_argument("--ccpd2020-root", type=Path, default=DEFAULT_CCPD2020_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cblprd-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--ccpd-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--ccpd2020-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--ccpd-val-ratio", type=float, default=DEFAULT_CCPD_VAL_RATIO)
    parser.add_argument("--plate-config", type=Path, default=DEFAULT_PLATE_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_dir = merge_datasets(
            args.cblprd_root, args.ccpd_root, args.ccpd2020_root, out_dir=args.out_dir,
            cblprd_ratio=args.cblprd_ratio, ccpd_ratio=args.ccpd_ratio,
            ccpd2020_ratio=args.ccpd2020_ratio,
            ccpd_val_ratio=args.ccpd_val_ratio,
            plate_config=args.plate_config,
        )
    except (ConversionError, FileExistsError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    LOGGER.info("merged dataset written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
