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
DEFAULT_CBLPRD_ROOT = Path("/zxk/plate_ocr/plate_ocr/asserts/CBLPRD-330K/separate/single_layer")
DEFAULT_CCPD_ROOT = Path("/zxk/plate_ocr/plate_ocr/asserts/CCPD/fast-plate-ocr")
DEFAULT_CCPD2020_ROOT = Path("/zxk/plate_ocr/plate_ocr/asserts/CCPD2020/fast-plate-ocr")
DEFAULT_CHALLENGE_ROOT = Path("/zxk/plate_ocr/plate_ocr/asserts/challenge_data")
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
    # Deduplicate by basename (keep first occurrence)
    seen: set[str] = set()
    deduped: list[SourceRow] = []
    dup_count = 0
    for row in rows:
        if row.basename not in seen:
            seen.add(row.basename)
            deduped.append(row)
        else:
            dup_count += 1
    if dup_count:
        LOGGER.warning("%s: %d duplicate image_path entries removed", annotations_file, dup_count)
    return deduped


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


def _read_ccpd_rows(
    root: Path, ratio: float, anhui_ratio: float, base_ratio: float
) -> tuple[list[MergedRow], list[MergedRow]]:
    """Read CCPD groups that already have train/val subdirectories, applying ratio selection."""
    train_rows: list[MergedRow] = []
    val_rows: list[MergedRow] = []
    for group in CCPD_GROUPS:
        # ccpd_base 使用独立比例控制总样本量（覆盖 anhui_ratio 和 ratio）
        if group == "ccpd_base":
            effective_ratio = base_ratio
            effective_anhui_ratio = base_ratio
        else:
            effective_ratio = ratio
            effective_anhui_ratio = anhui_ratio

        group_dir = root / group
        train_dir = group_dir / "train"
        val_dir = group_dir / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise ConversionError(f"{group_dir}: missing train/val subdirectories (run CCPD2Fastocr.py first)")

        source_train = _read_annotations(train_dir)
        source_val = _read_annotations(val_dir)

        # Separate Anhui vs non-Anhui
        def _is_anhui(r: SourceRow) -> bool:
            return r.plate_text.startswith("皖")

        train_anhui = [r for r in source_train if _is_anhui(r)]
        train_other = [r for r in source_train if not _is_anhui(r)]
        val_anhui = [r for r in source_val if _is_anhui(r)]
        val_other = [r for r in source_val if not _is_anhui(r)]

        # Apply independent ratios
        train_selected = (_select_rows(train_anhui, f"ccpd_anhui_train__{group}", effective_anhui_ratio)
                          + _select_rows(train_other, f"ccpd_other_train__{group}", effective_ratio))
        val_selected = (_select_rows(val_anhui, f"ccpd_anhui_val__{group}", effective_anhui_ratio)
                        + _select_rows(val_other, f"ccpd_other_val__{group}", effective_ratio))

        LOGGER.info(
            "selected %s: train=%d/%d val=%d/%d (安徽=%d/%d+%d/%d 其他=%d/%d+%d/%d)",
            group,
            len(train_selected), len(source_train), len(val_selected), len(source_val),
            len([r for r in train_selected if r.plate_text.startswith("皖")]), len(train_anhui),
            len([r for r in val_selected if r.plate_text.startswith("皖")]), len(val_anhui),
            len([r for r in train_selected if not r.plate_text.startswith("皖")]), len(train_other),
            len([r for r in val_selected if not r.plate_text.startswith("皖")]), len(val_other),
        )
        source_group = f"ccpd__{group}"
        train_rows.extend(_prefixed_rows(train_selected, source_group))
        val_rows.extend(_prefixed_rows(val_selected, source_group))
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


def _read_challenge_data(
    root: Path, ratio: float, val_ratio: float
) -> tuple[list[MergedRow], list[MergedRow]]:
    """Read challenge_data (flat annotations.csv + images/), split into train/val."""
    source_rows = _read_annotations(root)  # root 下直接有 annotations.csv
    selected = _select_rows(source_rows, "challenge", ratio)
    val_count = max(1, min(len(selected) - 1, round(len(selected) * val_ratio)))
    ranked = _rank_rows(selected, "challenge")
    train = ranked[val_count:]
    val = ranked[:val_count]
    LOGGER.info("selected challenge_data: %d/%d; train=%d val=%d",
                 len(selected), len(source_rows), len(train), len(val))
    return _prefixed_rows(train, "challenge"), _prefixed_rows(val, "challenge")


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
    cblprd_root: Path | str,
    ccpd_root: Path | str,
    ccpd2020_root: Path | str = DEFAULT_CCPD2020_ROOT,
    *,
    out_dir: Path | str,
    cblprd_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd_anhui_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd_base_ratio: float = DEFAULT_DATASET_RATIO,
    ccpd2020_ratio: float = DEFAULT_DATASET_RATIO,
    challenge_root: Path | str | None = None,
    challenge_ratio: float = 0.0,
    challenge_val_ratio: float = DEFAULT_CCPD_VAL_RATIO,
    plate_config: Path | str = DEFAULT_PLATE_CONFIG,
) -> Path:
    """Build one hard-linked dataset from converted CBLPRD, CCPD, and challenge datasets."""
    option_ratios = {
        "cblprd_ratio": cblprd_ratio, "ccpd_ratio": ccpd_ratio,
        "ccpd_anhui_ratio": ccpd_anhui_ratio, "ccpd2020_ratio": ccpd2020_ratio,
        "challenge_ratio": challenge_ratio,
    }
    for name, ratio in option_ratios.items():
        if not 0 <= ratio <= 1:
            raise ConversionError(f"{name} must be between zero and one")
    roots = {"cblprd": Path(cblprd_root), "ccpd2019": Path(ccpd_root), "ccpd2020": Path(ccpd2020_root)}
    ratios = {
        "cblprd": cblprd_ratio, "ccpd2019": ccpd_ratio,
        "ccpd_anhui": ccpd_anhui_ratio, "ccpd2020": ccpd2020_ratio,
    }
    if challenge_root:
        roots["challenge"] = Path(challenge_root)
        ratios["challenge"] = challenge_ratio
    output_dir, config_path = Path(out_dir), Path(plate_config)
    _ensure_output_available(output_dir)
    cbl_rows = _read_split_dataset(roots["cblprd"], "cblprd", cblprd_ratio) if cblprd_ratio else ([], [])
    ccpd_rows = _read_ccpd_rows(roots["ccpd2019"], ccpd_ratio, ccpd_anhui_ratio, ccpd_base_ratio) if ccpd_ratio else ([], [])
    ccpd2020_rows = (_read_split_dataset(roots["ccpd2020"], "ccpd2020", ccpd2020_ratio)
                     if ccpd2020_ratio else ([], []))
    train_rows = cbl_rows[0] + ccpd_rows[0] + ccpd2020_rows[0]
    val_rows = cbl_rows[1] + ccpd_rows[1] + ccpd2020_rows[1]
    if challenge_root and challenge_ratio:
        ch_train, ch_val = _read_challenge_data(roots["challenge"], challenge_ratio, challenge_val_ratio)
        train_rows.extend(ch_train)
        val_rows.extend(ch_val)
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
            train_rows=train_rows, val_rows=val_rows, val_ratio=challenge_val_ratio,
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
    parser.add_argument("--challenge-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cblprd-ratio", type=float, default=0.05)
    parser.add_argument("--ccpd-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--ccpd-anhui-ratio", type=float, default=DEFAULT_DATASET_RATIO,
                        help="CCPD2019 安徽车牌保留比例 (0-1, 默认全保留)")
    parser.add_argument("--ccpd-base-ratio", type=float, default=0.06,
                        help="CCPD2019 ccpd_base 组保留比例 (0-1, 默认全保留)")
    parser.add_argument("--ccpd2020-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--challenge-ratio", type=float, default=DEFAULT_DATASET_RATIO)
    parser.add_argument("--challenge-val-ratio", type=float, default=DEFAULT_CCPD_VAL_RATIO)
    parser.add_argument("--plate-config", type=Path, default=DEFAULT_PLATE_CONFIG)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        output_dir = merge_datasets(
            args.cblprd_root, args.ccpd_root, args.ccpd2020_root, out_dir=args.out_dir,
            cblprd_ratio=args.cblprd_ratio, ccpd_ratio=args.ccpd_ratio,
            ccpd_anhui_ratio=args.ccpd_anhui_ratio,
            ccpd_base_ratio=args.ccpd_base_ratio,
            ccpd2020_ratio=args.ccpd2020_ratio,
            challenge_root=args.challenge_root,
            challenge_ratio=args.challenge_ratio,
            challenge_val_ratio=args.challenge_val_ratio,
            plate_config=args.plate_config,
        )
    except (ConversionError, FileExistsError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    LOGGER.info("merged dataset written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
