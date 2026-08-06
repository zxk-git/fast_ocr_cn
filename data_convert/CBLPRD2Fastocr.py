#!/usr/bin/env python3
"""CBLPRD-330K 数据处理工具。

支持两个子命令:
  separate  — 将 CBLPRD 数据分离为单层/双层两个文件夹
  convert   — 将指定文件夹的数据转换为 fast-plate-ocr 格式
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HISTORICAL_IMAGE_PREFIX = "CBLPRD-330k"
DEFAULT_IMAGE_HEIGHT = 64
DEFAULT_IMAGE_WIDTH = 128

# Province abbreviation -> full name mapping
PROVINCE_MAP: dict[str, str] = {
    "京": "北京", "津": "天津", "沪": "上海", "渝": "重庆",
    "冀": "河北", "晋": "山西", "蒙": "内蒙古", "辽": "辽宁",
    "吉": "吉林", "黑": "黑龙江", "苏": "江苏", "浙": "浙江",
    "皖": "安徽", "闽": "福建", "赣": "江西", "鲁": "山东",
    "豫": "河南", "鄂": "湖北", "湘": "湖南", "粤": "广东",
    "桂": "广西", "琼": "海南", "川": "四川", "贵": "贵州",
    "云": "云南", "藏": "西藏", "陕": "陕西", "甘": "甘肃",
    "青": "青海", "宁": "宁夏", "新": "新疆", "港": "香港",
    "澳": "澳门",
    # Special plate type characters
    "使": "使馆", "领": "领馆", "学": "驾校", "挂": "挂车", "临": "临时",
}


def _province_from_plate(plate_text: str) -> str:
    """Extract the province/special-type from the first character of a plate."""
    if not plate_text:
        return "未知"
    first_char = plate_text[0]
    if first_char in PROVINCE_MAP:
        return PROVINCE_MAP[first_char]
    return "其他"


def _plate_layer_type(plate_type: str) -> str:
    """Classify a plate as 单层 (single-layer) or 双层 (double-layer)."""
    ptype = plate_type.strip()
    if "双" in ptype or "double" in ptype.lower():
        return "双层"
    return "单层"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ConversionError(ValueError):
    """Raised when the source dataset cannot be converted safely."""


@dataclass(frozen=True)
class LabelRow:
    split: str
    line_number: int
    basename: str
    source_path: Path
    plate_text: str
    plate_type: str


@dataclass(frozen=True)
class RejectedRow:
    split: str
    line_number: int
    source_path: Path
    plate_text: str
    plate_type: str
    reason: str


# ---------------------------------------------------------------------------
# Label reading
# ---------------------------------------------------------------------------


def _source_basename(label_path: str, source_file: Path, line_number: int) -> str:
    normalized = label_path.replace("\\", "/")
    prefix = f"{HISTORICAL_IMAGE_PREFIX}/"
    if not normalized.startswith(prefix):
        raise ConversionError(
            f"{source_file}:{line_number}: image path must start with {prefix}"
        )
    basename = normalized[len(prefix):]
    if not basename or "/" in basename or Path(basename).name != basename:
        raise ConversionError(
            f"{source_file}:{line_number}: image path must contain one JPG basename"
        )
    if not basename.endswith(".jpg"):
        raise ConversionError(
            f"{source_file}:{line_number}: image name must end with .jpg"
        )
    return basename


def _reject_control_characters(
    value: str, field_name: str, source_file: Path, line_number: int
) -> None:
    for character in value:
        codepoint = ord(character)
        if unicodedata.category(character) == "Cc":
            raise ConversionError(
                f"{source_file}:{line_number}: {field_name} contains control character "
                f"U+{codepoint:04X}"
            )
        if _is_yaml_nonprintable(codepoint):
            raise ConversionError(
                f"{source_file}:{line_number}: {field_name} contains YAML-nonprintable "
                f"character U+{codepoint:04X}"
            )


def _is_yaml_nonprintable(codepoint: int) -> bool:
    is_noncharacter = 0xFDD0 <= codepoint <= 0xFDEF or codepoint & 0xFFFF in {
        0xFFFE,
        0xFFFF,
    }
    if is_noncharacter:
        return True
    return not (
        codepoint in {0x09, 0x0A, 0x0D, 0x85}
        or 0x20 <= codepoint <= 0x7E
        or 0xA0 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _read_split(
    dataset_root: Path, split: str, max_plate_slots: int | None = None
) -> tuple[list[LabelRow], list[RejectedRow]]:
    label_file = dataset_root / "label" / f"{split}.txt"
    image_dir = dataset_root / "image"
    if not label_file.is_file():
        raise ConversionError(f"missing label file: {label_file}")
    if not image_dir.is_dir():
        raise ConversionError(f"missing image directory: {image_dir}")

    rows: list[LabelRow] = []
    rejected: list[RejectedRow] = []
    with label_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                raise ConversionError(f"{label_file}:{line_number}: empty rows are not allowed")
            fields = line.split()
            if len(fields) != 3:
                raise ConversionError(
                    f"{label_file}:{line_number}: expected three whitespace-separated fields"
                )
            label_path, plate_text, plate_type = fields
            if not plate_text:
                raise ConversionError(f"{label_file}:{line_number}: plate text is empty")
            _reject_control_characters(label_path, "image path", label_file, line_number)
            _reject_control_characters(plate_text, "plate text", label_file, line_number)
            _reject_control_characters(plate_type, "plate type", label_file, line_number)
            basename = _source_basename(label_path, label_file, line_number)
            source_path = image_dir / basename
            if not source_path.is_file():
                raise ConversionError(
                    f"{label_file}:{line_number}: source image does not exist: {source_path}"
                )

            if max_plate_slots is not None and len(plate_text) > max_plate_slots:
                rejected.append(
                    RejectedRow(
                        split=split,
                        line_number=line_number,
                        source_path=source_path,
                        plate_text=plate_text,
                        plate_type=plate_type,
                        reason=f"plate length {len(plate_text)} exceeds max {max_plate_slots}",
                    )
                )
                continue

            rows.append(
                LabelRow(
                    split=split,
                    line_number=line_number,
                    basename=basename,
                    source_path=source_path,
                    plate_text=plate_text,
                    plate_type=plate_type,
                )
            )
    return rows, rejected


def _validate_collisions(rows: Iterable[LabelRow]) -> None:
    seen: dict[str, LabelRow] = {}
    for row in rows:
        previous = seen.get(row.basename)
        if previous is not None:
            raise ConversionError(
                "collision for image basename "
                f"{row.basename}: {previous.split}:{previous.line_number} and "
                f"{row.split}:{row.line_number}"
            )
        seen[row.basename] = row


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _select_pad_char(alphabet: set[str]) -> str:
    for candidate in ("_", "*", "#", "~", "@", "|"):
        if candidate not in alphabet:
            return candidate
    for codepoint in range(0xE000, 0xF8FF + 1):
        candidate = chr(codepoint)
        if candidate not in alphabet:
            return candidate
    raise ConversionError("could not select a non-conflicting padding character")


def _write_annotations(rows: list[LabelRow], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"image_path": f"images/{row.basename}", "plate_text": row.plate_text}
            )


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _write_config(rows: list[LabelRow], output_file: Path) -> tuple[str, str, int]:
    observed_alphabet = {character for row in rows for character in row.plate_text}
    pad_char = _select_pad_char(observed_alphabet)
    alphabet = "".join(sorted(observed_alphabet)) + pad_char
    max_slots = max(len(row.plate_text) for row in rows)
    output_file.write_text(
        "\n".join(
            [
                f"max_plate_slots: {max_slots}",
                f"alphabet: {_yaml_quote(alphabet)}",
                f"pad_char: {_yaml_quote(pad_char)}",
                f"img_height: {DEFAULT_IMAGE_HEIGHT}",
                f"img_width: {DEFAULT_IMAGE_WIDTH}",
                "keep_aspect_ratio: true",
                "interpolation: linear",
                "image_color_mode: rgb",
                "padding_color: 114",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return alphabet, pad_char, max_slots


def _write_rejected_csv(rejected: list[RejectedRow], output_file: Path) -> None:
    fieldnames = ["split", "line_number", "source_path", "plate_text", "plate_type", "plate_length", "reason"]
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rejected:
            writer.writerow(
                {
                    "split": row.split,
                    "line_number": row.line_number,
                    "source_path": str(row.source_path),
                    "plate_text": row.plate_text,
                    "plate_type": row.plate_type,
                    "plate_length": len(row.plate_text),
                    "reason": row.reason,
                }
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _distribution(rows: Iterable[LabelRow], key) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))


def _rejected_length_distribution(rejected: Iterable[RejectedRow]) -> dict[str, int]:
    return dict(sorted(Counter(str(len(row.plate_text)) for row in rejected).items()))


def _build_report(
    source_root: Path,
    output_dir: Path,
    train_rows: list[LabelRow],
    val_rows: list[LabelRow],
    train_rejected: list[RejectedRow],
    val_rejected: list[RejectedRow],
    alphabet: str,
    pad_char: str,
    max_slots: int,
) -> dict[str, object]:
    all_rows = train_rows + val_rows
    all_rejected = train_rejected + val_rejected
    total_input = len(all_rows) + len(all_rejected)
    split_rows = {"train": train_rows, "val": val_rows}
    split_rejected = {"train": train_rejected, "val": val_rejected}
    split_stats = {
        split: {
            "input_file": str(source_root / "label" / f"{split}.txt"),
            "total_input_rows": len(rows) + len(split_rejected[split]),
            "kept_count": len(rows),
            "filtered_count": len(split_rejected[split]),
            "filter_pct": (
                round(len(split_rejected[split]) / (len(rows) + len(split_rejected[split])) * 100, 2)
                if (len(rows) + len(split_rejected[split])) > 0
                else 0.0
            ),
            "plate_length_distribution": _distribution(rows, lambda row: str(len(row.plate_text))),
            "plate_type_distribution": _distribution(rows, lambda row: row.plate_type),
            "province_distribution": _distribution(rows, lambda row: _province_from_plate(row.plate_text)),
            "layer_type_distribution": _distribution(rows, lambda row: _plate_layer_type(row.plate_type)),
        }
        for split, rows in split_rows.items()
    }
    return {
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "hard_link_count": len(all_rows),
        "max_plate_slots": max_slots,
        "alphabet": alphabet,
        "characters": sorted(set(alphabet) - {pad_char}),
        "pad_char": pad_char,
        "total_input_rows": total_input,
        "total_kept_rows": len(all_rows),
        "total_filtered_rows": len(all_rejected),
        "filter_pct": round(len(all_rejected) / total_input * 100, 2) if total_input > 0 else 0.0,
        "filtered_out_count": len(all_rejected),
        "filtered_out_plate_length_distribution": _rejected_length_distribution(all_rejected),
        "filtered_out_by_split": {"train": len(train_rejected), "val": len(val_rejected)},
        "plate_length_distribution": _distribution(all_rows, lambda row: str(len(row.plate_text))),
        "plate_type_distribution": _distribution(all_rows, lambda row: row.plate_type),
        "province_distribution": _distribution(all_rows, lambda row: _province_from_plate(row.plate_text)),
        "layer_type_distribution": _distribution(all_rows, lambda row: _plate_layer_type(row.plate_type)),
        "splits": split_stats,
    }


def _print_summary(output_dir: Path) -> None:
    report_path = output_dir / "conversion_report.json"
    if not report_path.is_file():
        return
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    total_input = report.get("total_input_rows", 0)
    total_kept = report.get("total_kept_rows", 0)
    total_filtered = report.get("total_filtered_rows", 0)
    filter_pct = report.get("filter_pct", 0.0)
    max_slots = report.get("max_plate_slots", "?")

    print()
    print("=" * 60)
    print("  数据转换统计报告")
    print("=" * 60)
    print(f"\n  📊 总体统计")
    print(f"     输入车牌总数        : {total_input}")
    print(f"     保留 (位数 <= {max_slots}) : {total_kept}")
    print(f"     过滤 (位数 > {max_slots})  : {total_filtered} ({filter_pct}%)")
    if total_filtered > 0:
        print(f"     过滤记录已保存至        : {report_path.parent / 'rejected_plates.csv'}")

    _print_distribution_table(report, "plate_length_distribution", "📏 车牌位数分布 (保留数据)", "位数")

    filtered = report.get("filtered_out_plate_length_distribution", {})
    if filtered:
        _print_distribution_table({}, "filtered_out_plate_length_distribution",
                                   f"⛔ 过滤车牌位数分布 (位数 > {max_slots})", "位数", custom_data=report)

    _print_distribution_table(report, "province_distribution", "🗺️  省份分布统计 (保留数据)", "省份")
    _print_distribution_table(report, "layer_type_distribution", "🚗 单/双层车牌统计 (保留数据)", "类型")
    _print_distribution_table(report, "plate_type_distribution",
                               "🏷️  原始标签类型分布 (保留数据, Top 15)", "标签类型", top_n=15)
    _print_split_breakdown(report)
    print("=" * 60)


def _print_distribution_table(
    report: dict,
    key: str,
    title: str,
    label: str,
    custom_data: dict | None = None,
    top_n: int | None = None,
) -> None:
    data = custom_data.get(key, {}) if custom_data is not None else report.get(key, {})
    if not data:
        return
    items = list(data.items())
    if top_n is not None and len(items) > top_n:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n  {title}")
    total = sum(count for _, count in items)
    for name, count in items:
        pct = count / total * 100 if total > 0 else 0.0
        print(f"     {name:<12} : {count:>7}  ({pct:5.1f}%)")
    if top_n is not None and len(data) > top_n:
        print(f"     ... 共 {len(data)} 种标签，以上为 Top {top_n}")


def _print_split_breakdown(report: dict) -> None:
    splits = report.get("splits", {})
    if not splits:
        return
    print(f"\n  📂 分集统计")
    for split_name in ("train", "val"):
        s = splits.get(split_name, {})
        k = s.get("kept_count", 0)
        f = s.get("filtered_count", 0)
        fp = s.get("filter_pct", 0.0)
        total_in = s.get("total_input_rows", 0)
        label_map = {"train": "训练集", "val": "验证集"}
        print(f"     {label_map.get(split_name, split_name):<6} : 输入={total_in}, 保留={k}, 过滤={f} ({fp}%)")


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def _ensure_output_is_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _materialize_split_links(
    rows: list[LabelRow],
    images_dir: Path,
) -> None:
    """Create hard links for all rows into images_dir."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = images_dir / row.basename
        try:
            os.link(row.source_path, dest)
        except OSError as error:
            raise ConversionError(
                f"could not create hard link for {row.source_path}: {error}"
            ) from error


# ---------------------------------------------------------------------------
# Subcommand: separate
# ---------------------------------------------------------------------------


def separate_dataset(
    dataset_root: Path,
    out_dir: Path,
    max_plate_slots: int | None = None,
) -> Path:
    """Separate CBLPRD-330K into single-layer and double-layer output directories."""
    root = Path(dataset_root)
    _ensure_output_is_available(out_dir)

    # Read labels
    train_rows, train_rejected = _read_split(root, "train", max_plate_slots)
    val_rows, val_rejected = _read_split(root, "val", max_plate_slots)
    all_rows = train_rows + val_rows
    all_rejected = train_rejected + val_rejected
    if not all_rows and not all_rejected:
        raise ConversionError("no labels found in train.txt and val.txt")
    _validate_collisions(all_rows)

    # Split into single / double layer
    single_rows = {"train": [], "val": []}
    double_rows = {"train": [], "val": []}
    for row in train_rows:
        target = double_rows if _plate_layer_type(row.plate_type) == "双层" else single_rows
        target["train"].append(row)
    for row in val_rows:
        target = double_rows if _plate_layer_type(row.plate_type) == "双层" else single_rows
        target["val"].append(row)

    # Also split rejected rows
    single_rejected = {"train": [], "val": []}
    double_rejected = {"train": [], "val": []}
    for r in train_rejected:
        target = double_rejected if _plate_layer_type(r.plate_type) == "双层" else single_rejected
        target["train"].append(r)
    for r in val_rejected:
        target = double_rejected if _plate_layer_type(r.plate_type) == "双层" else single_rejected
        target["val"].append(r)

    # Materialize both types
    output_parent = out_dir.parent if out_dir.parent.is_dir() else out_dir
    if not output_parent.is_dir():
        output_parent.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix=".cblprd-separate-", dir=output_parent))
    try:
        _write_separated(work_dir / "single_layer", single_rows, single_rejected, root)
        _write_separated(work_dir / "double_layer", double_rows, double_rejected, root)
        work_dir.replace(out_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    # Print summary
    print(f"\n分离完成: {out_dir}")
    print(f"  单层车牌: train={len(single_rows['train'])}, val={len(single_rows['val'])}")
    print(f"  双层车牌: train={len(double_rows['train'])}, val={len(double_rows['val'])}")
    _print_summary(out_dir / "single_layer")
    _print_summary(out_dir / "double_layer")

    return out_dir


def _write_separated(
    category_dir: Path,
    split_rows: dict[str, list[LabelRow]],
    split_rejected: dict[str, list[RejectedRow]],
    source_root: Path,
) -> None:
    """Write one category (single or double layer) into a subdirectory."""
    category_dir.mkdir(parents=True)

    all_rows = split_rows["train"] + split_rows["val"]
    if not all_rows:
        return  # empty category, nothing to write

    for split_name in ("train", "val"):
        rows = split_rows[split_name]
        if rows:
            _materialize_split_links(rows, category_dir / split_name / "images")
            _write_annotations(rows, category_dir / split_name / "annotations.csv")

    # Write config from all rows in this category
    alphabet, pad_char, max_slots = _write_config(all_rows, category_dir / "plate_config.yaml")

    # Write rejected CSV if any
    all_rejected = split_rejected["train"] + split_rejected["val"]
    if all_rejected:
        _write_rejected_csv(all_rejected, category_dir / "rejected_plates.csv")

    # Write report
    report = _build_report(
        source_root, category_dir,
        split_rows["train"], split_rows["val"],
        split_rejected["train"], split_rejected["val"],
        alphabet, pad_char, max_slots,
    )
    (category_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------


def convert_to_fastocr(
    source_dir: Path,
    out_dir: Path,
    max_plate_slots: int | None = None,
    plate_config_path: Path | None = None,
) -> Path:
    """Convert a directory (containing train/val annotations) to fast-plate-ocr format."""
    source = Path(source_dir)
    output_dir = Path(out_dir) if out_dir is not None else source / "fast-plate-ocr"
    _ensure_output_is_available(output_dir)

    # Read annotations from source directory
    train_annots = source / "train" / "annotations.csv"
    val_annots = source / "val" / "annotations.csv"
    if not train_annots.is_file():
        raise ConversionError(f"missing train annotations: {train_annots}")
    if not val_annots.is_file():
        raise ConversionError(f"missing val annotations: {val_annots}")

    train_rows, train_rejected = _read_annotations_csv(train_annots, source / "train", "train", max_plate_slots)
    val_rows, val_rejected = _read_annotations_csv(val_annots, source / "val", "val", max_plate_slots)
    all_rows = train_rows + val_rows
    all_rejected = train_rejected + val_rejected
    if not all_rows:
        raise ConversionError("no valid annotations found")
    _validate_collisions(all_rows)

    output_parent = output_dir.parent
    if not output_parent.is_dir():
        output_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".fast-plate-ocr-", dir=output_parent))

    try:
        _materialize_split_links(train_rows, work_dir / "train" / "images")
        _materialize_split_links(val_rows, work_dir / "val" / "images")
        _write_annotations(train_rows, work_dir / "train" / "annotations.csv")
        _write_annotations(val_rows, work_dir / "val" / "annotations.csv")

        # Use provided config or generate from data
        if plate_config_path and Path(plate_config_path).is_file():
            shutil.copyfile(plate_config_path, work_dir / "plate_config.yaml")
            # Parse config for report
            import yaml as _yaml
            with open(plate_config_path, encoding="utf-8") as f:
                cfg = _yaml.safe_load(f)
            alphabet = cfg.get("alphabet", "")
            pad_char = cfg.get("pad_char", "_")
            max_slots = cfg.get("max_plate_slots", 8)
        else:
            alphabet, pad_char, max_slots = _write_config(all_rows, work_dir / "plate_config.yaml")

        if all_rejected:
            _write_rejected_csv(all_rejected, work_dir / "rejected_plates.csv")

        report = _build_report(
            source, output_dir,
            train_rows, val_rows,
            train_rejected, val_rejected,
            alphabet, pad_char, max_slots,
        )
        (work_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        work_dir.replace(output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        raise

    _print_summary(output_dir)
    return output_dir


def _read_annotations_csv(
    csv_path: Path,
    source_dir: Path,
    split: str,
    max_plate_slots: int | None = None,
) -> tuple[list[LabelRow], list[RejectedRow]]:
    """Read annotations.csv and convert to LabelRow list."""
    rows: list[LabelRow] = []
    rejected: list[RejectedRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, 1):
            image_path = row.get("image_path", "").strip()
            plate_text = row.get("plate_text", "").strip()
            if not plate_text:
                raise ConversionError(
                    f"{csv_path}:{line_number}: plate_text is empty"
                )

            # Resolve image path: "images/xxx.jpg" or absolute
            if image_path.startswith("images/"):
                basename = image_path[len("images/"):]
            else:
                basename = Path(image_path).name

            source_path = source_dir / "images" / basename
            if not source_path.is_file():
                raise ConversionError(
                    f"{csv_path}:{line_number}: source image not found: {source_path}"
                )

            if max_plate_slots is not None and len(plate_text) > max_plate_slots:
                rejected.append(
                    RejectedRow(
                        split=split, line_number=line_number,
                        source_path=source_path, plate_text=plate_text,
                        plate_type="", reason=f"plate length {len(plate_text)} exceeds max {max_plate_slots}",
                    )
                )
                continue

            rows.append(
                LabelRow(
                    split=split, line_number=line_number,
                    basename=basename, source_path=source_path,
                    plate_text=plate_text, plate_type="",
                )
            )
    return rows, rejected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")


def _cmd_separate(args: argparse.Namespace) -> int:
    separate_dataset(
        args.dataset_root,
        args.out_dir,
        max_plate_slots=args.max_plate_slots,
    )
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    convert_to_fastocr(
        args.source_dir,
        args.out_dir,
        max_plate_slots=args.max_plate_slots,
        plate_config_path=args.plate_config,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CBLPRD-330K 数据处理工具: separate / convert"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- separate ----
    sep_parser = subparsers.add_parser("separate", help="将 CBLPRD 数据分离为单层/双层两个文件夹")
    sep_parser.add_argument("--dataset-root", type=Path, required=True, help="CBLPRD-330K 数据集根目录")
    sep_parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    sep_parser.add_argument(
        "--max-plate-slots", type=int, default=8,
        help="过滤超过指定位数的车牌 (默认不过滤)",
    )

    # ---- convert ----
    conv_parser = subparsers.add_parser("convert", help="将指定文件夹转换为 fastocr 格式")
    conv_parser.add_argument("--source-dir", type=Path, required=True, help="包含 train/val 的源目录")
    conv_parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    conv_parser.add_argument(
        "--max-plate-slots", type=int, default=8,
        help="过滤超过指定位数的车牌 (默认不过滤)",
    )
    conv_parser.add_argument(
        "--plate-config", type=Path, default=None,
        help="指定 plate_config.yaml 路径 (不指定则从数据自动生成)",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "separate":
            return _cmd_separate(args)
        if args.command == "convert":
            return _cmd_convert(args)
        parser.error(f"unknown command: {args.command}")
    except (ConversionError, FileExistsError) as error:
        parser.error(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
