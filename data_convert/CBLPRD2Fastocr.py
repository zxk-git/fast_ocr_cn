#!/usr/bin/env python3
"""Convert CBLPRD-330K labels to a fast-plate-ocr dataset."""

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
DEFAULT_OUTPUT_NAME = "fast-plate-ocr"
DEFAULT_IMAGE_HEIGHT = 48
DEFAULT_IMAGE_WIDTH = 128


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


def _source_basename(label_path: str, source_file: Path, line_number: int) -> str:
    normalized = label_path.replace("\\", "/")
    prefix = f"{HISTORICAL_IMAGE_PREFIX}/"
    if not normalized.startswith(prefix):
        raise ConversionError(
            f"{source_file}:{line_number}: image path must start with {prefix}"
        )

    basename = normalized[len(prefix) :]
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


def _read_split(dataset_root: Path, split: str) -> list[LabelRow]:
    label_file = dataset_root / "label" / f"{split}.txt"
    image_dir = dataset_root / "image"
    if not label_file.is_file():
        raise ConversionError(f"missing label file: {label_file}")
    if not image_dir.is_dir():
        raise ConversionError(f"missing image directory: {image_dir}")

    rows: list[LabelRow] = []
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
    return rows


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


def _distribution(rows: Iterable[LabelRow], key) -> dict[str, int]:
    return dict(sorted(Counter(key(row) for row in rows).items()))


def _report(
    dataset_root: Path,
    output_dir: Path,
    train_rows: list[LabelRow],
    val_rows: list[LabelRow],
    alphabet: str,
    pad_char: str,
    max_slots: int,
) -> dict[str, object]:
    all_rows = train_rows + val_rows
    split_rows = {"train": train_rows, "val": val_rows}
    split_stats = {
        split: {
            "input_file": str(dataset_root / "label" / f"{split}.txt"),
            "total_rows": len(rows),
            "success_count": len(rows),
            "failure_count": 0,
            "plate_length_distribution": _distribution(rows, lambda row: str(len(row.plate_text))),
            "plate_type_distribution": _distribution(rows, lambda row: row.plate_type),
        }
        for split, rows in split_rows.items()
    }
    return {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "hard_link_count": len(all_rows),
        "max_plate_slots": max_slots,
        "alphabet": alphabet,
        "characters": sorted(set(alphabet) - {pad_char}),
        "pad_char": pad_char,
        "total_rows": len(all_rows),
        "success_count": len(all_rows),
        "failure_count": 0,
        "errors": [],
        "plate_length_distribution": _distribution(
            all_rows, lambda row: str(len(row.plate_text))
        ),
        "plate_type_distribution": _distribution(all_rows, lambda row: row.plate_type),
        "splits": split_stats,
    }


def _ensure_output_is_available(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")


def _materialize_split(rows: list[LabelRow], split_dir: Path) -> None:
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    for row in rows:
        try:
            os.link(row.source_path, images_dir / row.basename)
        except OSError as error:
            raise ConversionError(
                f"could not create hard link for {row.source_path}: {error}"
            ) from error
    _write_annotations(rows, split_dir / "annotations.csv")


def convert_dataset(
    dataset_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    hard_link: bool = True,
) -> Path:
    """Create a hard-linked fast-plate-ocr dataset from CBLPRD-330K."""
    if not hard_link:
        raise ConversionError("hard-link mode is required")

    root = Path(dataset_root)
    output_dir = Path(out_dir) if out_dir is not None else root / DEFAULT_OUTPUT_NAME
    _ensure_output_is_available(output_dir)

    train_rows = _read_split(root, "train")
    val_rows = _read_split(root, "val")
    all_rows = train_rows + val_rows
    if not all_rows:
        raise ConversionError("no labels found in train.txt and val.txt")
    _validate_collisions(all_rows)

    output_parent = output_dir.parent
    if not output_parent.is_dir():
        output_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".fast-plate-ocr-", dir=output_parent))

    try:
        _materialize_split(train_rows, work_dir / "train")
        _materialize_split(val_rows, work_dir / "val")
        alphabet, pad_char, max_slots = _write_config(all_rows, work_dir / "plate_config.yaml")
        report = _report(
            root, output_dir, train_rows, val_rows, alphabet, pad_char, max_slots
        )
        (work_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        work_dir.replace(output_dir)
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert CBLPRD-330K labels to fast-plate-ocr CSV annotations."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--hard-link",
        action="store_true",
        default=True,
        help="Create hard links to source images (always required).",
    )
    args = parser.parse_args(argv)
    try:
        output_dir = convert_dataset(
            args.dataset_root, out_dir=args.out_dir, hard_link=args.hard_link
        )
    except (ConversionError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Converted dataset written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
