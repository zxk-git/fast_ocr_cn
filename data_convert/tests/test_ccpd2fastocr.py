"""Tests for CCPD-to-fast-plate-ocr conversion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from data_convert.CCPD2Fastocr import (
    CCPD_GROUPS,
    ConversionError,
    convert_dataset,
    parse_ccpd_filename,
    rectify_plate,
)


PLATE_INDEXES = "0_0_27_33_33_23_26"


def _filename(prefix: str = "100", polygon: str | None = None) -> str:
    points = polygon or "18&16_3&15_5&3_17&5"
    return f"{prefix}-0_0-3&3_18&16-{points}-{PLATE_INDEXES}-0-0.jpg"


def _write_source_image(path: Path) -> None:
    image = np.zeros((20, 22, 3), dtype=np.uint8)
    polygon = np.array([[18, 16], [3, 15], [5, 3], [17, 5]], dtype=np.int32)
    cv2.fillConvexPoly(image, polygon, (220, 220, 220))
    cv2.circle(image, (6, 4), 2, (0, 0, 255), -1)
    cv2.circle(image, (16, 6), 2, (0, 255, 0), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _build_dataset(root: Path, *, files_per_group: int = 1) -> None:
    for group_index, group in enumerate(CCPD_GROUPS):
        for file_index in range(files_per_group):
            name = _filename(prefix=f"{group_index:02d}{file_index:02d}")
            _write_source_image(root / group / name)
    (root / "ccpd_np").mkdir(parents=True)
    _write_source_image(root / "ccpd_np" / "4668.jpg")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_parse_ccpd_filename_decodes_plate_and_reorders_polygon() -> None:
    annotation = parse_ccpd_filename(_filename())

    assert annotation.plate_text == "皖A399Z2"
    assert annotation.polygon == ((5, 3), (17, 5), (18, 16), (3, 15))


def test_rectify_plate_uses_natural_perspective_dimensions() -> None:
    image = np.zeros((20, 22, 3), dtype=np.uint8)
    polygon = np.array([[18, 16], [3, 15], [5, 3], [17, 5]], dtype=np.int32)
    cv2.fillConvexPoly(image, polygon, (255, 255, 255))
    cv2.circle(image, (6, 4), 2, (0, 0, 255), -1)
    cv2.circle(image, (16, 6), 2, (0, 255, 0), -1)

    crop = rectify_plate(image, ((5, 3), (17, 5), (18, 16), (3, 15)))

    assert crop.shape[:2] == (12, 15)
    assert crop[:, : crop.shape[1] // 2, 2].max() > 150
    assert crop[:, crop.shape[1] // 2 :, 1].max() > 150
    assert float(crop.mean()) > 80


def test_rectify_plate_clamps_inclusive_image_edges() -> None:
    image = np.full((10, 10, 3), 200, dtype=np.uint8)

    crop = rectify_plate(image, ((1, 1), (10, 1), (10, 10), (1, 10)))

    assert crop.shape[:2] == (8, 8)
    assert float(crop.mean()) > 150


def test_rectify_plate_rejects_coordinates_beyond_inclusive_edges() -> None:
    image = np.full((10, 10, 3), 200, dtype=np.uint8)

    with pytest.raises(ConversionError, match="outside image bounds"):
        rectify_plate(image, ((1, 1), (11, 1), (11, 10), (1, 10)))


def test_converts_eight_groups_and_excludes_unlabelled_np(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root)

    out_dir = convert_dataset(dataset_root, workers=2)

    assert out_dir == dataset_root / "fast-plate-ocr"
    for group in CCPD_GROUPS:
        rows = _read_csv(out_dir / group / "annotations.csv")
        assert len(rows) == 1
        assert rows[0]["plate_text"] == "皖A399Z2"
        crop = cv2.imread(str(out_dir / group / rows[0]["image_path"]))
        assert crop is not None
        assert crop.shape[:2] == (12, 15)
    assert not (out_dir / "ccpd_np").exists()

    config = yaml.safe_load((out_dir / "plate_config.yaml").read_text(encoding="utf-8"))
    assert config == {
        "max_plate_slots": 7,
        "alphabet": "239AZ皖_",
        "pad_char": "_",
        "img_height": 64,
        "img_width": 128,
        "keep_aspect_ratio": True,
        "interpolation": "linear",
        "image_color_mode": "rgb",
        "padding_color": 114,
    }

    report = json.loads((out_dir / "conversion_report.json").read_text(encoding="utf-8"))
    assert report["total_images"] == 8
    assert report["success_count"] == 8
    assert report["failure_count"] == 0
    assert report["excluded_groups"] == ["ccpd_np"]
    assert set(report["groups"]) == set(CCPD_GROUPS)
    assert all(report["groups"][group]["success_count"] == 1 for group in CCPD_GROUPS)
    assert report["conversion"]["crop_mode"] == "four_point_perspective"
    assert report["conversion"]["jpeg_quality"] == 95


def test_annotations_are_sorted_despite_parallel_completion(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root, files_per_group=3)

    out_dir = convert_dataset(dataset_root, workers=3)

    rows = _read_csv(out_dir / "ccpd_base" / "annotations.csv")
    assert [row["image_path"] for row in rows] == sorted(
        row["image_path"] for row in rows
    )


@pytest.mark.parametrize(
    "bad_name",
    [
        "bad.jpg",
        _filename(polygon="1&1_1&1_1&1_1&1"),
        f"100-0_0-3&3_18&16-18&16_3&15_5&3_17&5-99_0_27_33_33_23_26-0-0.jpg",
    ],
    ids=["invalid_fields", "degenerate_polygon", "invalid_index"],
)
def test_invalid_annotation_does_not_create_output(tmp_path: Path, bad_name: str) -> None:
    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root)
    valid = next((dataset_root / "ccpd_base").iterdir())
    valid.unlink()
    _write_source_image(dataset_root / "ccpd_base" / bad_name)
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError):
        convert_dataset(dataset_root, out_dir=out_dir, workers=1)

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".fast-plate-ocr-*"))


def test_corrupt_jpeg_rolls_back_existing_empty_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root)
    corrupt = next((dataset_root / "ccpd_blur").iterdir())
    corrupt.write_bytes(b"not-a-jpeg")
    out_dir = tmp_path / "converted"
    out_dir.mkdir()

    with pytest.raises(ConversionError, match="decode"):
        convert_dataset(dataset_root, out_dir=out_dir, workers=2)

    assert list(out_dir.iterdir()) == []
    assert not list(tmp_path.glob(".fast-plate-ocr-*"))


def test_write_failure_rolls_back_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root)
    out_dir = tmp_path / "converted"

    import data_convert.CCPD2Fastocr as converter

    monkeypatch.setattr(converter.cv2, "imwrite", lambda *_args, **_kwargs: False)

    with pytest.raises(ConversionError, match="write"):
        convert_dataset(dataset_root, out_dir=out_dir, workers=1)

    assert not out_dir.exists()
    assert not list(tmp_path.glob(".fast-plate-ocr-*"))


def test_missing_group_nonempty_output_and_invalid_workers_are_rejected(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    with pytest.raises(ConversionError, match="missing CCPD group"):
        convert_dataset(missing_root, out_dir=tmp_path / "missing-output")

    dataset_root = tmp_path / "CCPD2019"
    _build_dataset(dataset_root)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    sentinel = nonempty / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        convert_dataset(dataset_root, out_dir=nonempty)
    assert sentinel.read_text(encoding="utf-8") == "keep"

    with pytest.raises(ConversionError, match="workers"):
        convert_dataset(dataset_root, out_dir=tmp_path / "bad-workers", workers=0)
