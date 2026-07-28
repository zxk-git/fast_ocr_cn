"""
Tests for annotations loading helpers.
"""

import pandas as pd

from fast_plate_ocr.train.data.annotations import read_annotations_csv


def test_read_annotations_csv_fills_missing_plate_text(tmp_path) -> None:
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "image_path": ["a.png", "b.png"],
            "plate_text": [None, "AB123"],
        }
    ).to_csv(csv_path, index=False)

    annotations = read_annotations_csv(csv_path)

    assert annotations["plate_text"].tolist() == ["", "AB123"]


def test_read_annotations_csv_reads_plate_text_as_string(tmp_path) -> None:
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "image_path": ["a.png", "b.png"],
            "plate_text": [123, 7],
        }
    ).to_csv(csv_path, index=False)

    annotations = read_annotations_csv(csv_path)

    assert annotations["plate_text"].tolist() == ["123", "7"]
