import csv
import json
from pathlib import Path

import pytest

from data_convert.merge_cblprd_ccpd import ConversionError, merge_datasets


CCPD_GROUPS = (
    "ccpd_base",
    "ccpd_blur",
    "ccpd_challenge",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
)


def _write_annotations(root: Path, rows: list[tuple[str, str]]) -> list[Path]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for image_name, plate_text in rows:
        source = images_dir / image_name
        source.write_bytes(f"image:{image_name}".encode())
        sources.append(source)
    with (root / "annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "plate_text"])
        writer.writeheader()
        writer.writerows(
            {"image_path": f"images/{name}", "plate_text": text}
            for name, text in rows
        )
    return sources


def _write_ccpd_dataset(root: Path) -> dict[str, list[Path]]:
    sources = {}
    for group in CCPD_GROUPS:
        rows = [("first.jpg", "京A12345"), ("second.jpg", "津B12345")]
        if group == "ccpd_base":
            rows = [("base_train.jpg", "沪B12345"), ("base_val.jpg", "粤C12345")]
        sources[group] = _write_annotations(root / group, rows)
    return sources


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_sources(tmp_path: Path) -> tuple[Path, Path, dict[str, list[Path]]]:
    cbl_root = tmp_path / "cblprd"
    _write_annotations(cbl_root / "train", [("same.jpg", "冀D12345")])
    _write_annotations(cbl_root / "val", [("same.jpg", "鲁E12345")])
    ccpd_root = tmp_path / "ccpd"
    ccpd_sources = _write_ccpd_dataset(ccpd_root)
    return cbl_root, ccpd_root, ccpd_sources


def test_merges_datasets_excludes_blur_and_splits_every_ccpd_group(tmp_path: Path) -> None:
    cbl_root, ccpd_root, ccpd_sources = _build_sources(tmp_path)
    out_dir = tmp_path / "merged"

    result = merge_datasets(
        cbl_root,
        ccpd_root,
        out_dir=out_dir,
        ccpd_val_ratio=0.5,
    )

    assert result == out_dir
    train_rows = _read_rows(out_dir / "train" / "annotations.csv")
    val_rows = _read_rows(out_dir / "val" / "annotations.csv")
    assert {row["image_path"] for row in train_rows} >= {
        "images/cblprd__train__same.jpg",
    }
    assert {row["image_path"] for row in val_rows} >= {
        "images/cblprd__val__same.jpg",
    }
    for group in CCPD_GROUPS:
        if group == "ccpd_blur":
            continue
        prefix = f"images/ccpd__{group}__"
        assert sum(row["image_path"].startswith(prefix) for row in train_rows) == 1
        assert sum(row["image_path"].startswith(prefix) for row in val_rows) == 1
    assert not any("blur" in row["image_path"] for row in train_rows + val_rows)
    linked = next((out_dir / split / "images" / "ccpd__ccpd_challenge__first.jpg")
                  for split in ("train", "val")
                  if (out_dir / split / "images" / "ccpd__ccpd_challenge__first.jpg").exists())
    assert linked.stat().st_ino == ccpd_sources["ccpd_challenge"][0].stat().st_ino

    report = json.loads((out_dir / "merge_report.json").read_text(encoding="utf-8"))
    assert report["excluded_ccpd_groups"] == ["ccpd_blur"]
    assert report["ccpd_validation_ratio"] == 0.5
    assert report["splits"]["train"]["total_rows"] == 8
    assert report["splits"]["val"]["total_rows"] == 8
    assert (out_dir / "plate_config.yaml").read_text(encoding="utf-8") == (
        Path("config/cn_plate_config.yaml").read_text(encoding="utf-8")
    )


def test_rejects_group_with_fewer_than_two_samples(tmp_path: Path) -> None:
    cbl_root, ccpd_root, _ = _build_sources(tmp_path)
    _write_annotations(ccpd_root / "ccpd_base", [("only.jpg", "沪B12345")])
    out_dir = tmp_path / "merged"

    with pytest.raises(ConversionError, match="at least two"):
        merge_datasets(cbl_root, ccpd_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_rejects_nonempty_output_without_modifying_it(tmp_path: Path) -> None:
    cbl_root, ccpd_root, _ = _build_sources(tmp_path)
    out_dir = tmp_path / "merged"
    out_dir.mkdir()
    sentinel = out_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        merge_datasets(cbl_root, ccpd_root, out_dir=out_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep"
