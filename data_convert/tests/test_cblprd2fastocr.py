import csv
import json
from pathlib import Path

import pytest
import yaml

from data_convert.CBLPRD2Fastocr import ConversionError, convert_dataset


def _write_image(root: Path, name: str, content: bytes) -> Path:
    path = root / "image" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_labels(root: Path, train_lines: list[str], val_lines: list[str]) -> None:
    label_dir = root / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (label_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _valid_dataset(root: Path) -> dict[str, Path]:
    standard = _write_image(root, "train_standard.jpg", b"standard-image")
    new_energy = _write_image(root, "train_new_energy.jpg", b"new-energy-image")
    double_layer = _write_image(root, "val_double_layer.jpg", b"raw-double-layer-image")
    _write_labels(
        root,
        [
            "CBLPRD-330k/train_standard.jpg 沪A12345 普通蓝牌",
            "CBLPRD-330k/train_new_energy.jpg 粤AD12345 新能源小型车",
        ],
        ["CBLPRD-330k/val_double_layer.jpg 甘N66LK挂 双层黄牌"],
    )
    return {"standard": standard, "new_energy": new_energy, "double_layer": double_layer}


def test_converts_to_fast_plate_ocr_hard_linked_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    sources = _valid_dataset(dataset_root)

    out_dir = convert_dataset(dataset_root)

    assert out_dir == dataset_root / "fast-plate-ocr"
    assert _read_csv(out_dir / "train" / "annotations.csv") == [
        {"image_path": "images/train_standard.jpg", "plate_text": "沪A12345"},
        {"image_path": "images/train_new_energy.jpg", "plate_text": "粤AD12345"},
    ]
    assert _read_csv(out_dir / "val" / "annotations.csv") == [
        {"image_path": "images/val_double_layer.jpg", "plate_text": "甘N66LK挂"}
    ]

    linked_double_layer = out_dir / "val" / "images" / "val_double_layer.jpg"
    assert linked_double_layer.exists()
    assert linked_double_layer.stat().st_ino == sources["double_layer"].stat().st_ino
    assert linked_double_layer.read_bytes() == b"raw-double-layer-image"
    assert (out_dir / "train" / "images" / "train_standard.jpg").stat().st_ino == sources["standard"].stat().st_ino

    config = yaml.safe_load((out_dir / "plate_config.yaml").read_text(encoding="utf-8"))
    observed_chars = set("沪A12345粤AD12345甘N66LK挂")
    assert config["max_plate_slots"] == 8
    assert config["alphabet"] == "".join(sorted(observed_chars)) + "_"
    assert config["pad_char"] == "_"
    assert "沪" in config["alphabet"]
    assert config["img_height"] == 48
    assert config["img_width"] == 128
    assert config["keep_aspect_ratio"] is True
    assert config["image_color_mode"] == "rgb"

    report = json.loads((out_dir / "conversion_report.json").read_text(encoding="utf-8"))
    assert report["hard_link_count"] == 3
    assert report["errors"] == []
    assert report["characters"] == sorted(observed_chars)
    assert report["plate_type_distribution"] == {
        "双层黄牌": 1,
        "新能源小型车": 1,
        "普通蓝牌": 1,
    }

    expected_paths = {
        "conversion_report.json",
        "plate_config.yaml",
        "train/annotations.csv",
        "train/images/train_new_energy.jpg",
        "train/images/train_standard.jpg",
        "val/annotations.csv",
        "val/images/val_double_layer.jpg",
    }
    assert {path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*") if path.is_file()} == expected_paths


def test_resolves_historical_source_prefix_to_image_basename(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    source = _write_image(dataset_root, "only_image_dir.jpg", b"image")
    _write_labels(
        dataset_root,
        ["CBLPRD-330k/only_image_dir.jpg 京A12345 单层黄牌"],
        ["CBLPRD-330k/only_image_dir_val.jpg 沪B12345 普通蓝牌"],
    )
    _write_image(dataset_root, "only_image_dir_val.jpg", b"val-image")

    out_dir = convert_dataset(dataset_root, out_dir=tmp_path / "converted")

    target = out_dir / "train" / "images" / "only_image_dir.jpg"
    assert target.stat().st_ino == source.stat().st_ino
    assert _read_csv(out_dir / "train" / "annotations.csv") == [
        {"image_path": "images/only_image_dir.jpg", "plate_text": "京A12345"}
    ]


@pytest.mark.parametrize(
    ("train_line", "missing_image"),
    [
        ("CBLPRD-330k/bad.jpg 沪A12345", False),
        ("CBLPRD-330k/missing.jpg 沪A12345 普通蓝牌", True),
    ],
)
def test_invalid_input_does_not_create_output(
    tmp_path: Path, train_line: str, missing_image: bool
) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    if not missing_image:
        _write_image(dataset_root, "bad.jpg", b"image")
    _write_image(dataset_root, "valid_val.jpg", b"image")
    _write_labels(
        dataset_root,
        [train_line],
        ["CBLPRD-330k/valid_val.jpg 京A12345 普通蓝牌"],
    )
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("train_line", "image_name"),
    [
        ("CBLPRD-330k/empty.jpg  普通蓝牌", "empty.jpg"),
        ("CBLPRD-330k/nested.jpg/unsafe.jpg 京A12345 普通蓝牌", "unsafe.jpg"),
        ("CBLPRD-330k/not_jpg.png 京A12345 普通蓝牌", "not_jpg.png"),
    ],
    ids=["empty_plate_text", "unsafe_path", "non_jpg"],
)
def test_invalid_plate_text_or_basename_does_not_create_output(
    tmp_path: Path, train_line: str, image_name: str
) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _write_image(dataset_root, image_name, b"image")
    _write_image(dataset_root, "valid_val.jpg", b"image")
    _write_labels(
        dataset_root,
        [train_line],
        ["CBLPRD-330k/valid_val.jpg 京A12345 普通蓝牌"],
    )
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_duplicate_output_name_within_split_does_not_create_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _write_image(dataset_root, "same.jpg", b"image")
    _write_image(dataset_root, "valid_val.jpg", b"image")
    _write_labels(
        dataset_root,
        [
            "CBLPRD-330k/same.jpg 京A12345 普通蓝牌",
            "CBLPRD-330k/same.jpg 沪B12345 普通蓝牌",
        ],
        ["CBLPRD-330k/valid_val.jpg 粤C12345 普通蓝牌"],
    )
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError, match="collision"):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_control_character_in_plate_text_does_not_create_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _write_image(dataset_root, "control.jpg", b"image")
    _write_image(dataset_root, "valid_val.jpg", b"image")
    _write_labels(
        dataset_root,
        ["CBLPRD-330k/control.jpg 京A\x0712345 普通蓝牌"],
        ["CBLPRD-330k/valid_val.jpg 沪B12345 普通蓝牌"],
    )
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError, match="control character"):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_yaml_nonprintable_plate_text_does_not_create_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _write_image(dataset_root, "nonprintable.jpg", b"image")
    _write_image(dataset_root, "valid_val.jpg", b"image")
    _write_labels(
        dataset_root,
        ["CBLPRD-330k/nonprintable.jpg 京A\ufffe12345 普通蓝牌"],
        ["CBLPRD-330k/valid_val.jpg 沪B12345 普通蓝牌"],
    )
    out_dir = tmp_path / "converted"

    with pytest.raises(ConversionError, match="YAML-nonprintable"):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert not out_dir.exists()


def test_existing_empty_output_stays_empty_when_hard_link_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _valid_dataset(dataset_root)
    out_dir = tmp_path / "converted"
    out_dir.mkdir()

    import data_convert.CBLPRD2Fastocr as converter

    original_link = converter.os.link
    calls = 0

    def fail_second_link(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced hard-link failure")
        original_link(source, target)

    monkeypatch.setattr(converter.os, "link", fail_second_link)

    with pytest.raises(ConversionError, match="could not create hard link"):
        convert_dataset(dataset_root, out_dir=out_dir)

    assert list(out_dir.iterdir()) == []


def test_collisions_and_nonempty_output_do_not_change_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "CBLPRD-330K"
    _write_image(dataset_root, "same.jpg", b"image")
    _write_labels(
        dataset_root,
        ["CBLPRD-330k/same.jpg 京A12345 普通蓝牌"],
        ["CBLPRD-330k/same.jpg 沪B12345 普通蓝牌"],
    )
    collision_out = tmp_path / "collision-output"

    with pytest.raises(ConversionError, match="collision"):
        convert_dataset(dataset_root, out_dir=collision_out)

    assert not collision_out.exists()

    nonempty_out = tmp_path / "nonempty-output"
    nonempty_out.mkdir()
    sentinel = nonempty_out / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        convert_dataset(dataset_root, out_dir=nonempty_out)

    assert sentinel.read_text(encoding="utf-8") == "keep"
