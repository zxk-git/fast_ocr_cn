"""
Tests for dataset module.
"""

import pathlib
import textwrap

import cv2
import numpy as np
import pandas as pd

from fast_plate_ocr.train.data.dataset import PlateRecognitionPyDataset
from fast_plate_ocr.train.model.config import PlateConfig, load_plate_config_from_yaml


def _write_plate_config(tmp_path: pathlib.Path, contents: str) -> PlateConfig:
    cfg_path = tmp_path / "plate_config.yaml"
    cfg_path.write_text(contents)
    return load_plate_config_from_yaml(cfg_path)


def _write_images(root: pathlib.Path, count: int) -> list[str]:
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rel_paths: list[str] = []
    for idx in range(count):
        img = (np.random.rand(32, 128, 3) * 255).astype("uint8")
        img_path = img_dir / f"img_{idx}.png"
        cv2.imwrite(str(img_path), img)
        rel_paths.append(str(img_path.relative_to(root)))
    return rel_paths


def _write_annotations(
    root: pathlib.Path,
    rel_paths: list[str],
    plate_texts: list[str],
    regions: list[str] | None = None,
) -> pathlib.Path:
    data = {"image_path": rel_paths, "plate_text": plate_texts}
    if regions is not None:
        data["plate_region"] = regions
    df = pd.DataFrame(data)
    csv_path = root / "annotations.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_dataset_len_and_shapes(dummy_dataset: pathlib.Path, dummy_plate_config: str, tmp_path: pathlib.Path) -> None:
    plate_config = _write_plate_config(tmp_path, dummy_plate_config)
    dataset = PlateRecognitionPyDataset(
        annotations_file=dummy_dataset,
        plate_config=plate_config,
        batch_size=2,
        shuffle=False,
    )

    assert len(dataset) == 2

    batch_x, batch_y = dataset[0]
    assert batch_x.shape == (2, plate_config.img_height, plate_config.img_width, plate_config.num_channels)
    assert batch_y["plate"].shape == (2, plate_config.max_plate_slots, plate_config.vocabulary_size)


def test_dataset_accepts_empty_plate_text(tmp_path: pathlib.Path, dummy_plate_config: str) -> None:
    plate_config = _write_plate_config(tmp_path, dummy_plate_config)
    root = tmp_path / "data"
    root.mkdir()

    rel_paths = _write_images(root, count=2)
    csv_path = _write_annotations(root, rel_paths, plate_texts=["", "AB"])

    dataset = PlateRecognitionPyDataset(
        annotations_file=csv_path, plate_config=plate_config, batch_size=2, shuffle=False
    )

    _, batch_y = dataset[0]
    plate_targets = batch_y["plate"]
    pad_idx = plate_config.pad_idx

    empty_plate = np.argmax(plate_targets[0], axis=-1)
    assert np.all(empty_plate == pad_idx)

    filled_plate = np.argmax(plate_targets[1], axis=-1)
    assert filled_plate[0] == plate_config.alphabet.index("A")
    assert filled_plate[1] == plate_config.alphabet.index("B")
    assert np.all(filled_plate[2:] == pad_idx)


def test_dataset_region_head_outputs_one_hot(tmp_path: pathlib.Path) -> None:
    plate_config_yaml = textwrap.dedent(
        """
        max_plate_slots: 4
        alphabet: 'AB_'
        pad_char: '_'
        img_height: 32
        img_width: 64
        plate_regions: ['AR', 'BR']
        """
    )
    plate_config = _write_plate_config(tmp_path, plate_config_yaml)
    root = tmp_path / "data"
    root.mkdir()

    rel_paths = _write_images(root, count=2)
    csv_path = _write_annotations(root, rel_paths, plate_texts=["AB", "A"], regions=["AR", "BR"])

    dataset = PlateRecognitionPyDataset(
        annotations_file=csv_path, plate_config=plate_config, batch_size=2, shuffle=False
    )

    assert dataset.region_recognition is True

    _, batch_y = dataset[0]
    assert "region" in batch_y
    expected = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.array_equal(batch_y["region"], expected)
