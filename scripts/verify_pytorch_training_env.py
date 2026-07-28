#!/usr/bin/env python3
import importlib.metadata as metadata
import importlib.util
import os
import pathlib

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = pathlib.Path("/zxk/plate_ocr/plate/CBLPRD-330K/fast-plate-ocr")


def validate_annotations(csv_path: pathlib.Path, alphabet: set[str], max_slots: int) -> int:
    annotations = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_columns = {"image_path", "plate_text"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing_columns)}")

    too_long = annotations["plate_text"].str.len() > max_slots
    if too_long.any():
        raise ValueError(f"{csv_path}: plate text exceeds {max_slots} slots")

    observed_chars = set("".join(annotations["plate_text"]))
    unsupported = observed_chars - alphabet
    if unsupported:
        raise ValueError(f"{csv_path}: characters outside the configured alphabet: {sorted(unsupported)}")

    csv_root = csv_path.parent
    missing_image = next(
        (path for path in annotations["image_path"] if not (csv_root / path).is_file()),
        None,
    )
    if missing_image:
        raise ValueError(f"{csv_path}: missing referenced image {missing_image}")

    image_dir = csv_root / "images"
    image_count = sum(path.is_file() for path in image_dir.iterdir())
    if image_count != len(annotations):
        raise ValueError(f"{csv_path}: {len(annotations)} rows but {image_count} images")
    return len(annotations)


def main() -> None:
    if os.environ.get("KERAS_BACKEND") != "torch":
        raise RuntimeError("Set KERAS_BACKEND=torch before running verification")
    if importlib.util.find_spec("tensorflow") is not None:
        raise RuntimeError("TensorFlow is installed; expected a PyTorch-only training environment")

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    import keras
    import numpy as np
    import torch
    import torchvision

    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.model.model_builders import build_model
    from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml

    if metadata.version("torch") != "2.4.0" or metadata.version("torchvision") != "0.19.0":
        raise RuntimeError("Protected PyTorch package versions changed")
    if keras.backend.backend() != "torch":
        raise RuntimeError(f"Unexpected Keras backend: {keras.backend.backend()}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch accelerator is not available")

    dataset_root = pathlib.Path(os.environ.get("FAST_PLATE_OCR_DATASET_ROOT", DEFAULT_DATASET_ROOT))
    plate_config = load_plate_config_from_yaml(PROJECT_ROOT / "config" / "cn_plate_config.yaml")
    model_config_path = pathlib.Path(
        os.environ.get("FAST_PLATE_OCR_MODEL_CONFIG", PROJECT_ROOT / "models" / "cct_s_v2.yaml")
    )
    model_config = load_model_config_from_yaml(model_config_path)

    train_rows = validate_annotations(
        dataset_root / "train" / "annotations.csv",
        set(plate_config.alphabet),
        plate_config.max_plate_slots,
    )
    val_rows = validate_annotations(
        dataset_root / "val" / "annotations.csv",
        set(plate_config.alphabet),
        plate_config.max_plate_slots,
    )

    model = build_model(model_config, plate_config, enable_region_head=False)
    channels = 3 if plate_config.image_color_mode == "rgb" else 1
    sample = np.zeros((1, plate_config.img_height, plate_config.img_width, channels), dtype=np.float32)
    with torch.no_grad():
        model(sample, training=False)

    print(f"keras_backend={keras.backend.backend()}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"train_rows={train_rows}")
    print(f"val_rows={val_rows}")
    print("model_forward=ok")


if __name__ == "__main__":
    main()
