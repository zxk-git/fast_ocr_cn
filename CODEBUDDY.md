# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

## Common Commands

### Setup and Dependencies
```bash
# Install with all dev/test/train/onnx deps (recommended)
make install

# Minimal PyTorch-only training environment
python3 scripts/plate_workflow.py setup --skip-verify

# Verify environment, data, and model forward pass
KERAS_BACKEND=torch \
FAST_PLATE_OCR_DATASET_ROOT=/path/to/FastOCRData \
python3 scripts/verify_pytorch_training_env.py
```

### Lint and Format
```bash
# Auto-format (sort imports + ruff format)
make format

# Run all linters (ruff, yamllint, pylint, mypy)
make lint

# Run individual linter
make ruff        # Ruff only
make pylint      # Pylint only
make mypy        # MyPy only
make check_format  # Check formatting without changing files
```

### Testing
```bash
# Run all tests (parallel)
make test
# which runs: uv run pytest -n auto

# Run a single test file
uv run pytest test/fast_lp_ocr/train/test_dataset.py

# Run a specific test function
uv run pytest test/fast_lp_ocr/train/test_config.py -k "test_plate_config"
```

### Training
```bash
# Quick validation run (3 epochs)
FAST_PLATE_OCR_DATASET_ROOT=/path/to/FastOCRData \
python3 scripts/plate_workflow.py train --quick --gpu 0

# Full training with CLI
fast-plate-ocr train \
  --model-config-file models/cct_s_v2.yaml \
  --plate-config-file config/cn_plate_config.yaml \
  --annotations /path/to/FastOCRData/train/annotations.csv \
  --val-annotations /path/to/FastOCRData/val/annotations.csv \
  --epochs 200 --batch-size 1024 --workers 16 \
  --output-dir trained_models/my_run

# Or via the convenience wrapper
FAST_PLATE_OCR_DATASET_ROOT=/path/to/FastOCRData \
FAST_PLATE_OCR_MODEL_CONFIG=models/cct_s_v2.yaml \
FAST_PLATE_OCR_EPOCHS=200 \
python3 scripts/plate_workflow.py train --gpu 0
```

### Evaluation
```bash
python3 scripts/test_single_image.py \
  --image /path/to/cropped_plate.jpg \
  --model /path/to/best.keras \
  --plate-config config/cn_plate_config.yaml \
  --keras-backend torch

# Evaluate full dataset (see README for evaluate_model.py usage)
```

### ONNX
```bash
# Export Keras model to ONNX (requires TensorFlow backend, see README caveat)
python3 scripts/plate_workflow.py export /path/to/best.keras
```

### Data Conversion
```bash
# Convert CBLPRD-330K
python3 data_convert/CBLPRD2Fastocr.py --dataset-root <path> --out-dir <path>

# Convert CCPD2019 / CCPD2020
python3 data_convert/CCPD2Fastocr.py --dataset-root <path> --out-dir <path> --workers 16
python3 data_convert/CCPD20202Fastocr.py --dataset-root <path> --out-dir <path> --workers 16

# Merge datasets
python3 data_convert/merge_cblprd_ccpd.py \
  --cblprd-root <path> --ccpd-root <path> --ccpd2020-root <path> --out-dir <path>
```

## Architecture Overview

### Dual Config System
There are **two independent `PlateConfig` implementations** serving different purposes:
- **`fast_plate_ocr.train.model.config.PlateConfig`** — Pydantic-based, frozen, used during training. Validates alphabet/padding, computes `vocabulary_size`, `pad_idx`, `num_channels`. Loaded via `load_plate_config_from_yaml()`.
- **`fast_plate_ocr.inference.config.PlateConfig`** — Plain dataclass-based, used during ONNX inference. Keeps dependencies minimal (no Pydantic). Loaded via `PlateConfig.from_yaml()`.

Both represent the same logical data but are intentionally separate to keep the inference package lightweight.

### Package Structure

```
fast_plate_ocr/
├── core/             # Image I/O, resize, preprocess, postprocess — no Keras/ONNX deps
│   ├── process.py    # read_plate_image, resize_image, preprocess_image, postprocess_output
│   └── types.py      # PlatePrediction, type aliases (ImageColorMode, PathLike, etc.)
├── train/            # Keras training pipeline
│   ├── model/
│   │   ├── config.py        # Pydantic PlateConfig for training
│   │   ├── model_schema.py  # Pydantic schemas for YAML model configs (CCTModelConfig, LayerConfig)
│   │   ├── model_builders.py # build_model() — assembles CCT model from configs
│   │   ├── layers.py        # Custom Keras layers: TransformerBlock, TokenReducer, DyT, MaxBlurPooling2D, etc.
│   │   ├── loss.py          # cce_loss, focal_cce_loss
│   │   └── metric.py        # cat_acc_metric, plate_acc_metric, top_3_k_metric, plate_len_acc_metric
│   ├── data/
│   │   ├── dataset.py       # PlateRecognitionPyDataset (Keras PyDataset subclass)
│   │   ├── annotations.py   # read_annotations_csv
│   │   └── augmentation.py  # default_train_augmentation (Albumentations pipeline)
│   └── utilities/
│       ├── utils.py         # target_transform (plate string -> one-hot)
│       └── backend_utils.py
├── inference/        # ONNX Runtime inference (minimal deps)
│   ├── plate_recognizer.py  # LicensePlateRecognizer class
│   └── config.py            # Dataclass-based PlateConfig
└── cli/              # Click-based CLI (fast-plate-ocr command)
    ├── cli.py               # Main CLI group
    ├── train.py             # fast-plate-ocr train subcommand
    ├── valid.py             # Validation subcommand
    ├── export.py            # ONNX export subcommand
    └── validate_dataset.py  # Dataset validation
```

### Model Architecture (CCT — Compact Convolutional Transformer)

The model is a **fixed-position classification** architecture (not CTC, not autoregressive). Each of N character positions independently outputs a probability distribution over the alphabet.

Pipeline: `Input(64×128×3) -> Rescaling -> Conv Stem (4×Conv+SiLU, MaxBlurPooling) -> 2×2 PatchExtractor -> PatchMLP + PositionEmbedding -> 5×TransformerBlock -> TokenReducer (512->8) -> 3×Post-Reduce TransformerBlock -> VocabularyProjection (Dense+Softmax) -> Output(8×75)`

Key architectural decisions:
- **DyT (Dynamic Tanh)** is used instead of LayerNorm/RMSNorm in the default `cct_s_v2` config.
- **StochasticDepth** is applied during training only, with dropout probability linearly increasing from 0 to the configured value across layers.
- **TokenReducer** uses learned query tokens with cross-attention to compress 512 visual tokens into 8 position tokens (one per character slot).
- **Position embeddings** are learnable and added to patch tokens before the transformer blocks — they preserve left-to-right character order.

### Configuration Files

| File | Purpose | Controlled by |
|------|---------|--------------|
| `config/cn_plate_config.yaml` | Character set, max slots (8), input dimensions (64×128 RGB), padding | User/training |
| `models/cct_s_v2.yaml` | Conv layers, transformer depth/heads/dims, dropout, normalization | User/training |
| Training output dir | Saves `model_config.yaml` + `plate_config.yaml` copies for reproducibility | Training script |

The two config types are paired — `model_config` defines the architecture, `plate_config` defines the data. Changing `alphabet` changes the final Dense layer size; changing `max_plate_slots` changes TokenReducer query count and output positions. Training, eval, and inference must use the same `plate_config.yaml`.

### Training Pipeline

1. `PlateConfig` and `ModelConfig` are loaded from YAML and validated via Pydantic
2. `PlateRecognitionPyDataset` reads annotations CSV, loads images via OpenCV, resizes, applies Albumentations augmentation
3. `build_model()` constructs the Keras model from configs
4. Training uses AdamW optimizer with cosine decay and optional EMA, early stopping, ModelCheckpoint
5. Outputs: `best.keras`, `last.keras`, `model_config.yaml`, `plate_config.yaml`, `hyper_params.json`, `training_log.csv`

### Inference Pipeline (ONNX Runtime)

1. `LicensePlateRecognizer` loads ONNX model and `PlateConfig`
2. `run()` accepts paths or numpy arrays, resizes via `core/process.py`, runs ONNX inference
3. `postprocess_output()` decodes logits to plate strings, optionally returning confidence and region

### Testing

Tests live in `test/fast_lp_ocr/` mirroring the source package structure. `test/conftest.py` provides shared fixtures (`dummy_dataset`, `dummy_plate_config`, `dummy_cct_model_config`) that create minimal configs suitable for unit tests. Tests use temp directories and small synthetic models to avoid requiring real data or GPU.

### Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `KERAS_BACKEND` | `torch` | Keras 3 backend (must be `torch`, not tensorflow) |
| `FAST_PLATE_OCR_DATASET_ROOT` | `/zxk/plate_ocr/plate/FastOCRData` | Training/validation data root |
| `FAST_PLATE_OCR_MODEL_CONFIG` | `models/cct_s_v2.yaml` | Model architecture config |
| `FAST_PLATE_OCR_PLATE_CONFIG` | `config/cn_plate_config.yaml` | Character set config |
| `FAST_PLATE_OCR_EPOCHS` | `200` | Training epochs |
| `FAST_PLATE_OCR_BATCH_SIZE` | `1024` | Batch size |
| `FAST_PLATE_OCR_LR` | `0.001` | Initial learning rate |
| `FAST_PLATE_OCR_OUTPUT_DIR` | `trained_models/cblprd_cct_s_v2_torch` | Model output directory |
| `FAST_PLATE_OCR_GPU` | unset | Physical GPU device number |

### Important Constraints

- **No TensorFlow**: The PyTorch-only training environment must NOT have TensorFlow installed. Environment setup scripts validate this. The ONNX export path currently requires TensorFlow (known limitation documented in README).
- **Same filesystem for hard links**: Data conversion scripts use hard links for images. Source datasets and output directories must reside on the same filesystem.
- **Config consistency**: Training, evaluation, and inference must all use the same `plate_config.yaml` — mismatched alphabet or max_plate_slots will produce garbage output.
- **Pre-cropped input**: The OCR model expects already-cropped license plate images. It does NOT include a plate detection/ROI extraction module.
