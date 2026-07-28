# Training the OCR Model

The `train` script launches the end-to-end training process for a license plate OCR model, using:

- A user-defined **model architecture** (`model_config.yaml`)
- A **plate configuration** (`plate_config.yaml`)
- A training/validation dataset in CSV format
- Optional augmentation, loss, optimizer and callback settings

---

## Basic Usage

```shell
# You can set the backend to either TensorFlow, JAX or PyTorch
KERAS_BACKEND=tensorflow fast-plate-ocr train \
  --model-config-file models/cct_s_v2.yaml \
  --plate-config-file config/latin_plates.yaml \
  --annotations data/train.csv \
  --val-annotations data/val.csv \
  --epochs 150 \
  --batch-size 64 \
  --output-dir trained_models/
```

???+ warning "Keras Backend"
    Make sure you have **installed** a **backend** for **training**. See [Backend](../backend.md) section for more details.

???+ tip
    Use `--weights-path <path/to/model.keras>` when fine-tuning. All pre-trained models are uploaded to this
    [GH release > Assets](https://github.com/ankandrew/fast-plate-ocr/releases/tag/arg-plates).

---

## Region Recognition (Optional)

Region recognition is enabled automatically when:

1. Your CSV includes a `plate_region` column.
2. Your plate config defines `plate_regions`.

When enabled, you can tune region loss and weighting via:
`--region-loss`, `--region-loss-weight`, `--region-focal-alpha`, and `--region-focal-gamma`.

## Dataset Validation (Optional)

You can pre-validate train and val CSVs before training to catch missing images, corrupt files,
invalid chars, and length issues:

```shell
fast-plate-ocr train \
  --model-config-file models/cct_s_v2.yaml \
  --plate-config-file config/latin_plates.yaml \
  --annotations data/train.csv \
  --val-annotations data/val.csv \
  --validate-dataset warn
```

Use `--validate-dataset error` to abort on errors, or adjust size thresholds with
the standalone `fast-plate-ocr validate-dataset` command.

## Logging and Checkpoints

The script saves:

- Best model (based on monitored metric)
- Final model weights
- Training parameters and configs
- Optional TensorBoard logs

---

## Metrics Tracked

The model is compiled with:

- `plate_acc`: Fully correct plate predictions
- `cat_acc`: Character-level accuracy
- `top_3_k`: Accuracy if target is in top-3 predictions
- `plate_len_acc`: Plate length matches

See [**Metrics**](../metrics.md) section for more details about metrics.

??? note "View all CLI flags"
    The CLI supports over 30 options. You can view them with:

    ```bash
    fast-plate-ocr train --help
    ```
