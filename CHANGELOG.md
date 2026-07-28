# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-03-13

### Added

- New `cct-xs-v2-global-model` and `cct-s-v2-global-model` with plate region recognition support for 65+ countries.
- Optional region-aware training, validation, and inference flow, including `plate_region` dataset support.
- New region evaluation metrics, including `val_region_macro_f1`, plus per-region evaluation in validation.
- New CCT v2 model configs, parameterized plate configs, and focal loss support for region classification.
- Inference now returns `PlatePrediction`, exposing region outputs when available.
- Export pipeline improvements for multi-output models, plus support for selecting the ONNX opset version.
- Expanded dataset validation and annotation checks, including region validation and warnings on unexpected columns.

### Changed

- V2 pre-trained models were trained on roughly 3x more data than the v1 generation.
- V2 training now includes empty plates built from backgrounds, noise, textures, and other padded-plate negatives.
- Region recognition now includes an `Unknown` class trained mainly with synthetic data.
- Updated the new CCT v2 models to use `silu` instead of `gelu` to avoid export issues with some library versions.
- Added the corrected `attention_layout` behavior so split projection dimensions are distributed per head instead of reusing the full `projection_dim`.
- The shipped v2 `xs` and `s` pre-trained models both exceed `0.99` `val_region_macro_f1` on a held-out validation split with more than `114_000` samples.
- Transformer blocks and training defaults are more configurable, including projection validation, loss weighting, and milder augmentations.
- Inference now removes the pad character by default from decoded output.
- TFLite export now uses LiteRT via `ai-edge-litert` following TensorFlow deprecation changes.

### Fixed

- Fixed region recognition mismatches during validation.
- Fixed `EarlyStopping` metric selection when training single-head models.
- Fixed learning-rate decay step calculation to account for warmup steps.

## [1.0.2] - 2025-09-03

### Added

- ReLU xs/s model versions
- Use CSVLogger logger callback by default
- Replace `opencv-python` with `opencv-python-headless`

## [1.0.0] - 2025-06-07

### Added

- Inference now works smoothly with different onnxruntime variants like `onnxruntime-gpu`, `onnxruntime-openvino`, etc.
- Support for building and customizing CCT (Compact Convolutional Transformer) models from YAML configs.
- New model building logic, allows users to build custom-based architectures while validating it with Pydantic.
- New metric `val_plate_len_acc`.
- Added support for categorical focal loss.
- Added more test coverage (configs, train scripts, etc.).
- New `validate_dataset.py` script to help check datasets before training.
- New `dataset_stats.py` script to display dataset statistics.
- Export script now officially supports more formats like TFLite and CoreML.
- New plate config support: `keep_aspect_ratio`, `interpolation`, `image_color_mode` and `padding_color`.
- New default augmentation for RGB image mode.
- New default models, trained with much more data.
- Added examples and more docs.

### Changed

- Visualize augmentation script now respects config-based preprocessing.
- Improved plate config validation.
- `ONNXPlateRecognizer` is now called `LicensePlateRecognizer`.

## [0.3.0] - 2024-12-08

### Added

- New Global model using MobileViTV2 trained with data from +65 countries, with 85k+ plates 🚀 .

[0.2.0]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.2.0...v0.3.0

## [0.2.0] - 2024-10-14

### Added

- New European model using MobileViTV2 - trained on +40 countries 🚀 .
- Added more logging to train script.

[0.2.0]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.1.6...v0.2.0

## [0.1.6] - 2024-05-09

### Added

- Add new Argentinian model trained with more (synthetic) data.
- Add option to visualize only predictions which have low char prob.
- Add onnxsim for simplifying ONNX model when exporting.

[1.1.0]: https://github.com/ankandrew/fast-plate-ocr/compare/v1.0.2...v1.1.0
[0.1.6]: https://github.com/ankandrew/fast-plate-ocr/compare/v0.1.5...v0.1.6
