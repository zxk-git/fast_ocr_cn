# PyTorch Training Environment Design

## Goal

Prepare the current `fast-plate-ocr` checkout to train on the converted
CBLPRD-330K dataset with the Keras 3 PyTorch backend. Preserve the working
vendor PyTorch installation and do not install or use TensorFlow.

## Existing Environment

- Python: `/usr/local/bin/python3` (Python 3.10)
- PyTorch: `2.4.0`
- TorchVision: `0.19.0`
- Accelerator: `PPU-ZW810E`, exposed through `torch.cuda`
- Dataset root:
  `/zxk/plate_ocr/plate/CBLPRD-330K/fast-plate-ocr`
- Training rows/images: 325,005
- Validation rows/images: 17,105
- Plate lengths: 7 or 8 characters
- Plate configuration: `config/cn_plate_config.yaml`

The existing plate configuration matches the dataset conversion report:
it has eight output slots, contains every observed character, and uses `_`
as the padding character.

## Chosen Approach

Use the current system Python environment. Install an explicit minimal set of
training dependencies, then install this checkout in editable mode without
dependency resolution. Do not use the project's `train` extra because that
extra currently includes TensorFlow and TensorFlow-specific export packages.

Preserve the installed `torch==2.4.0` and `torchvision==0.19.0`. Package
installation must not upgrade, downgrade, or replace either package.

## Dependencies

Install the packages required by the training and validation paths, including:

- Keras 3 (`>=3.10.0,<4`)
- Albumentations
- Click
- Matplotlib
- OpenCV headless
- Pandas
- Pillow
- Pydantic 2 (`>=2.5.2,<3`)
- PyYAML
- Rich
- scikit-learn
- TensorBoard
- tqdm

Install the local project with `pip install --no-deps -e .` after these direct
dependencies. TensorBoard is a logging client and does not require TensorFlow
for Keras' PyTorch training path.

Do not install these packages as part of environment setup:

- TensorFlow
- tf2onnx
- ai-edge-litert
- Core ML Tools
- ONNX export tooling

Those packages are unrelated to the requested training path or are used only
by optional export commands.

## Backend Selection And Training Entry Point

Add a repository-local launcher for CBLPRD-330K training. The launcher sets
`KERAS_BACKEND=torch` before Python or Keras starts, so backend selection does
not depend on a user's global Keras configuration.

The launcher will use:

- `models/cct_s_v1.yaml` as the initial model configuration
- `config/cn_plate_config.yaml` as the plate configuration
- The converted training and validation CSV files under the dataset root
- A repository-local output directory for model artifacts
- A repository-local TensorBoard log directory when logging is enabled

Runtime tuning values such as epochs, batch size, worker count, mixed
precision, and learning rate remain command-line arguments rather than being
hard-coded as machine assumptions. The launcher provides conservative defaults
and forwards additional arguments to the project's training CLI.

## Data Flow

The training CLI reads each annotation CSV, resolves every image path relative
to that CSV, applies the configured resize and augmentation pipeline, and feeds
batches to a Keras model using the PyTorch backend. Checkpoints and logs are
written inside the repository; source images and annotations remain read-only
in the external dataset directory.

## Failure Handling

Environment verification stops with a clear error when:

- `KERAS_BACKEND` does not resolve to `torch`
- TensorFlow is importable in the configured environment
- PyTorch cannot see the accelerator
- Keras cannot build the selected model or complete a forward pass
- Dataset CSV files, referenced image files, or required CSV columns are missing
- Dataset text contains characters outside the configured alphabet or exceeds
  eight plate slots

The setup must not remove existing packages or alter the external dataset.

## Verification

After installation:

1. Confirm the installed PyTorch and TorchVision versions are unchanged.
2. Confirm `tensorflow` is not installed or importable.
3. Import Keras with `KERAS_BACKEND=torch` and confirm
   `keras.backend.backend()` returns `torch`.
4. Confirm `torch.cuda.is_available()` is true and report the active device.
5. Load both YAML configurations, build the model, and complete one forward
   pass with a correctly shaped synthetic batch.
6. Verify CSV row counts match image counts for both splits and validate the
   annotation schema, plate lengths, alphabet coverage, and referenced paths.
7. Confirm the training launcher and CLI help execute successfully.

Full model training is outside this environment-setup task.
