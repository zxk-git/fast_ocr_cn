# Unified Python Workflow and Single-Image Inference Design

## Goal

Replace the three Shell launchers in `scripts/` with one Python CLI, and add a standalone
single-image inference script that supports local ONNX and Keras models.

## Scope

The implementation will:

- replace `setup_pytorch_env.sh`, `train_cblprd_torch.sh`, and `export_onnx_torch.sh` with
  `scripts/plate_workflow.py`;
- preserve the current environment-variable overrides and pass-through CLI arguments;
- preserve the working-tree ONNX export defaults: TensorFlow Keras backend, float32 NHWC input,
  dynamic batch, validation enabled, opset 13, and simplification enabled;
- add `scripts/test_single_image.py` for one-image inference with `.onnx` and `.keras` files;
- migrate launcher tests from Shell-specific assertions to Python CLI behavior tests;
- retain lazy loading of the public ONNX recognizer so Keras and training submodules do not
  require a working ONNX Runtime installation;
- verify both model formats with the local trained model assets.

The implementation will not add Keras inference to the package's public
`LicensePlateRecognizer` API or change training/export internals.

## Unified Workflow CLI

`scripts/plate_workflow.py` will expose three subcommands:

```text
python scripts/plate_workflow.py setup [--skip-verify]
python scripts/plate_workflow.py train [--quick] [--gpu ID] [train options]
python scripts/plate_workflow.py export [MODEL.keras] [export options]
```

The CLI will use `argparse`, `pathlib`, `os.environ`, `shutil.which`, and list-form
`subprocess.run` calls. It will not execute constructed shell command strings.

### Setup

The `setup` command will preserve the existing PyTorch-only environment workflow:

- select the child interpreter from `PYTHON_BIN`, defaulting to the current interpreter;
- reject environments containing TensorFlow;
- record the installed PyTorch and torchvision versions;
- install `requirements/train-torch.txt`, the pinned ONNX packages, and the project in editable
  mode without dependencies;
- verify that PyTorch versions did not change;
- verify Keras uses the Torch backend and CUDA is available;
- run `verify_pytorch_training_env.py` unless `--skip-verify` is supplied.

The TensorFlow-based `export` command is intentionally allowed to run from a separate environment.
The setup command remains focused on the existing training environment and does not install
TensorFlow.

### Train

The `train` command will preserve the current dataset, model, plate configuration, output,
epoch, batch size, optimizer, worker, and early-stopping defaults. It will:

- validate required config and annotation files before launching;
- validate `--gpu` as a non-negative integer;
- set `KERAS_BACKEND=torch`, `NO_ALBUMENTATIONS_UPDATE=1`, and the optional
  `CUDA_VISIBLE_DEVICES` value;
- apply quick-mode defaults without forwarding `--quick` to `fast-plate-ocr`;
- append unknown train options after defaults so explicit caller overrides continue to win.

### Export

The `export` command will accept an optional model path. If omitted, it will select the newest
`best.keras` below `FAST_PLATE_OCR_TRAIN_OUTPUT_DIR`. It will:

- prefer `FAST_PLATE_OCR_MODEL_PATH` when no positional model is provided;
- discover `plate_config.yaml` beside the model before falling back to the repository config;
- create and resolve the output directory;
- check the CLI and required ONNX export imports;
- set `KERAS_BACKEND=tensorflow` and the existing auxiliary environment variables;
- invoke `fast-plate-ocr export` with float32, channels-last, dynamic-batch, validation, opset 13,
  and simplification defaults;
- append caller options last so they may override defaults.

## Single-Image Inference

`scripts/test_single_image.py` will use this interface:

```text
python scripts/test_single_image.py IMAGE MODEL [--plate-config PATH]
    [--keras-backend {torch,tensorflow,jax}] [--device {auto,cpu,cuda}]
```

The model suffix selects the engine. Only `.onnx` and `.keras` are accepted. If
`--plate-config` is omitted, the script will use `plate_config.yaml` in the model directory.

Shared preprocessing will read the plate config, load and resize the image with the project's
existing image helpers, add a batch dimension, and preserve channels-last layout.

For ONNX models, the script will create an ONNX Runtime session, inspect the model input metadata,
and cast the batch to `uint8` for `tensor(uint8)` or `float32` for `tensor(float)`. Other input
types will fail with a clear error. Provider selection will follow `--device`.

For Keras models, the script will set `KERAS_BACKEND` before importing Keras, import the project's
custom model layers to register serializable types, load the model without compilation, and run
inference. The default Keras backend will be `torch`, matching the trained model workflow, while
the option permits TensorFlow or JAX environments.

Both engines will normalize dict/list/tensor outputs into the plate-head NumPy array and decode it
with `postprocess_output`. The command will print the detected model type and decoded plate text.
Failures will identify missing files, unsupported suffixes, missing adjacent configuration,
unsupported ONNX input types, and missing runtime dependencies.

## Testing

Tests will be written before implementation and will cover:

- setup command construction and preservation of installed PyTorch versions;
- train defaults, quick mode, GPU selection, forwarding, and invalid GPU IDs;
- export latest-model discovery, adjacent plate config discovery, TensorFlow backend, and retained
  float32/opset/validation defaults;
- `.onnx` and `.keras` dispatch;
- ONNX `tensor(uint8)` and `tensor(float)` input conversion;
- Keras output normalization and single-image decoding;
- unsupported model suffix and missing input/config errors.

After unit tests pass, smoke verification will invoke the single-image script against the local
`best.onnx` and `best.keras` files using one local image. If an installed runtime or backend is
unavailable, that limitation will be reported rather than hidden.

## File Changes

- Create `scripts/plate_workflow.py`.
- Create `scripts/test_single_image.py`.
- Modify `fast_plate_ocr/__init__.py` to preserve optional ONNX Runtime isolation.
- Replace Shell-oriented tests under `test/fast_lp_ocr/cli/` with Python CLI tests.
- Stabilize the optional-runtime regression in `test/fast_lp_ocr/test_package_init.py`.
- Add focused single-image inference tests under `test/fast_lp_ocr/cli/`.
- Delete the three superseded `.sh` files after their behavior is covered.
