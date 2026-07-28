# Unified Python Workflow and Single-Image Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three Shell launchers with one Python CLI and add single-image inference for local ONNX and Keras models.

**Architecture:** Keep workflow orchestration and inference in two script-level modules. The workflow CLI builds argument lists and isolated child-process environments for `setup`, `train`, and `export`; the inference script shares image decoding and postprocessing while selecting an ONNX Runtime or Keras adapter from the model suffix.

**Tech Stack:** Python 3.10+, argparse, pathlib, subprocess, NumPy, ONNX Runtime, Keras 3, pytest

**Execution Status:** Tasks 1-5 and the task-specific acceptance checks in Task 6 are complete.
Focused tests and real ONNX/Keras smoke tests pass. The full repository suite remains unverified:
the available environments split pytest and training extras, and HUB inference tests require
network access plus a writable user cache.

---

## File Map

- Create `scripts/plate_workflow.py`: unified `setup`, `train`, and `export` CLI.
- Create `scripts/test_single_image.py`: suffix-based ONNX/Keras single-image inference.
- Modify `fast_plate_ocr/__init__.py`: keep the optional ONNX recognizer lazily loaded.
- Modify `test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py`: setup and export Python CLI contracts.
- Modify `test/fast_lp_ocr/cli/test_torch_launcher.py`: training Python CLI contracts.
- Create `test/fast_lp_ocr/cli/test_single_image_script.py`: inference dispatch, dtype, output, and errors.
- Modify `test/fast_lp_ocr/test_package_init.py`: make optional ONNX isolation deterministic.
- Delete `scripts/setup_pytorch_env.sh`, `scripts/train_cblprd_torch.sh`, and
  `scripts/export_onnx_torch.sh`: superseded launchers approved for replacement.

No Git commit step is included because the user has not authorized Git history changes.

### Task 1: Specify The Unified Workflow Contract

**Files:**

- Modify: `test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py`
- Modify: `test/fast_lp_ocr/cli/test_torch_launcher.py`
- Test: `test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py`
- Test: `test/fast_lp_ocr/cli/test_torch_launcher.py`

- [ ] **Step 1: Point launcher tests at the Python entry point**

Define this launcher in both test modules:

```python
import sys

WORKFLOW = PROJECT_ROOT / "scripts" / "plate_workflow.py"


def workflow_command(*args: str) -> list[str]:
    return [sys.executable, str(WORKFLOW), *args]
```

Replace setup invocations with `workflow_command("setup", "--skip-verify")`, export invocations
with `workflow_command("export", "--no-simplify")`, and train invocations with
`workflow_command("train", *args)`.

- [ ] **Step 2: Lock the retained export defaults**

Update the export assertions to require the current working-tree behavior:

```python
assert captured[0] == "tensorflow"
assert captured[1] == "export"
assert "--no-skip-validation" in captured
assert captured[captured.index("--onnx-opset-version") + 1] == "13"
assert captured[captured.index("--onnx-input-dtype") + 1] == "float32"
assert captured[captured.index("--onnx-data-format") + 1] == "channels_last"
assert captured[-1] == "--no-simplify"
```

- [ ] **Step 3: Run the focused tests and observe RED**

Run:

```bash
pytest test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py \
  test/fast_lp_ocr/cli/test_torch_launcher.py -q
```

Expected: FAIL because `scripts/plate_workflow.py` does not exist.

### Task 2: Implement The Unified Workflow CLI

**Files:**

- Create: `scripts/plate_workflow.py`
- Test: `test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py`
- Test: `test/fast_lp_ocr/cli/test_torch_launcher.py`

- [ ] **Step 1: Add parser and shared process helpers**

Implement these stable interfaces:

```python
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
ONNX_PACKAGES = (
    "onnx==1.17.0",
    "onnxruntime==1.23.2",
    "onnxscript==0.1.0",
    "onnxslim==0.1.82",
)


class WorkflowError(RuntimeError):
    pass


def python_bin() -> str:
    return os.environ.get("PYTHON_BIN", sys.executable)


def require_command(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise WorkflowError(f"Command not found: {command}")
    return resolved


def runtime_env(backend: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    pathlib.Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env
```

The parser must expose `setup`, `train`, and `export`. Use `parse_known_args` so unknown train and
export arguments can be appended after repository defaults.

- [ ] **Step 2: Implement setup without changing protected frameworks**

Implement `run_setup(skip_verify: bool) -> None` with list-form subprocess calls in this order:

```python
interpreter = require_command(python_bin())
assert_tensorflow_absent(interpreter)
versions_before = read_torch_versions(interpreter)
run([interpreter, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
     "-r", str(PROJECT_ROOT / "requirements" / "train-torch.txt")])
run([interpreter, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed", *ONNX_PACKAGES])
run([interpreter, "-m", "pip", "install", "--no-deps", "-e", str(PROJECT_ROOT)])
versions_after = read_torch_versions(interpreter)
if versions_after != versions_before:
    raise WorkflowError(f"PyTorch was unexpectedly changed: {versions_before} -> {versions_after}")
assert_tensorflow_absent(interpreter)
verify_imports(interpreter)
if not skip_verify:
    run([interpreter, str(PROJECT_ROOT / "scripts" / "verify_pytorch_training_env.py")],
        env=runtime_env("torch"))
```

The TensorFlow presence probe must treat exit code 0 as installed and exit code 1 as absent;
unexpected codes must be surfaced as errors.

- [ ] **Step 3: Implement train argument validation and command construction**

Implement the exact interface
`build_train_command(quick: bool, gpu: str | None, forwarded: list[str]) -> tuple[list[str], dict[str, str]]`.
`run_train` then executes its returned values as follows:

```python
def run_train(quick: bool, gpu: str | None, forwarded: list[str]) -> None:
    command, env = build_train_command(quick, gpu, forwarded)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
```

`build_train_command` must validate the four required files, reject non-decimal GPU IDs, preserve
all `FAST_PLATE_OCR_*` defaults from the Shell launcher, use three epochs in quick mode, set
`KERAS_BACKEND=torch`, and append `forwarded` last.

- [ ] **Step 4: Implement latest-model discovery and export construction**

Implement `build_export_command(model_arg: str | None, forwarded: list[str]) -> tuple[list[str], dict[str, str]]`
and these complete discovery/execution helpers:

```python
def find_latest_model(search_root: pathlib.Path) -> pathlib.Path | None:
    candidates = (path for path in search_root.rglob("best.keras") if path.is_file())
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def run_export(model_arg: str | None, forwarded: list[str]) -> None:
    command, env = build_export_command(model_arg, forwarded)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
```

The command defaults must contain:

```python
[
    cli, "export", "--model", str(model_path),
    "--plate-config-file", str(plate_config),
    "--format", "onnx", "--save-dir", str(output_dir),
    "--dynamic-batch", "--no-skip-validation",
    "--onnx-opset-version", "13", "--simplify",
    "--onnx-input-dtype", "float32",
    "--onnx-data-format", "channels_last",
    *forwarded,
]
```

Before launching, run the ONNX dependency import probe with `KERAS_BACKEND=tensorflow`.

- [ ] **Step 5: Convert domain errors to concise CLI failures**

`main(argv: list[str] | None = None) -> int` must return 0 on success, 1 for `WorkflowError` or a
failed child process, and 2 for parser validation failures. The module entry point must use
`raise SystemExit(main())`.

- [ ] **Step 6: Run workflow tests and observe GREEN**

Run:

```bash
pytest test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py \
  test/fast_lp_ocr/cli/test_torch_launcher.py -q
```

Expected: all setup, train, and export contract tests pass.

### Task 3: Specify Single-Image Inference Behavior

**Files:**

- Create: `test/fast_lp_ocr/cli/test_single_image_script.py`
- Test: `test/fast_lp_ocr/cli/test_single_image_script.py`

- [ ] **Step 1: Write model/config and dtype tests**

Add focused tests for these desired functions:

```python
def test_detects_supported_model_types() -> None:
    assert single_image.model_type(pathlib.Path("model.onnx")) == "onnx"
    assert single_image.model_type(pathlib.Path("model.keras")) == "keras"


def test_rejects_unsupported_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model format"):
        single_image.model_type(pathlib.Path("model.pt"))


@pytest.mark.parametrize(
    ("onnx_type", "expected"),
    [("tensor(uint8)", np.uint8), ("tensor(float)", np.float32)],
)
def test_casts_input_from_onnx_metadata(onnx_type: str, expected: type[np.generic]) -> None:
    batch = np.array([[[[255]]]], dtype=np.uint8)
    assert single_image.cast_onnx_input(batch, onnx_type).dtype == expected
```

Also test that `resolve_plate_config(model, None)` selects adjacent `plate_config.yaml`, and that
missing explicit or inferred configs raise `FileNotFoundError` naming the path.

- [ ] **Step 2: Write output normalization tests**

Cover Keras/ONNX output shapes without loading a runtime:

```python
def test_selects_named_plate_output() -> None:
    expected = np.ones((1, 8, 3), dtype=np.float32)
    assert np.array_equal(single_image.plate_output({"plate": expected}), expected)


def test_selects_first_sequence_output() -> None:
    expected = np.ones((1, 8, 3), dtype=np.float32)
    assert np.array_equal(single_image.plate_output([expected]), expected)
```

Add an error test for an empty output collection.

- [ ] **Step 3: Run the tests and observe RED**

Run:

```bash
pytest test/fast_lp_ocr/cli/test_single_image_script.py -q
```

Expected: FAIL because `scripts/test_single_image.py` does not exist.

### Task 4: Implement ONNX And Keras Single-Image Inference

**Files:**

- Create: `scripts/test_single_image.py`
- Test: `test/fast_lp_ocr/cli/test_single_image_script.py`

- [ ] **Step 1: Add dependency-light parsing and validation**

Keep Keras, ONNX Runtime, and `fast_plate_ocr` imports inside engine/preprocessing functions so
`--help` and validation errors do not require optional runtimes. Implement:

```python
def model_type(model_path: pathlib.Path) -> Literal["onnx", "keras"]:
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix == ".keras":
        return "keras"
    raise ValueError(f"Unsupported model format: {model_path.suffix or '<none>'}")


def resolve_plate_config(model_path: pathlib.Path, requested: pathlib.Path | None) -> pathlib.Path:
    config_path = requested if requested is not None else model_path.with_name("plate_config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Plate configuration not found: {config_path}")
    return config_path.resolve()
```

The CLI positional order is `IMAGE MODEL`; options are `--plate-config`, `--keras-backend`, and
`--device`. Validate image and model existence before loading optional packages.

- [ ] **Step 2: Implement shared preprocessing and decoding**

Implement `load_batch(image_path, config_path) -> tuple[np.ndarray, PlateConfig]` by calling
`PlateConfig.from_yaml`, `read_and_resize_plate_image`, and `preprocess_image`. Implement
`decode_plate(raw_output, config) -> str` by passing `raw_output`, `config.max_plate_slots`,
`config.alphabet`, and `config.pad_char` to `postprocess_output`, then returning the first
prediction's `plate` value.

- [ ] **Step 3: Implement ONNX dtype and provider adaptation**

Implement:

```python
ONNX_DTYPES = {"tensor(uint8)": np.uint8, "tensor(float)": np.float32}


def cast_onnx_input(batch: np.ndarray, input_type: str) -> np.ndarray:
    dtype = ONNX_DTYPES.get(input_type)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input type: {input_type}")
    return batch.astype(dtype, copy=False)
```

`run_onnx` must select CPU, CUDA, or all available providers from `--device`, inspect the first
input's `name` and `type`, run the model, prefer the output named `plate`, and otherwise use the
first output.

- [ ] **Step 4: Implement delayed Keras loading**

Before importing Keras, set `os.environ["KERAS_BACKEND"]` from the parsed option. Import
`fast_plate_ocr.train.model.layers` to register custom serializable layers, then call:

```python
model = keras.models.load_model(model_path, compile=False)
outputs = model(batch, training=False)
return plate_output(outputs)
```

`plate_output` must accept a `{"plate": tensor}` mapping, a non-empty list/tuple, or a single
tensor and return the selected value through `np.asarray(selected_output)`.

- [ ] **Step 5: Print a stable single-image result**

On success, print exactly these two lines:

```text
model_type=onnx
plate=<decoded text>
```

or `model_type=keras`. Convert validation/runtime errors into a concise `error: <reason>` line on
stderr and a non-zero exit code.

- [ ] **Step 6: Run single-image unit tests and observe GREEN**

Run:

```bash
pytest test/fast_lp_ocr/cli/test_single_image_script.py -q
```

Expected: all model selection, dtype, output, and error tests pass.

### Task 5: Remove Superseded Shell Launchers

**Files:**

- Delete: `scripts/setup_pytorch_env.sh`
- Delete: `scripts/train_cblprd_torch.sh`
- Delete: `scripts/export_onnx_torch.sh`

- [ ] **Step 1: Confirm no active test still launches a Shell script**

Run:

```bash
rg -n 'setup_pytorch_env\.sh|train_cblprd_torch\.sh|export_onnx_torch\.sh' test scripts
```

Expected: matches exist only in the three files about to be removed or in no active test.

- [ ] **Step 2: Delete the three approved replacement targets**

Remove only the three Shell files listed above. Preserve the user's current export defaults in
the Python command before removing `export_onnx_torch.sh`.

- [ ] **Step 3: Verify the Python CLI help surfaces**

Run:

```bash
python scripts/plate_workflow.py --help
python scripts/plate_workflow.py setup --help
python scripts/plate_workflow.py train --help
python scripts/plate_workflow.py export --help
python scripts/test_single_image.py --help
```

Expected: every command exits 0 and documents its supported arguments.

### Task 6: Integration Verification And Quality Gate

**Files:**

- Verify: `scripts/plate_workflow.py`
- Verify: `scripts/test_single_image.py`
- Verify: `test/fast_lp_ocr/cli/`

- [ ] **Step 1: Run all focused tests**

Run:

```bash
pytest test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py \
  test/fast_lp_ocr/cli/test_torch_launcher.py \
  test/fast_lp_ocr/cli/test_single_image_script.py -q
```

Expected: all focused tests pass without warnings.

- [ ] **Step 2: Run the CLI test directory regression suite**

Run:

```bash
pytest test/fast_lp_ocr/cli -q
```

Expected: all CLI tests pass.

- [ ] **Step 3: Run Ruff and syntax compilation**

Run:

```bash
ruff check scripts/plate_workflow.py scripts/test_single_image.py \
  test/fast_lp_ocr/cli/test_pytorch_shell_scripts.py \
  test/fast_lp_ocr/cli/test_torch_launcher.py \
  test/fast_lp_ocr/cli/test_single_image_script.py
python -m py_compile scripts/plate_workflow.py scripts/test_single_image.py
```

Expected: both commands exit 0 with no findings.

- [ ] **Step 4: Smoke-test the real ONNX model**

Run:

```bash
python scripts/test_single_image.py \
  test/assets/test_plate_1.png \
  trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/best.onnx \
  --device cpu
```

Expected: exit 0, `model_type=onnx`, and a decoded `plate=` line. The model input is detected as
float32 and receives a float32 batch.

- [ ] **Step 5: Smoke-test the real Keras model**

Run:

```bash
python scripts/test_single_image.py \
  test/assets/test_plate_1.png \
  trained_models/cblprd_cct_s_v2_torch/2026-07-27_11-57-15/best.keras \
  --keras-backend torch
```

Expected: exit 0, `model_type=keras`, and a decoded `plate=` line.

- [ ] **Step 6: Inspect scope, observability, and stale references**

Run:

```bash
git diff --check
git status --short
rg -n 'setup_pytorch_env\.sh|train_cblprd_torch\.sh|export_onnx_torch\.sh' \
  --glob '!docs/superpowers/**' .
```

Expected: no whitespace errors; only intended files are changed; no active code or current tests
reference deleted scripts. Confirm errors contain actionable paths/reasons, progress output does
not expose sensitive data, complex conversions have concise comments, and no debug prints remain.
