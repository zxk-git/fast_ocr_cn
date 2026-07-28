# PyTorch Training Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure the current Python installation to train CBLPRD-330K with fast-plate-ocr through the Keras 3 PyTorch backend, without installing TensorFlow or replacing the vendor PyTorch packages.

**Architecture:** Keep environment dependencies, training invocation, and verification as separate units. A pinned requirements file documents the TensorFlow-free environment, a shell launcher owns stable paths and backend selection, and a Python verifier checks the installed backend, accelerator, dataset contract, and model forward pass before training begins.

**Tech Stack:** Python 3.10, PyTorch 2.4.0, Keras 3, Click, Albumentations, Pandas, Bash, unittest

---

## File Map

- Create `requirements/train-torch.txt`: direct training dependencies with no TensorFlow or export packages.
- Create `scripts/train_cblprd_torch.sh`: stable PyTorch training launcher for the converted CBLPRD-330K dataset.
- Create `scripts/verify_pytorch_training_env.py`: reusable environment, data, and model smoke verification.
- Modify `fast_plate_ocr/__init__.py`: lazy-load the optional ONNX inference class so training imports do not require ONNX Runtime.
- Create `test/fast_lp_ocr/cli/test_torch_launcher.py`: launcher environment and argument contract tests.
- Create `test/fast_lp_ocr/cli/test_verify_pytorch_training_env.py`: focused annotation validation tests.
- Create `test/fast_lp_ocr/test_package_init.py`: regression coverage for training imports without ONNX Runtime.
- Modify `.gitignore`: ignore the launcher's TensorBoard output directory.

### Task 1: Define The TensorFlow-Free Dependency Set

**Files:**
- Create: `requirements/train-torch.txt`

- [ ] **Step 1: Record the protected framework versions before installation**

Run:

```bash
python3 - <<'PY'
import importlib.util
import torch
import torchvision

print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"tensorflow_installed={importlib.util.find_spec('tensorflow') is not None}")
PY
```

Expected: `torch=2.4.0`, `torchvision=0.19.0`, and `tensorflow_installed=False`.

- [ ] **Step 2: Create the explicit dependency file**

Create `requirements/train-torch.txt` with:

```text
# Keras training dependencies for the existing PyTorch/PPU environment.
# Keep NumPy on the 1.x ABI to minimize risk to the vendor PyTorch build.
numpy>=1.24.4,<2
scipy==1.15.3
keras==3.12.1

# The PPU package index replaces these two Albucore dependencies with local
# builds that do not support this torch 2.4.0/CUDA 12.4 image. Use the standard
# CPython 3.10 x86_64 wheels recorded in uv.lock.
simsimd @ https://files.pythonhosted.org/packages/6d/a0/84e128cc7be66797132c1279fc359a581e54c3b86f71e7e13604e006d8de/simsimd-6.5.12-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
stringzilla @ https://files.pythonhosted.org/packages/f4/6e/678528037ceecedf990828dfb3bee130d57a4c79ad4da6cc231ddb36afb3/stringzilla-4.6.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_28_x86_64.whl

albumentations==2.0.8
click==8.3.1
matplotlib==3.10.8
opencv-python-headless==4.11.0.86
pandas==2.3.3
pillow==12.0.0
pydantic==2.12.5
pyyaml>=6.0,<7
rich==14.2.0
scikit-learn==1.7.2
tensorboard==2.19.0
tqdm==4.67.1
pytest==9.0.3
```

- [ ] **Step 3: Check the dependency file excludes forbidden packages**

Run:

```bash
if rg -n '^(tensorflow|tf2onnx|ai-edge-litert|coremltools|onnx)' requirements/train-torch.txt; then
  exit 1
fi
```

Expected: exit code 0 with no matches.

- [ ] **Step 4: Install direct dependencies and the local project**

Run:

```bash
python3 -m pip install --upgrade-strategy only-if-needed -r requirements/train-torch.txt
python3 -m pip install --no-deps -e .
```

Expected: packages install successfully; pip does not install or replace `torch`, `torchvision`, or TensorFlow.

- [ ] **Step 5: Re-check protected packages immediately**

Run:

```bash
python3 - <<'PY'
import importlib.metadata as metadata
import importlib.util

assert metadata.version("torch") == "2.4.0"
assert metadata.version("torchvision") == "0.19.0"
assert importlib.util.find_spec("tensorflow") is None
print("protected packages unchanged; TensorFlow absent")
PY
```

Expected: `protected packages unchanged; TensorFlow absent`.

- [ ] **Step 6: Commit the dependency contract**

```bash
git add requirements/train-torch.txt
git commit -m "build: add PyTorch training requirements"
```

### Task 2: Add The CBLPRD-330K PyTorch Launcher

**Files:**
- Create: `scripts/train_cblprd_torch.sh`
- Create: `test/fast_lp_ocr/cli/test_torch_launcher.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing launcher contract test**

Create `test/fast_lp_ocr/cli/test_torch_launcher.py` with:

```python
import os
import pathlib
import subprocess
import tempfile
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_ROOT / "scripts" / "train_cblprd_torch.sh"


class TorchLauncherTest(unittest.TestCase):
    def test_launcher_sets_torch_backend_and_forwards_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = pathlib.Path(tmp_dir)
            dataset_root = root / "dataset"
            for split in ("train", "val"):
                split_dir = dataset_root / split
                split_dir.mkdir(parents=True)
                (split_dir / "annotations.csv").write_text(
                    "image_path,plate_text\nimages/sample.jpg,京A12345\n",
                    encoding="utf-8",
                )

            capture_path = root / "capture.txt"
            fake_cli = root / "fake-fast-plate-ocr"
            fake_cli.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"${KERAS_BACKEND}\" \"$@\" > \"${CAPTURE_PATH}\"\n",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "CAPTURE_PATH": str(capture_path),
                    "FAST_PLATE_OCR_BIN": str(fake_cli),
                    "FAST_PLATE_OCR_DATASET_ROOT": str(dataset_root),
                }
            )
            subprocess.run(
                [str(LAUNCHER), "--batch-size", "8"],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
            )

            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[0], "torch")
            self.assertEqual(captured[1], "train")
            self.assertIn(str(dataset_root / "train" / "annotations.csv"), captured)
            self.assertIn(str(dataset_root / "val" / "annotations.csv"), captured)
            self.assertEqual(captured[-2:], ["--batch-size", "8"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest test.fast_lp_ocr.cli.test_torch_launcher -v
```

Expected: ERROR because `scripts/train_cblprd_torch.sh` does not exist.

- [ ] **Step 3: Implement the launcher**

Create `scripts/train_cblprd_torch.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${FAST_PLATE_OCR_DATASET_ROOT:-/zxk/plate_ocr/plate/CBLPRD-330K/fast-plate-ocr}"
CLI_BIN="${FAST_PLATE_OCR_BIN:-fast-plate-ocr}"
OUTPUT_DIR="${FAST_PLATE_OCR_OUTPUT_DIR:-${PROJECT_ROOT}/trained_models/cblprd_330k}"
TENSORBOARD_DIR="${FAST_PLATE_OCR_TENSORBOARD_DIR:-${PROJECT_ROOT}/tensorboard_logs/cblprd_330k}"

TRAIN_ANNOTATIONS="${DATASET_ROOT}/train/annotations.csv"
VAL_ANNOTATIONS="${DATASET_ROOT}/val/annotations.csv"

for required_path in \
    "${PROJECT_ROOT}/models/cct_s_v1.yaml" \
    "${PROJECT_ROOT}/config/cn_plate_config.yaml" \
    "${TRAIN_ANNOTATIONS}" \
    "${VAL_ANNOTATIONS}"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required file not found: ${required_path}" >&2
        exit 1
    fi
done

export KERAS_BACKEND=torch

cd "${PROJECT_ROOT}"
exec "${CLI_BIN}" train \
    --model-config-file "${PROJECT_ROOT}/models/cct_s_v1.yaml" \
    --plate-config-file "${PROJECT_ROOT}/config/cn_plate_config.yaml" \
    --annotations "${TRAIN_ANNOTATIONS}" \
    --val-annotations "${VAL_ANNOTATIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    "$@"
```

Make it executable:

```bash
chmod +x scripts/train_cblprd_torch.sh
```

- [ ] **Step 4: Ignore generated TensorBoard logs**

Append this entry under `.gitignore`'s training output section:

```gitignore
tensorboard_logs/
```

- [ ] **Step 5: Run the focused launcher tests**

Run:

```bash
python3 -m unittest test.fast_lp_ocr.cli.test_torch_launcher -v
bash -n scripts/train_cblprd_torch.sh
```

Expected: one test passes and Bash syntax validation exits 0.

- [ ] **Step 6: Commit the launcher**

```bash
git add .gitignore scripts/train_cblprd_torch.sh test/fast_lp_ocr/cli/test_torch_launcher.py
git commit -m "feat: add CBLPRD PyTorch training launcher"
```

### Task 3: Add Reusable Environment Verification

**Files:**
- Create: `scripts/verify_pytorch_training_env.py`
- Create: `test/fast_lp_ocr/cli/test_verify_pytorch_training_env.py`
- Modify: `fast_plate_ocr/__init__.py`
- Create: `test/fast_lp_ocr/test_package_init.py`

- [ ] **Step 1: Write failing annotation validation tests**

Create `test/fast_lp_ocr/cli/test_verify_pytorch_training_env.py` with:

```python
import pathlib
import tempfile
import unittest

from scripts.verify_pytorch_training_env import validate_annotations


class AnnotationValidationTest(unittest.TestCase):
    def test_accepts_valid_annotations_and_referenced_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京A12345\n",
                encoding="utf-8",
            )

            self.assertEqual(validate_annotations(csv_path, set("京A12345_"), 8), 1)

    def test_rejects_out_of_alphabet_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = pathlib.Path(tmp_dir)
            image_dir = split_dir / "images"
            image_dir.mkdir()
            (image_dir / "sample.jpg").write_bytes(b"test")
            csv_path = split_dir / "annotations.csv"
            csv_path.write_text(
                "image_path,plate_text\nimages/sample.jpg,京B12345\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside the configured alphabet"):
                validate_annotations(csv_path, set("京A12345_"), 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
KERAS_BACKEND=torch python3 -m unittest test.fast_lp_ocr.cli.test_verify_pytorch_training_env -v
```

Expected: ERROR because `scripts.verify_pytorch_training_env` does not exist.

- [ ] **Step 3: Implement annotation validation and the smoke verifier**

Create `scripts/verify_pytorch_training_env.py` with:

```python
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
    missing_images = [path for path in annotations["image_path"] if not (csv_root / path).is_file()]
    if missing_images:
        raise ValueError(f"{csv_path}: missing referenced image {missing_images[0]}")

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
    model_config = load_model_config_from_yaml(PROJECT_ROOT / "models" / "cct_s_v1.yaml")

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
```

Make it executable:

```bash
chmod +x scripts/verify_pytorch_training_env.py
```

- [ ] **Step 4: Run the focused verifier tests**

Run:

```bash
KERAS_BACKEND=torch python3 -m unittest test.fast_lp_ocr.cli.test_verify_pytorch_training_env -v
```

Expected: two tests pass.

- [ ] **Step 5: Commit the verifier**

```bash
git add scripts/verify_pytorch_training_env.py test/fast_lp_ocr/cli/test_verify_pytorch_training_env.py
git commit -m "test: add PyTorch training environment verifier"
```

### Task 4: Perform End-To-End Verification

**Files:**
- Verify: `requirements/train-torch.txt`
- Verify: `scripts/train_cblprd_torch.sh`
- Verify: `scripts/verify_pytorch_training_env.py`
- Verify: `config/cn_plate_config.yaml`

- [ ] **Step 1: Run the complete environment verifier**

Run:

```bash
KERAS_BACKEND=torch python3 scripts/verify_pytorch_training_env.py
```

Expected output includes:

```text
keras_backend=torch
torch=2.4.0
torchvision=0.19.0
device=PPU-ZW810E
train_rows=325005
val_rows=17105
model_forward=ok
```

- [ ] **Step 2: Verify the installed CLI through the launcher**

Run:

```bash
scripts/train_cblprd_torch.sh --help
```

Expected: the `fast-plate-ocr train` help exits 0 and lists options such as `--batch-size`, `--epochs`, and `--mixed-precision-policy`.

- [ ] **Step 3: Run focused and existing training tests**

Run:

```bash
KERAS_BACKEND=torch python3 -m unittest \
  test.fast_lp_ocr.cli.test_torch_launcher \
  test.fast_lp_ocr.cli.test_verify_pytorch_training_env -v
KERAS_BACKEND=torch python3 -m pytest test/fast_lp_ocr/cli/test_train.py -q
```

Expected: all focused tests and existing training CLI tests pass.

- [ ] **Step 4: Inspect package and repository state**

Run:

```bash
python3 -m pip show fast-plate-ocr keras torch torchvision
git status --short
```

Expected: editable `fast-plate-ocr` installation points at this checkout, Keras is 3.12.1, PyTorch packages retain their original versions, and only pre-existing user files remain untracked.

- [ ] **Step 5: Record the ready-to-run training command**

Use the launcher with explicit machine tuning when full training is desired:

```bash
scripts/train_cblprd_torch.sh \
  --epochs 150 \
  --batch-size 64 \
  --workers 4 \
  --tensorboard \
  --seed 42
```

Expected: this is documented for handoff but not started during environment setup.
