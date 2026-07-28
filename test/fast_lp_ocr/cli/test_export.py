"""
Tests for the export script.
"""

import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fast_plate_ocr.cli.export import export as export_cli
from fast_plate_ocr.train.model.config import (
    PlateConfig,
    load_plate_config_from_yaml,
)
from fast_plate_ocr.train.model.model_builders import build_model
from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
from test import MODEL_CONFIG_PATHS

EXCLUDE_MODELS = ("cct_s_v1", "cct_xs_v1")
"""
Models to exclude from testing. Currently v1 models fail to be exported w/ 'gelu' with some lib versions, so excluding
them here.
"""
MODEL_CONFIG_PATHS_TO_TEST = [m for m in MODEL_CONFIG_PATHS if m.stem not in EXCLUDE_MODELS]
"""Model configs to test."""


PLATE_CFG_BASE_YAML = """
max_plate_slots: 7
alphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
pad_char: '_'
img_height: 64
img_width: 128
keep_aspect_ratio: False
interpolation: linear
image_color_mode: rgb
"""
"""Plate config without region recognition"""

PLATE_CFG_WITH_REGIONS_YAML = """
max_plate_slots: 7
alphabet: '_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
pad_char: '_'
img_height: 64
img_width: 128
keep_aspect_ratio: False
interpolation: linear
image_color_mode: rgb
plate_regions:
  - Argentina
  - Uruguay
  - Brazil
  - Chile
  - Paraguay
"""
"""Plate config with region recognition"""

PLATE_CONFIG_VARIANTS = [
    pytest.param(PLATE_CFG_BASE_YAML, id="plate_config_base"),
    pytest.param(PLATE_CFG_WITH_REGIONS_YAML, id="plate_config_with_regions"),
]
"""Plate config variants to test"""


def _build_and_save_keras_model(model_cfg_path: Path, plate_cfg_path: Path, save_dir: Path) -> tuple[Path, PlateConfig]:
    model_cfg = load_model_config_from_yaml(model_cfg_path)
    plate_cfg = load_plate_config_from_yaml(plate_cfg_path)

    model = build_model(model_cfg, plate_cfg, enable_region_head=plate_cfg.has_region_recognition)
    model_save_path = save_dir / "model.keras"
    model.save(model_save_path)

    return model_save_path, plate_cfg


def _write_plate_config(tmp_dir: Path, yaml_content: str) -> Path:
    with NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="plate_config_",
        dir=tmp_dir,
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(yaml_content)
        return Path(f.name)


@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS_TO_TEST, ids=lambda p: p.stem)
@pytest.mark.parametrize("plate_config_yaml", PLATE_CONFIG_VARIANTS)
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_export_to_onnx(
    model_config_path: Path,
    plate_config_yaml: str,
    dynamic_batch: bool,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    # Write the plate config to a tmp YAML file
    plate_config_path = _write_plate_config(tmp_path, plate_config_yaml)
    # Build and save the Keras model
    model_save_path, _ = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Export with given parameters
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "onnx",
    ]
    if dynamic_batch:
        args.append("--dynamic-batch")

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".onnx")
    assert exported_path.exists(), f"Expected exported ONNX file at {exported_path}"


@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS_TO_TEST, ids=lambda p: p.stem)
@pytest.mark.parametrize("plate_config_yaml", PLATE_CONFIG_VARIANTS)
def test_export_to_onnx_nchw_float32(
    model_config_path: Path,
    plate_config_yaml: str,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    # Write the plate config to a tmp YAML file
    plate_config_path = _write_plate_config(tmp_path, plate_config_yaml)
    # Build and save the Keras model
    model_save_path, plate_config = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Export with channels first and float32 input dtype
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "onnx",
        "--onnx-input-dtype",
        "float32",
        "--onnx-data-format",
        "channels_first",
    ]
    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output

    exported_path = model_save_path.with_suffix(".onnx")
    assert exported_path.exists(), f"Expected ONNX file at {exported_path}"

    onnx_model = onnx.load(str(exported_path))
    graph_input = onnx_model.graph.input[0]

    inp_type = graph_input.type.tensor_type.elem_type
    assert inp_type == TensorProto.FLOAT, "Expected input with float32 dtype"

    dims = graph_input.type.tensor_type.shape.dim
    assert len(dims) == 4, f"Input should have 4 dims, got {len(dims)}"
    c_dim_value = dims[1].dim_value
    assert c_dim_value == plate_config.num_channels, f"Expected {plate_config.num_channels} num of channels"


@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS_TO_TEST, ids=lambda p: p.stem)
@pytest.mark.parametrize("plate_config_yaml", PLATE_CONFIG_VARIANTS)
def test_export_to_tflite(
    model_config_path: Path,
    plate_config_yaml: str,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    # Write the plate config to a tmp YAML file
    plate_config_path = _write_plate_config(tmp_path, plate_config_yaml)
    # Build and save the Keras model
    model_save_path, _ = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Construct CLI arguments for TFLite
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "tflite",
    ]

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".tflite")
    assert exported_path.exists(), f"Expected exported TFLite file at {exported_path}"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("model_config_path", MODEL_CONFIG_PATHS_TO_TEST, ids=lambda p: p.stem)
@pytest.mark.parametrize("plate_config_yaml", PLATE_CONFIG_VARIANTS)
def test_export_to_coreml(
    model_config_path: Path,
    plate_config_yaml: str,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    # Write the plate config to a tmp YAML file
    plate_config_path = _write_plate_config(tmp_path, plate_config_yaml)
    # Build and save the Keras model
    model_save_path, _ = _build_and_save_keras_model(model_config_path, plate_config_path, tmp_path)
    # Construct CLI arguments for CoreML
    args = [
        "-m",
        str(model_save_path),
        "--plate-config-file",
        str(plate_config_path),
        "--format",
        "coreml",
    ]

    result = runner.invoke(export_cli, args)
    assert result.exit_code == 0, result.output
    exported_path = model_save_path.with_suffix(".mlpackage")
    assert exported_path.exists(), f"Expected exported CoreML file at {exported_path}"
