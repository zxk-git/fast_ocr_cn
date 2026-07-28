import pathlib

import numpy as np
import pytest

from scripts import compare_model_outputs as compare


FAKE_BATCH = np.ones((1, 2, 3, 1), dtype=np.uint8)
FAKE_CONFIG = object()


def fake_load_batch(_image: pathlib.Path, _config: pathlib.Path) -> tuple[np.ndarray, object]:
    return FAKE_BATCH, FAKE_CONFIG


def fake_run_keras(_batch: np.ndarray, _model: pathlib.Path, _backend: str) -> np.ndarray:
    return np.array([[1.0, 2.0]], dtype=np.float32)


def fake_run_onnx(_batch: np.ndarray, _model: pathlib.Path, _device: str) -> np.ndarray:
    return np.array([[2.0, 4.0]], dtype=np.float32)


def fake_decode(output: np.ndarray, _config: object) -> str:
    return "KERAS" if output[0, 0] == 1.0 else "ONNX"


def test_compares_same_shape_outputs() -> None:
    keras_output = np.array([[1.0, 2.0]], dtype=np.float32)
    onnx_output = np.array([[2.0, 4.0]], dtype=np.float32)

    report = compare.compare_outputs(keras_output, onnx_output, atol=1e-6, rtol=1e-6)

    assert report == {
        "keras_shape": "1x2",
        "onnx_shape": "1x2",
        "comparable": "true",
        "max_abs_diff": "2",
        "mean_abs_diff": "1.5",
        "rmse": "1.58113883",
        "cosine_similarity": "1",
        "allclose": "false",
    }


def test_reports_mismatched_shapes_without_numeric_metrics() -> None:
    keras_output = np.ones((1, 2), dtype=np.float32)
    onnx_output = np.ones((1, 3), dtype=np.float32)

    report = compare.compare_outputs(keras_output, onnx_output, atol=1e-5, rtol=1e-5)

    assert report == {
        "keras_shape": "1x2",
        "onnx_shape": "1x3",
        "comparable": "false",
    }


def test_rejects_empty_outputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compare.compare_outputs(np.array([]), np.array([]), atol=1e-5, rtol=1e-5)


def test_main_reports_difference_without_failing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "plate.png"
    keras_path = tmp_path / "best.keras"
    onnx_path = tmp_path / "best.onnx"
    config_path = tmp_path / "plate_config.yaml"
    for path in (image_path, keras_path, onnx_path, config_path):
        path.touch()

    monkeypatch.setattr(compare.inference, "load_batch", fake_load_batch)
    monkeypatch.setattr(compare.inference, "run_keras", fake_run_keras)
    monkeypatch.setattr(compare.inference, "run_onnx", fake_run_onnx)
    monkeypatch.setattr(compare.inference, "decode_plate", fake_decode)

    return_code = compare.main(
        [
            "--image",
            str(image_path),
            "--keras-model",
            str(keras_path),
            "--onnx-model",
            str(onnx_path),
            "--plate-config",
            str(config_path),
        ]
    )

    assert return_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "keras_plate=KERAS",
        "onnx_plate=ONNX",
        "keras_shape=1x2",
        "onnx_shape=1x2",
        "comparable=true",
        "max_abs_diff=2",
        "mean_abs_diff=1.5",
        "rmse=1.58113883",
        "cosine_similarity=1",
        "allclose=false",
    ]


def test_main_reports_missing_model(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    image_path = tmp_path / "plate.png"
    config_path = tmp_path / "plate_config.yaml"
    image_path.touch()
    config_path.touch()

    return_code = compare.main(
        [
            "--image",
            str(image_path),
            "--keras-model",
            str(tmp_path / "missing.keras"),
            "--onnx-model",
            str(tmp_path / "missing.onnx"),
            "--plate-config",
            str(config_path),
        ]
    )

    assert return_code == 1
    assert "Keras model not found" in capsys.readouterr().err
