import csv
import json
import pathlib
import types

import numpy as np
import pytest

from scripts import evaluate_model as evaluation


def _config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        max_plate_slots=3,
        alphabet="AB_",
        pad_char="_",
        img_height=2,
        img_width=3,
        image_color_mode="rgb",
        keep_aspect_ratio=True,
        interpolation="linear",
        padding_color=114,
    )


def _scores(indices: list[list[int]], vocabulary_size: int = 3) -> np.ndarray:
    output = np.zeros((len(indices), len(indices[0]), vocabulary_size), dtype=np.float32)
    for row, values in enumerate(indices):
        for column, value in enumerate(values):
            output[row, column, value] = 1.0
    return output


def _write_annotations(directory: pathlib.Path, rows: list[tuple[str, str]]) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "annotations.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "plate_text"])
        writer.writerows(rows)
    return path


def test_resolves_csv_dataset_directory_and_grouped_root(tmp_path: pathlib.Path) -> None:
    direct = _write_annotations(tmp_path / "direct", [])
    grouped = tmp_path / "grouped"
    base = _write_annotations(grouped / "ccpd_base", [])
    blur = _write_annotations(grouped / "ccpd_blur", [])

    assert evaluation.resolve_datasets(direct) == {"direct": direct.resolve()}
    assert evaluation.resolve_datasets(direct.parent) == {"direct": direct.resolve()}
    assert evaluation.resolve_datasets(grouped) == {
        "ccpd_base": base.resolve(),
        "ccpd_blur": blur.resolve(),
    }


def test_rejects_dataset_without_annotations(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotations.csv"):
        evaluation.resolve_datasets(tmp_path)


def test_loads_and_validates_annotation_rows(tmp_path: pathlib.Path) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.touch()
    annotations = _write_annotations(tmp_path, [("images/one.jpg", "AB")])

    records = evaluation.load_records(annotations, "validation", _config())

    assert records == [evaluation.Record(image.resolve(), "AB", "validation")]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["missing.jpg", "AB"], "image not found"),
        (["images/one.jpg", "AC"], "outside plate alphabet"),
        (["images/one.jpg", "ABAB"], "max_plate_slots"),
    ],
)
def test_rejects_invalid_annotations(
    tmp_path: pathlib.Path, rows: list[str], message: str
) -> None:
    image = tmp_path / "images" / "one.jpg"
    image.parent.mkdir()
    image.touch()
    annotations = _write_annotations(tmp_path, [(rows[0], rows[1])])

    with pytest.raises(ValueError, match=message):
        evaluation.load_records(annotations, "validation", _config())


def test_calculates_unified_quality_and_runtime_metrics() -> None:
    accumulator = evaluation.MetricAccumulator(_config())
    predictions = _scores([[0, 1, 2], [0, 0, 2]])

    accumulator.update(predictions, ["AB", "A"], inference_seconds=0.02)
    report = accumulator.report(end_to_end_seconds=0.04)

    assert report == {
        "samples": 2,
        "correct_plates": 1,
        "plate_accuracy": 0.5,
        "character_accuracy": pytest.approx(5 / 6),
        "plate_length_accuracy": 0.5,
        "top3_character_accuracy": 1.0,
        "character_error_rate": pytest.approx(1 / 3),
        "inference_ms_per_image": 10.0,
        "inference_latency_ms_p50": 10.0,
        "inference_latency_ms_p95": 10.0,
        "inference_throughput_images_per_second": 100.0,
        "end_to_end_throughput_images_per_second": 50.0,
    }


def test_top3_metric_can_miss_true_character() -> None:
    config = types.SimpleNamespace(max_plate_slots=1, alphabet="ABC_", pad_char="_")
    accumulator = evaluation.MetricAccumulator(config)
    prediction = np.array([[[0.4, 0.3, 0.2, 0.1]]], dtype=np.float32)

    accumulator.update(prediction, ["_"], inference_seconds=0.01)

    assert accumulator.report(0.02)["top3_character_accuracy"] == 0.0


def test_rejects_model_output_with_wrong_shape() -> None:
    accumulator = evaluation.MetricAccumulator(_config())

    with pytest.raises(ValueError, match="model output"):
        accumulator.update(np.ones((2, 4), dtype=np.float32), ["AB", "A"], 0.01)


def test_selects_onnx_provider_and_gpu_id() -> None:
    available = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    assert evaluation.onnx_providers("auto", 2, available) == [
        ("CUDAExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]
    assert evaluation.onnx_providers("cpu", 0, available) == ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        evaluation.onnx_providers("cuda", 0, ["CPUExecutionProvider"])


def test_dispatches_keras_and_onnx_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, int]] = []
    keras_runtime = object()
    onnx_runtime = object()
    monkeypatch.setattr(
        evaluation,
        "load_keras_runtime",
        lambda _path, device, gpu: calls.append(("keras", device, gpu)) or keras_runtime,
    )
    monkeypatch.setattr(
        evaluation,
        "load_onnx_runtime",
        lambda _path, device, gpu: calls.append(("onnx", device, gpu)) or onnx_runtime,
    )

    assert evaluation.load_runtime(pathlib.Path("best.keras"), "cuda", 1) is keras_runtime
    assert evaluation.load_runtime(pathlib.Path("best.onnx"), "cpu", 0) is onnx_runtime
    assert calls == [("keras", "cuda", 1), ("onnx", "cpu", 0)]


def test_keras_cpu_runtime_does_not_probe_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    class UnexpectedCuda:
        @staticmethod
        def is_available() -> bool:
            raise AssertionError("CPU evaluation must not initialize CUDA")

    class FakeModel:
        def __call__(self, batch: np.ndarray, training: bool) -> np.ndarray:
            return batch

    modules = {
        "torch": types.SimpleNamespace(cuda=UnexpectedCuda()),
        "keras": types.SimpleNamespace(
            models=types.SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeModel())
        ),
        "fast_plate_ocr.train.model.layers": object(),
    }
    monkeypatch.setattr(evaluation.importlib, "import_module", modules.__getitem__)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "previous")

    runtime = evaluation.load_keras_runtime(tmp_path / "best.keras", "cpu", 0)

    assert runtime.device == "cpu"
    assert runtime.model_type == "keras"


def test_evaluate_groups_updates_image_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    updates: list[int] = []
    progress_options: dict[str, object] = {}
    closed = False

    class FakeProgress:
        def update(self, count: int) -> None:
            updates.append(count)

        def close(self) -> None:
            nonlocal closed
            closed = True

    def fake_progress(**kwargs: object) -> FakeProgress:
        progress_options.update(kwargs)
        return FakeProgress()

    records = [evaluation.Record(tmp_path / f"{index}.jpg", "AB", "validation") for index in range(3)]
    runtime = evaluation.Runtime(
        "onnx", "cpu", lambda batch: _scores([[0, 1, 2]] * len(batch)), lambda: None
    )
    monkeypatch.setattr(evaluation, "tqdm", fake_progress)
    monkeypatch.setattr(
        evaluation,
        "_load_batch",
        lambda selected, _config, _pool: np.zeros((len(selected), 2, 3, 3), dtype=np.uint8),
    )

    overall, groups = evaluation.evaluate_groups(
        {"validation": records}, _config(), runtime,
        batch_size=2, workers=1, show_progress=True,
    )

    assert overall["samples"] == 3
    assert groups["validation"]["samples"] == 3
    assert progress_options == {
        "total": 3, "desc": "validation", "unit": "img",
        "disable": False, "dynamic_ncols": True,
    }
    assert updates == [2, 1]
    assert closed is True


def test_progress_can_be_disabled_from_cli() -> None:
    assert evaluation.build_parser().parse_args([]).no_progress is False
    assert evaluation.build_parser().parse_args(["--no-progress"]).no_progress is True


def test_main_prints_chinese_summary_and_writes_json_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = tmp_path / "best.onnx"
    config = tmp_path / "plate_config.yaml"
    dataset = tmp_path / "dataset"
    output = tmp_path / "metrics.json"
    model.touch()
    config.touch()
    _write_annotations(dataset, [])
    report = {
        "model_type": "onnx",
        "runtime_device": "CPUExecutionProvider",
        "overall": {
            "samples": 10,
            "plate_accuracy": 0.9,
            "character_accuracy": 0.95,
            "plate_length_accuracy": 1.0,
        },
        "groups": {},
    }
    monkeypatch.setattr(evaluation, "run_evaluation", lambda **_kwargs: report)

    code = evaluation.main(
        [
            "--model", str(model), "--plate-config", str(config),
            "--dataset", str(dataset), "--output", str(output),
        ]
    )

    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    console = capsys.readouterr().out
    assert "评估完成" in console
    assert "模型类型：ONNX" in console
    assert "运行设备：CPUExecutionProvider" in console
    assert "总体指标" in console
    assert "样本数：10" in console
    assert "车牌准确率：90.00%" in console
    assert "字符准确率：95.00%" in console
    assert "车牌长度准确率：100.00%" in console
