"""Backend-independent OCR evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


PERCENT_METRICS = {
    "plate_accuracy": "车牌准确率",
    "character_accuracy": "字符准确率",
    "plate_length_accuracy": "车牌长度准确率",
    "top3_character_accuracy": "字符 Top-3 准确率",
    "character_error_rate": "字符错误率",
}
RUNTIME_METRICS = {
    "inference_ms_per_image": ("平均单张推理耗时", "毫秒"),
    "inference_latency_ms_p50": ("单张推理耗时 P50", "毫秒"),
    "inference_latency_ms_p95": ("单张推理耗时 P95", "毫秒"),
    "inference_throughput_images_per_second": ("推理吞吐量", "张/秒"),
    "end_to_end_throughput_images_per_second": ("端到端吞吐量", "张/秒"),
}


def _format_metrics(metrics: dict[str, Any], indent: str = "  ") -> list[str]:
    lines: list[str] = []
    if "samples" in metrics:
        lines.append(f"{indent}样本数：{metrics['samples']}")
    if "correct_plates" in metrics:
        lines.append(f"{indent}完全正确车牌数：{metrics['correct_plates']}")
    for key, label in PERCENT_METRICS.items():
        if key in metrics:
            lines.append(f"{indent}{label}：{float(metrics[key]) * 100:.2f}%")
    for key, (label, unit) in RUNTIME_METRICS.items():
        if key in metrics:
            lines.append(f"{indent}{label}：{float(metrics[key]):.2f} {unit}")
    return lines


def format_report_chinese(report: dict[str, Any]) -> str:
    """Format the machine-readable evaluation report for terminal users."""
    lines = ["评估完成"]
    model_type = {"keras": "Keras（PyTorch）", "onnx": "ONNX"}.get(
        str(report.get("model_type")), str(report.get("model_type", "未知"))
    )
    for label, key, value in (
        ("模型", "model", report.get("model")),
        ("模型类型", "model_type", model_type),
        ("运行设备", "runtime_device", report.get("runtime_device")),
        ("车牌配置", "plate_config", report.get("plate_config")),
    ):
        if key in report:
            lines.append(f"{label}：{value}")
    datasets = report.get("datasets", {})
    if datasets:
        lines.append("数据集：")
        lines.extend(f"  {name}：{path}" for name, path in datasets.items())
    settings = report.get("settings", {})
    if settings:
        lines.append(
            "评估参数："
            f"批次大小={settings.get('batch_size')}，"
            f"加载线程={settings.get('workers')}，GPU={settings.get('gpu')}"
        )
    lines.append("总体指标：")
    lines.extend(_format_metrics(report.get("overall", {})))
    groups = report.get("groups", {})
    if groups:
        lines.append("分组指标：")
        for name, metrics in groups.items():
            lines.append(f"  [{name}]")
            lines.extend(_format_metrics(metrics, indent="    "))
    return "\n".join(lines)


def edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance using one row of working memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


class MetricAccumulator:
    """Accumulate identical quality and runtime metrics for any model backend."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.samples = 0
        self.correct_plates = 0
        self.correct_characters = 0
        self.correct_lengths = 0
        self.top3_characters = 0
        self.edit_errors = 0
        self.truth_characters = 0
        self.inference_seconds = 0.0
        self.latencies_ms: list[float] = []

    def _scores(self, output: np.ndarray, batch_size: int) -> np.ndarray:
        expected = batch_size * self.config.max_plate_slots * len(self.config.alphabet)
        values = np.asarray(output)
        if values.size != expected:
            raise ValueError(
                f"Unexpected model output shape {values.shape}; expected {expected} values."
            )
        return values.reshape(batch_size, self.config.max_plate_slots, len(self.config.alphabet))

    def _truth_indices(self, truths: list[str]) -> np.ndarray:
        lookup = {character: index for index, character in enumerate(self.config.alphabet)}
        padded = [truth.ljust(self.config.max_plate_slots, self.config.pad_char) for truth in truths]
        return np.asarray([[lookup[character] for character in text] for text in padded])

    def update(
        self,
        output: np.ndarray,
        truths: list[str],
        inference_seconds: float,
    ) -> None:
        scores = self._scores(output, len(truths))
        predicted_indices = np.argmax(scores, axis=-1)
        truth_indices = self._truth_indices(truths)
        top_k = min(3, scores.shape[-1])
        top_indices = np.argpartition(scores, -top_k, axis=-1)[..., -top_k:]
        alphabet = np.asarray(list(self.config.alphabet))
        predicted_texts = [
            "".join(row).rstrip(self.config.pad_char) for row in alphabet[predicted_indices]
        ]
        self.samples += len(truths)
        self.correct_plates += sum(predicted == truth for predicted, truth in zip(predicted_texts, truths))
        self.correct_characters += int(np.sum(predicted_indices == truth_indices))
        self.correct_lengths += sum(len(predicted) == len(truth) for predicted, truth in zip(predicted_texts, truths))
        self.top3_characters += int(np.sum(np.any(top_indices == truth_indices[..., None], axis=-1)))
        self.edit_errors += sum(edit_distance(predicted, truth) for predicted, truth in zip(predicted_texts, truths))
        self.truth_characters += sum(len(truth) for truth in truths)
        self.inference_seconds += inference_seconds
        self.latencies_ms.append(inference_seconds * 1000 / len(truths))

    def report(self, end_to_end_seconds: float) -> dict[str, int | float]:
        if self.samples == 0:
            raise ValueError("Cannot report metrics for an empty dataset.")
        character_slots = self.samples * self.config.max_plate_slots
        latencies = np.asarray(self.latencies_ms)
        return {
            "samples": self.samples,
            "correct_plates": self.correct_plates,
            "plate_accuracy": self.correct_plates / self.samples,
            "character_accuracy": self.correct_characters / character_slots,
            "plate_length_accuracy": self.correct_lengths / self.samples,
            "top3_character_accuracy": self.top3_characters / character_slots,
            "character_error_rate": self.edit_errors / max(1, self.truth_characters),
            "inference_ms_per_image": self.inference_seconds * 1000 / self.samples,
            "inference_latency_ms_p50": float(np.percentile(latencies, 50)),
            "inference_latency_ms_p95": float(np.percentile(latencies, 95)),
            "inference_throughput_images_per_second": self.samples / self.inference_seconds,
            "end_to_end_throughput_images_per_second": self.samples / end_to_end_seconds,
        }
