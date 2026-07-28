"""
Tests for ONNX hub module.
"""

from http import HTTPStatus

import pytest
import requests

from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS


def _check_url(url: str) -> tuple[str, int | str]:
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return url, response.status_code
    except requests.RequestException as e:
        return url, str(e)


@pytest.mark.parametrize(
    ("model_name", "artifact_kind", "url"),
    [
        pytest.param(model_name, artifact_kind, url, id=f"{model_name}:{artifact_kind}")
        for model_name, (model_url, config_url) in AVAILABLE_ONNX_MODELS.items()
        for artifact_kind, url in (("model", model_url), ("config", config_url))
    ],
)
def test_model_and_config_urls(model_name: str, artifact_kind: str, url: str) -> None:
    _, result = _check_url(url)
    assert result == HTTPStatus.OK, f"{model_name} {artifact_kind} URL {url} is not accessible, got {result}"
