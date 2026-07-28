"""
Tests for inference process module.
"""

import numpy as np
import numpy.typing as npt
import pytest

from fast_plate_ocr.core.process import postprocess_output


@pytest.mark.parametrize(
    "model_output, max_plate_slots, model_alphabet, expected_plates",
    [
        (
            np.array(
                [
                    [[0.5, 0.4, 0.1], [0.2, 0.6, 0.2], [0.1, 0.4, 0.5]],
                    [[0.1, 0.1, 0.8], [0.2, 0.2, 0.6], [0.1, 0.4, 0.5]],
                ],
                dtype=np.float32,
            ),
            3,
            "ABC",
            ["ABC", "CCC"],
        ),
        (
            np.array(
                [[[0.1, 0.4, 0.5], [0.6, 0.2, 0.2], [0.1, 0.5, 0.4]]],
                dtype=np.float32,
            ),
            3,
            "ABC",
            ["CAB"],
        ),
    ],
)
def test_postprocess_output(
    model_output: npt.NDArray,
    max_plate_slots: int,
    model_alphabet: str,
    expected_plates: list[str],
) -> None:
    plate_prediction = postprocess_output(model_output, max_plate_slots, model_alphabet)
    actual_plates = [x.plate for x in plate_prediction]
    assert actual_plates == expected_plates


@pytest.mark.parametrize(
    "model_output, region_output, region_labels, expected_plate, expected_region",
    [
        (
            np.array([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]], dtype=np.float32),
            np.array([[0.1, 0.2, 0.7]], dtype=np.float32),
            ["AR", "BR", "CL"],
            "AB",
            "CL",
        ),
    ],
)
def test_postprocess_output_with_region(
    model_output, region_output, region_labels, expected_plate, expected_region
) -> None:
    preds = postprocess_output(
        model_output,
        max_plate_slots=2,
        model_alphabet="ABC",
        return_region=True,
        region_output=region_output,
        region_labels=region_labels,
    )
    assert preds[0].plate == expected_plate
    assert preds[0].region == expected_region


@pytest.mark.parametrize(
    "model_output, region_output, region_labels, expected_plate, expected_region",
    [
        (
            np.array([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]], dtype=np.float32),
            np.array([[0.6, 0.3, 0.1]], dtype=np.float32),
            ["AR", "BR", "CL"],
            "AB",
            "AR",
        ),
    ],
)
def test_postprocess_output_with_region_and_confidence(
    model_output, region_output, region_labels, expected_plate, expected_region
) -> None:
    preds = postprocess_output(
        model_output,
        max_plate_slots=2,
        model_alphabet="ABC",
        return_confidence=True,
        return_region=True,
        region_output=region_output,
        region_labels=region_labels,
    )
    assert preds[0].plate == expected_plate
    assert preds[0].region == expected_region
    assert preds[0].char_probs is not None


@pytest.mark.parametrize(
    "model_output, max_plate_slots, model_alphabet",
    [
        (
            np.array([[[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]], dtype=np.float32),
            2,
            "ABC",
        ),
    ],
)
def test_postprocess_output_missing_region_logits_raises(model_output, max_plate_slots, model_alphabet) -> None:
    with pytest.raises(ValueError):
        postprocess_output(
            model_output,
            max_plate_slots=max_plate_slots,
            model_alphabet=model_alphabet,
            return_region=True,
        )
