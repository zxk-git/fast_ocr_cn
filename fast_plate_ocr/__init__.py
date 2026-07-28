"""
fast-plate-ocr package.
"""

from typing import TYPE_CHECKING

from fast_plate_ocr.core.types import PlatePrediction

if TYPE_CHECKING:
    from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer

__all__ = ["LicensePlateRecognizer", "PlatePrediction"]


def __getattr__(name: str) -> object:
    if name == "LicensePlateRecognizer":
        from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer

        return LicensePlateRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
