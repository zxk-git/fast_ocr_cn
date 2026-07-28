"""fast-plate-ocr package."""

# Author: zxk zxk.zdg666888@gmail.com
# Date: 2026-07-28 11:11:31
# LastEditors: zxk zxk.zdg666888@gmail.com
# LastEditTime: 2026-07-28 11:11:53

from typing import TYPE_CHECKING

from fast_plate_ocr.core.types import PlatePrediction

if TYPE_CHECKING:
    from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer

__all__ = ["LicensePlateRecognizer", "PlatePrediction"]


def __getattr__(name: str) -> object:
    if name == "LicensePlateRecognizer":
        from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer  # noqa: PLC0415

        return LicensePlateRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
