'''
Author: zxk zxk.zdg666888@gmail.com
Date: 2026-07-28 11:11:31
LastEditors: zxk zxk.zdg666888@gmail.com
LastEditTime: 2026-07-28 11:11:53
'''
"""
fast-plate-ocr package.
"""

from fast_plate_ocr.core.types import PlatePrediction
from fast_plate_ocr.inference.plate_recognizer import LicensePlateRecognizer

__all__ = ["LicensePlateRecognizer", "PlatePrediction"]
