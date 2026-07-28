# LicensePlateRecognizer

`LicensePlateRecognizer` is the main ONNX Runtime inference entry point. It accepts file paths, NumPy arrays,
lists of images, and pre-batched arrays, and returns a list of `PlatePrediction` objects.

When passing NumPy arrays, they should already match the model configuration: `uint8`, `channels_last`, and the
expected color mode (`grayscale` or RGB). For RGB models, arrays are assumed to be RGB, not OpenCV-style BGR.

```python
from fast_plate_ocr import LicensePlateRecognizer

recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
predictions = recognizer.run("test_plate.png")
```

::: fast_plate_ocr.inference.plate_recognizer.LicensePlateRecognizer
