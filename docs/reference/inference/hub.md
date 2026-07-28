# Model Hub

The model hub helpers download packaged ONNX models and matching config files so they can be used directly by
`LicensePlateRecognizer`.

```python
from fast_plate_ocr.inference.hub import download_model

model_path, config_path = download_model("cct-s-v2-global-model")
```

::: fast_plate_ocr.inference.hub
