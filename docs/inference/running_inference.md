### Inference Guide

The `fast-plate-ocr` library performs **high-performance** license plate recognition using **ONNX Runtime** for **inference**.

To run inference use the [`LicensePlateRecognizer`](../reference/inference/inference_class.md) class, which supports a wide
range of input types:

- File paths (str or Path)
- NumPy arrays representing single images (grayscale or RGB)
- Lists of paths or NumPy arrays
- Pre-batched NumPy arrays (4D shape: (N, H, W, C))

The model automatically handles resizing, padding, and format conversion according to its configuration. Predictions
can optionally include character-level confidence scores.

### NumPy array requirements

When passing in-memory NumPy arrays instead of image paths, make sure the arrays already match the model input
convention:

- Use `uint8` inputs. Arrays are cast to `uint8` before inference; floating-point arrays are not normalized.
- Use `channels_last` layout: `(H, W, C)` for a single image or `(N, H, W, C)` for a batch.
- For grayscale models, pass grayscale arrays with shape `(H, W)` or `(H, W, 1)`.
- For RGB models, pass RGB arrays with shape `(H, W, 3)`.
- If you loaded images with OpenCV (`cv2.imread`), convert BGR to RGB before passing them to an RGB model.

If you pass image paths instead, `fast-plate-ocr` handles disk loading and the required grayscale/RGB conversion
for you.



### Predict a single image

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
print(plate_recognizer.run("test_plate.png"))
```

### Use your own exported ONNX model

If you exported your own model, load it with both the ONNX file and the matching plate config:

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer(
    onnx_model_path="path/to/trained_model/best.onnx",
    plate_config_path="path/to/trained_model/plate_config.yaml",
)
print(plate_recognizer.run("test_plate.png"))
```

Important:

- Use the `plate_config.yaml` from the same trained model that produced the ONNX file.
- To use the exported model with `LicensePlateRecognizer`, keep the default ONNX export settings:
  `channels_last` input layout and `uint8` input dtype.

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/inference_demo.gif?raw=true" alt="Inference Demo"/>
</div>

</details>

### Predict a batch in memory

```python
import cv2
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
imgs = [
    cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    for p in ["plate1.jpg", "plate2.jpg"]
]
res = plate_recognizer.run(imgs)
```

### Return confidence scores

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
pred = plate_recognizer.run("test_plate.png", return_confidence=True)[0]
print(pred.plate, pred.char_probs)
```

### Region prediction (optional)

If the loaded model exports a **region** head and the plate config includes `plate_regions`, each prediction
includes a `region` label. The `region_prob` field is also populated when `return_confidence=True`:

```python
from fast_plate_ocr import LicensePlateRecognizer

plate_recognizer = LicensePlateRecognizer("cct-s-v2-global-model")
pred = plate_recognizer.run("test_plate.png", return_confidence=True)[0]
print(pred.plate, pred.region, pred.region_prob)
```

### Benchmark the model

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer("cct-s-v2-global-model")
m.benchmark()
```

<details>
  <summary>Demo</summary>

<div style="margin-top: 10px;">
<img src="https://github.com/ankandrew/fast-plate-ocr/blob/ac3d110c58f62b79072e3a7af15720bb52a45e4e/extra/benchmark_demo.gif?raw=true" alt="Benchmark Demo"/>
</div>

</details>

For a full list of options see [Reference](../reference/inference/inference_class.md).
