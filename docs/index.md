# Fast & Lightweight License Plate OCR

![Intro](https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/readme_demo.gif)

`fast-plate-ocr` is a **lightweight** and **fast** OCR framework for **license plate text recognition**. You can train
models from scratch or use the trained models for inference.

The idea is to use this after a plate object detector, since the OCR expects the cropped plates.

!!! info "🚀 Try it on Hugging Face Spaces!"
    You can try `fast-plate-ocr` pre-trained models in [**Hugging Spaces**](https://huggingface.co/spaces/ankandrew/fast-alpr).
    No setup required!


### Features

- **Keras 3 Backend Support**: Train seamlessly using **[TensorFlow](https://www.tensorflow.org/)**, **[JAX](https://github.com/google/jax)**, or **[PyTorch](https://pytorch.org/)** backends 🧠
- **Efficient Execution**: **Lightweight** models that are cheap to run 💰
- **ONNX Runtime Inference**: **Fast** and **optimized** inference with **[ONNX runtime](https://onnxruntime.ai/)** ⚡
- **User-Friendly CLI**: Simplified **CLI** for **training** and **validating** OCR models 🛠️
- **Region Recognition (Optional)**: Predict **region/country** of the license plate 🌍
- **Model HUB**: Access to a collection of **pre-trained models** ready for inference 🌟
- **Train**/**Fine-tune**: Easily train or **fine-tune** your own models 🔧
- **Export-Friendly**: Export easily to **CoreML**, **TFLite**, or **ONNX** formats 📦

### Quick Installation

Install for **inference**:

```shell
pip install fast-plate-ocr[onnx-gpu]
```

Install for **training**:

```shell
pip install fast-plate-ocr[train]
```

For full installation options (like GPU backends or ONNX variants), see the [**Installation Guide**](installation.md).

### Quick Usage

Run **OCR** on a **cropped** license plate image using [`LicensePlateRecognizer`](reference/inference/inference_class.md):

```python
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer("cct-s-v2-global-model")
print(m.run("test_plate.png"))
```

For **more examples** and input formats (NumPy arrays, batches, etc.), see the [**Inference Guide**](inference/running_inference.md).

If your model includes a **region head** and `plate_regions` is defined, predictions include `region`.
The `region_prob` field is populated when `return_confidence=True`.

### Use it with FastALPR

If you prefer not to use `fast-plate-ocr` directly on **cropped plates**, you can easily leverage it through **FastALPR**,
an end-to-end Automatic License Plate Recognition library where `fast-plate-ocr` serves as the **default OCR backend**.


```python
from fast_alpr import ALPR  # (1)!

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",  # (2)!
)

alpr_results = alpr.predict("assets/test_image.png")
print(alpr_results)
```

1. **Requires** `fast-alpr` package to be **installed**!
2. Can be **any** of the default `fast-plate-ocr` **trained** models or **custom** ones too!

!!! tip "Explore More"
    Check out the [**FastALPR**](https://github.com/ankandrew/fast-alpr) docs for full ALPR pipeline and integration tips!
