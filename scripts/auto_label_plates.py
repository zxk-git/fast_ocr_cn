#!/usr/bin/env python3
"""Auto-label raw car/scene images with plate bounding box (4 corners) + plate text.

Pipeline per image:
    1. Detect plate via the YOLOv5-face adapter -> 4 corner landmarks + single/double class
    2. Perspective-rectify the 4 corners -> cropped plate patch
    3. Recognize plate text with the fast-plate-ocr model (ONNX or Keras)
    4. Append a row to the output CSV

Output CSV columns:
    image_path, det_index, plate_text, x1..y4, det_conf, plate_class

Visualization modes:
    --visualize          draw bbox+text on original images
    --feature-maps-dir   save preprocessed input and transformer heatmap per plate
                         (requires a .keras model, not available with ONNX)

Usage:
    python scripts/auto_label_plates.py \\
        --image-dir ../test_data/img \\
        --detector-repo-path ../license_plate_recognition/Chinese_license_plate_detection_recognition \\
        --detector-weights ../license_plate_recognition/Chinese_license_plate_detection_recognition/weights/plate_detect.pt \\
        --model trained_models/cblprd_ccpd_cct_s_v2_torch/2026-07-29_13-47-11/best.keras \\
        --plate-config config/cn_plate_config.yaml \\
        --output ../test_data/annotations.csv \\
        --visualize ../test_data/viz \\
        --feature-maps-dir ../test_data/feat_maps
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

# Make the project root importable regardless of how this script is launched.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also make scripts/ importable for `import test_single_image`.
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import test_single_image as inference  # type: ignore[import-untyped]

from data_convert.plate_detector_adapter import (  # noqa: E402
    Yolov5FaceLandmarkDetector,
    four_point_transform,
)
from fast_plate_ocr.core.process import preprocess_image, resize_image  # noqa: E402


LOGGER = logging.getLogger("auto_label_plates")
Device = Literal["auto", "cpu", "cuda"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CSV_FIELDS = [
    "image_path", "det_index", "plate_text",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    "det_conf", "plate_class",
]

# Bounding-box visualization defaults.
BBOX_COLORS = [
    (0, 255, 0),       # green
    (255, 0, 0),       # blue
    (0, 0, 255),       # red
    (255, 255, 0),     # cyan
    (255, 0, 255),     # magenta
    (0, 255, 255),     # yellow
]
BBOX_THICKNESS = 3
TEXT_BG_COLOR = (0, 0, 0)
TEXT_SCALE = 1.5
TEXT_THICKNESS = 3


@dataclass
class Recognizer:
    """Wraps a loaded fast-plate-ocr model and its plate config for repeated inference."""

    predict: Any          # callable: (batch_uint8_NHWC) -> raw_output_ndarray
    config: Any           # PlateConfig
    model_type: str       # "onnx" | "keras"
    keras_model: Any = None  # Keras model instance (only set for .keras models)

    @classmethod
    def from_path(cls, model_path: pathlib.Path, plate_config: pathlib.Path, device: Device) -> "Recognizer":
        selected = inference.model_type(model_path)
        config_module = importlib.import_module("fast_plate_ocr.inference.config")
        config = config_module.PlateConfig.from_yaml(plate_config)
        if selected == "onnx":
            ort = importlib.import_module("onnxruntime")
            providers = inference.select_onnx_providers(device, ort.get_available_providers())
            session = ort.InferenceSession(str(model_path), providers=providers)
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if not inputs or not outputs:
                raise ValueError("ONNX model must expose at least one input and output.")
            model_input = inputs[0]
            output_names = [o.name for o in outputs]
            fetch_names = ["plate"] if "plate" in output_names else [output_names[0]]

            def predict(batch: np.ndarray) -> np.ndarray:
                values = inference.cast_onnx_input(batch, model_input.type)
                return inference.plate_output(session.run(fetch_names, {model_input.name: values}))

            return cls(predict=predict, config=config, model_type=selected, keras_model=None)
        else:
            os.environ["KERAS_BACKEND"] = "torch"
            importlib.import_module("fast_plate_ocr.train.model.layers")
            keras = importlib.import_module("keras")
            model = keras.models.load_model(model_path, compile=False)

            def predict(batch: np.ndarray) -> np.ndarray:
                return inference.plate_output(model(batch, training=False))

            return cls(predict=predict, config=config, model_type=selected, keras_model=model)

    def recognize(self, crop_bgr: np.ndarray) -> str:
        """Recognize plate text from a BGR crop array."""
        _, text = self.recognize_with_input(crop_bgr)
        return text

    def recognize_with_input(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, str]:
        """Recognize plate text and also return the preprocessed model input batch (1, H, W, 3)."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = resize_image(
            rgb,
            img_height=self.config.img_height,
            img_width=self.config.img_width,
            image_color_mode=self.config.image_color_mode,
            keep_aspect_ratio=self.config.keep_aspect_ratio,
            interpolation_method=self.config.interpolation,
            padding_color=self.config.padding_color,
        )
        batch = preprocess_image(resized)
        raw = self.predict(batch)
        return batch, inference.decode_plate(raw, self.config)


def _spatial_heatmap(data: np.ndarray) -> np.ndarray:
    """Channel-mean → min-max normalize → JET colormap → upscale to 256×128."""
    hmap = np.mean(data, axis=-1)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    colored = cv2.applyColorMap((hmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.resize(colored, (256, 128))



def _draw_logits_heatmap(
    logits_tensor: Any,            # (8, V) Keras/Torch tensor
    alphabet: str,                 # full alphabet including pad_char
    plate_text: str,               # decoded plate for title
    save_path: pathlib.Path,
) -> None:
    """Render a matplotlib heatmap of per-slot classification probabilities.

    X-axis = alphabet characters, Y-axis = 8 plate slots.
    Active-character slots get a green border; padding-character slots get a
    grey/dashed border so you can immediately see where the 8→n reduction happens.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    try:
        from matplotlib import font_manager as _fm
        _fm.fontManager.addfont(_FONT_PATH)
        _fm.FontProperties(fname=_FONT_PATH)
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    except Exception:
        pass

    logits_np = logits_tensor.detach().cpu().numpy() if hasattr(logits_tensor, "detach") else np.array(logits_tensor)
    shifted = logits_np - logits_np.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)  # (8, V)

    chars = list(alphabet)
    n_slots, n_vocab = probs.shape
    pred_indices = np.argmax(probs, axis=-1)
    pred_chars = [chars[i] for i in pred_indices]
    pad_char = chars[-1]  # pad_char always at the end of the alphabet per plate config convention

    # Determine which slots are "padding" (predicted as the pad character).
    pad_slots = [i for i, ch in enumerate(pred_chars) if ch == pad_char]
    active_slots = [i for i in range(n_slots) if i not in pad_slots]

    fig_w = max(15, n_vocab * 0.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    im = ax.imshow(probs, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1, origin="upper")

    # X-axis: character labels, pad_char column painted distinctly.
    ax.set_xticks(range(n_vocab))
    ax.set_xticklabels(chars, rotation=90, fontsize=5)
    # Purple vertical line separating the padding character.
    ax.axvline(x=n_vocab - 1.5, color="#7B1FA2", linewidth=2, linestyle="--", alpha=0.6)
    ax.text(n_vocab - 1, n_slots + 0.6, "PAD", ha="center", fontsize=5,
            color="#7B1FA2", fontweight="bold")

    # Y-axis labels.
    slot_labels = [f"[{s}]" for s in range(n_slots)]
    ax.set_yticks(range(n_slots))
    ax.set_yticklabels(slot_labels, fontsize=8)

    # Highlight argmax cells: green for active, grey/dashed for padding.
    for slot in range(n_slots):
        max_col = pred_indices[slot]
        is_pad = chars[max_col] == pad_char
        edge_color = "#AAAAAA" if is_pad else "#00E676"
        edge_style = "-" if not is_pad else (0, (3, 2))
        rect = Rectangle((max_col - 0.5, slot - 0.5), 1, 1,
                         fill=False, edgecolor=edge_color, linewidth=2.5,
                         linestyle=edge_style)
        ax.add_patch(rect)

    # Right-side annotations: predicted character + probability per slot.
    for slot in range(n_slots):
        pred_ch = pred_chars[slot]
        prob_val = probs[slot, pred_indices[slot]]
        label = f"{pred_ch} {prob_val:.2f}" if pred_ch != pad_char else f"PAD {prob_val:.2f}"
        text_color = "#999999" if pred_ch == pad_char else "#222222"
        ax.text(n_vocab + 1.5, slot, label, va="center", fontsize=6, color=text_color,
                fontweight="normal")

    # Summary box.
    n_active = len(active_slots)
    n_stripped = len(pad_slots)
    stripped_detail = ", ".join(f"slot {s}" for s in pad_slots) if pad_slots else "none"
    raw_text = "".join(pred_chars)
    summary = (
        f"Raw:  \"{raw_text}\"\n"
        f"After rstrip('{pad_char}'):  \"{plate_text}\"\n"
        f"Active slots: {n_active} / {n_slots}  |  Stripped: {stripped_detail}"
    )
    ax.text(0.5, -0.7 - len(pad_slots) * 0.15, summary, transform=ax.transAxes,
            fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    ax.set_title(f"Classification logits — 8 slots → \"{plate_text}\" ({n_active} chars)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Character (alphabet)", fontsize=8)
    ax.set_ylabel("Plate slot", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Softmax probability", fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def show_feature_maps(
    recognizer: Recognizer,
    preprocessed_input: np.ndarray,         # (1, H, W, 3) uint8
    plate_text: str,                        # decoded plate text for the title
    save_prefix: pathlib.Path,              # e.g. feat_maps_dir / "img_name"
) -> None:
    """Save intermediate feature-map visualizations from the Keras recognizer.

    Creates:
        {stem}_feat_input.png    – preprocessed plate image (what the model sees)
        {stem}_feat_conv1.png    – 1st Conv2D (64×128×48) — shallowest feature, full resolution
        {stem}_feat_pool.png     – after anti-aliased pooling (32×64×48) — half-resolution edges
        {stem}_feat_conv.png     – conv_stem output (32×64×112) — all conv layers before tokenization
        {stem}_feat_logits.png   – per-slot classification logits (8 slots × vocab)

    Requires a Keras model (``recognizer.keras_model is not None``).
    """
    keras = importlib.import_module("keras")
    model = recognizer.keras_model
    if model is None:
        LOGGER.info("feature maps require a .keras model (not ONNX)")
        return

    save_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1. Preprocessed input image.
    input_img = preprocessed_input[0]  # (H, W, 3) uint8
    input_up = cv2.resize(input_img, (256, 128), interpolation=cv2.INTER_NEAREST)
    input_bgr = cv2.cvtColor(input_up, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_prefix.parent / f"{save_prefix.stem}_feat_input.png"), input_bgr)

    # Collect available intermediate layers.
    layer_map: dict[str, Any] = {layer.name: layer for layer in model.layers}

    # 2. conv_stem internal layers — shallow CNN features.
    if "conv_stem" in layer_map:
        cs = layer_map["conv_stem"]  # Sequential: conv2d → pooling → conv2d_1 → conv2d_2 → conv2d_3
        # Run rescaling separately so we can feed it through partial stem models.
        rescale_model = keras.Model(inputs=model.input, outputs=layer_map["rescaling"].output)
        normed = rescale_model(preprocessed_input.astype(np.float32), training=False)

        # 2a. First conv (full resolution, 48 filters) — the most shallow feature map.
        if hasattr(cs, "layers") and len(cs.layers) >= 1:
            stem_to_conv1 = keras.Sequential([cs.layers[0]])  # reuses trained weights
            conv1_out = stem_to_conv1(normed, training=False)
            conv1_data = conv1_out[0].detach().cpu().numpy()  # (64, 128, 48)
            hmap = _spatial_heatmap(conv1_data)
            cv2.imwrite(str(save_prefix.parent / f"{save_prefix.stem}_feat_conv1.png"),
                        cv2.resize(hmap, (512, 256)))  # double-size for full-resolution detail

        # 2b. After anti-aliased pooling — half resolution, 48 filters.
        if hasattr(cs, "layers") and len(cs.layers) >= 2:
            stem_to_pool = keras.Sequential(list(cs.layers[:2]))
            pool_out = stem_to_pool(normed, training=False)
            pool_data = pool_out[0].detach().cpu().numpy()  # (32, 64, 48)
            cv2.imwrite(str(save_prefix.parent / f"{save_prefix.stem}_feat_pool.png"),
                        _spatial_heatmap(pool_data))

        # 2c. Full conv_stem output (32, 64, 112) — same as before.
        conv_stem = keras.Model(inputs=model.input, outputs=cs.output)
        conv_out = conv_stem(preprocessed_input.astype(np.float32), training=False)
        conv_data = conv_out[0].detach().cpu().numpy()
        cv2.imwrite(str(save_prefix.parent / f"{save_prefix.stem}_feat_conv.png"),
                    _spatial_heatmap(conv_data))

    # 3. Plate classification logits (8 slots × vocab_size) — matplotlib heatmap.
    if "plate" in layer_map:
        plate_model = keras.Model(inputs=model.input, outputs=layer_map["plate"].output)
        plate_out = plate_model(preprocessed_input.astype(np.float32), training=False)
        _draw_logits_heatmap(
            plate_out[0],     # (8, V) tensor
            alphabet=recognizer.config.alphabet,
            plate_text=plate_text,
            save_path=save_prefix.parent / f"{save_prefix.stem}_feat_logits.png",
        )


def iter_images(image_dir: pathlib.Path) -> list[pathlib.Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {image_dir} (supported: {sorted(IMAGE_SUFFIXES)})")
    return images


def draw_plate_annotation(
    img_bgr: np.ndarray,
    detections: list[tuple[np.ndarray, str, float]],
) -> np.ndarray:
    """Draw all detection boxes + text labels on a **copy** of the image."""
    annotated = img_bgr.copy()
    for i, (landmarks, plate_text, det_conf) in enumerate(detections):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]
        text_color = (255 - color[0], 255 - color[1], 255 - color[2])
        pts = landmarks.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=BBOX_THICKNESS)

        tx, ty = pts[0][0]
        label = f"[{i}] {plate_text}" if plate_text else f"[{i}] <none>"
        if det_conf > 0:
            label += f" {det_conf:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)
        text_y = ty - 40
        bg_top = text_y - text_size[1] - 10
        bg_bottom = text_y + 5
        bg_left, bg_right = tx, tx + text_size[0] + 10
        cv2.rectangle(annotated, (bg_left, bg_top), (bg_right, bg_bottom), TEXT_BG_COLOR, -1)
        cv2.putText(
            annotated, label, (tx + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, text_color, TEXT_THICKNESS,
        )
    return annotated


def label_one(
    image_path: pathlib.Path,
    detector: Yolov5FaceLandmarkDetector,
    recognizer: Recognizer,
    *,
    save_crops_dir: pathlib.Path | None,
    visualize_dir: pathlib.Path | None,
    feature_maps_dir: pathlib.Path | None,
) -> list[dict[str, Any]]:
    """Run detector+recognizer on one image, returning one row per detection."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        LOGGER.warning("unreadable image: %s", image_path)
        return [_empty_row(image_path, 0)]

    detections = detector.detect_all(img_bgr)
    if not detections:
        LOGGER.info("no plate detected: %s", image_path.name)
        return [_empty_row(image_path, 0)]

    rows: list[dict[str, Any]] = []
    viz_infos: list[tuple[np.ndarray, str, float]] = []

    for det_index, detection in enumerate(detections):
        crop_bgr = four_point_transform(img_bgr, detection.landmarks)
        try:
            pre_input, plate_text = recognizer.recognize_with_input(crop_bgr)
        except Exception as error:
            LOGGER.warning("recognition failed for %s[%d]: %s", image_path.name, det_index, error)
            plate_text = ""

        # Feature maps.
        if feature_maps_dir is not None and plate_text:
            stem = image_path.stem
            feat_prefix = feature_maps_dir / (f"{stem}_{det_index}" if det_index > 0 else stem)
            show_feature_maps(recognizer, pre_input, plate_text, feat_prefix)

        if save_crops_dir is not None:
            save_crops_dir.mkdir(parents=True, exist_ok=True)
            stem = image_path.stem
            suffix = f"_{det_index}.jpg" if det_index > 0 else ".jpg"
            cv2.imwrite(str(save_crops_dir / f"{stem}{suffix}"), crop_bgr)

        lm = detection.landmarks
        rows.append({
            "image_path": str(image_path),
            "det_index": det_index,
            "plate_text": plate_text,
            "x1": int(lm[0, 0]), "y1": int(lm[0, 1]),
            "x2": int(lm[1, 0]), "y2": int(lm[1, 1]),
            "x3": int(lm[2, 0]), "y3": int(lm[2, 1]),
            "x4": int(lm[3, 0]), "y4": int(lm[3, 1]),
            "det_conf": round(float(detection.conf), 4),
            "plate_class": detection.class_id,
        })
        viz_infos.append((detection.landmarks, plate_text, detection.conf))

    if visualize_dir is not None and viz_infos:
        visualize_dir.mkdir(parents=True, exist_ok=True)
        annotated = draw_plate_annotation(img_bgr, viz_infos)
        cv2.imwrite(str(visualize_dir / f"{image_path.stem}.jpg"), annotated)

    return rows


def _empty_row(image_path: pathlib.Path, det_index: int = 0) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "det_index": det_index,
        "plate_text": "",
        "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0, "x4": 0, "y4": 0,
        "det_conf": 0.0, "plate_class": "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image-dir", type=pathlib.Path, default=pathlib.Path("../test_data/img"),
        help="Directory of raw car/scene images (default: ../test_data/img)",
    )
    image_group.add_argument(
        "--image", type=pathlib.Path, default=None,
        help="Process a single image (overrides --image-dir); CSV output is disabled, results printed to stdout",
    )
    parser.add_argument(
        "--model", type=pathlib.Path,
        default=pathlib.Path("trained_models/cblprd_ccpd_cct_s_v2_torch/2026-07-29_13-47-11/best.keras"),
        help="Path to a .onnx or .keras fast-plate-ocr recognizer model",
    )
    parser.add_argument(
        "--plate-config", type=pathlib.Path,
        default=pathlib.Path("config/cn_plate_config.yaml"),
        help="Plate config YAML; defaults to config/cn_plate_config.yaml",
    )
    parser.add_argument(
        "--detector-repo-path", type=pathlib.Path,
        default=pathlib.Path("../license_plate_recognition/Chinese_license_plate_detection_recognition"),
        help="Clone of we0091234/Chinese_license_plate_detection_recognition",
    )
    parser.add_argument(
        "--detector-weights", type=pathlib.Path,
        default=pathlib.Path(
            "../license_plate_recognition/Chinese_license_plate_detection_recognition/weights/plate_detect.pt"
        ),
        help="Path to plate_detect.pt from the detector repo",
    )
    parser.add_argument("--detector-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--detector-img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument(
        "--recognizer-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="ONNX execution device for the recognizer",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("../test_data/annotations.csv"),
        help="Output CSV path (default: ../test_data/annotations.csv)",
    )
    parser.add_argument(
        "--save-crops", type=pathlib.Path, default=None,
        help="Optional directory to save rectified plate crops",
    )
    parser.add_argument(
        "--visualize", type=pathlib.Path, default=None,
        help="Optional directory to save annotated images with bbox+text drawn",
    )
    parser.add_argument(
        "--feature-maps-dir", type=pathlib.Path, default=None,
        help="Optional directory to save intermediate feature maps (requires .keras model)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable per-image progress logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.no_progress else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    try:
        model_path = inference.existing_file(args.model, "Recognizer model")
        config_path = inference.resolve_plate_config(model_path, args.plate_config)
        recognizer = Recognizer.from_path(model_path, config_path, args.recognizer_device)
        detector = Yolov5FaceLandmarkDetector(
            repo_path=args.detector_repo_path,
            weights_path=args.detector_weights,
            device=args.detector_device,
            img_size=args.detector_img_size,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )
        images: list[pathlib.Path]
        single_mode = False
        if args.image:
            img_path = inference.existing_file(args.image, "Image")
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image format: {img_path.suffix}")
            images = [img_path]
            single_mode = True
            # Disable CSV in single mode — print to stdout instead.
            csv_out = None
        else:
            images = iter_images(args.image_dir)
            csv_out = args.output

        if csv_out:
            csv_out.parent.mkdir(parents=True, exist_ok=True)

        detected = 0
        handle = csv_out.open("w", encoding="utf-8", newline="") if csv_out else None
        try:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS) if handle else None
            if writer:
                writer.writeheader()
            for idx, img in enumerate(images, 1):
                rows = label_one(
                    img, detector, recognizer,
                    save_crops_dir=args.save_crops,
                    visualize_dir=args.visualize,
                    feature_maps_dir=args.feature_maps_dir,
                )
                for row in rows:
                    if writer:
                        writer.writerow(row)
                    if row["plate_text"]:
                        detected += 1
                if single_mode:
                    # Print results to stdout for single-image mode.
                    print(f"Image: {img}")
                    for row in rows:
                        status = row["plate_text"] or "<none>"
                        print(f"  [{row['det_index']}] plate={status}  conf={row['det_conf']:.4f}  class={row['plate_class']}")
                        if row["plate_text"]:
                            print(f"       corners: ({row['x1']},{row['y1']}) ({row['x2']},{row['y2']}) "
                                  f"({row['x3']},{row['y3']}) ({row['x4']},{row['y4']})")
                    if args.visualize:
                        print(f"  visual: {args.visualize / f'{img.stem}.jpg'}")
                    if args.feature_maps_dir:
                        for row in rows:
                            if row["plate_text"]:
                                suffix = f"_{row['det_index']}" if row['det_index'] > 0 else ""
                                pref = args.feature_maps_dir / f'{img.stem}{suffix}'
                                print(f"  feat:   {pref}_feat_input.png")
                                print(f"  feat:   {pref}_feat_conv1.png")
                                print(f"  feat:   {pref}_feat_pool.png")
                                print(f"  feat:   {pref}_feat_conv.png")
                                print(f"  feat:   {pref}_feat_logits.png")
                elif not args.no_progress:
                    plates = [r["plate_text"] for r in rows if r["plate_text"]]
                    confs = [f"{r['det_conf']:.3f}" for r in rows if r["plate_text"]]
                    summary = ", ".join(f"{p}({c})" for p, c in zip(plates, confs)) if plates else "<none>"
                    LOGGER.info("[%d/%d] %s -> %s", idx, len(images), img.name, summary)
        finally:
            if handle:
                handle.close()
        total = len(images)
        LOGGER.warning(
            "done: %d detections across %d images%s",
            detected, total, f" -> {csv_out}" if csv_out else "",
        )
        return 0
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
