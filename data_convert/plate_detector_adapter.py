#!/usr/bin/env python3
"""
plate_detector_adapter.py

对接 https://github.com/we0091234/Chinese_license_plate_detection_recognition
的车牌检测模型（YOLOv5-face 结构，输出 bbox + 4 个角点 landmark + 单双层分类）。

使用方式：
    1. git clone https://github.com/we0091234/Chinese_license_plate_detection_recognition
    2. 按其 README 下载 weights/plate_detect.pt
    3. 本文件放在你自己的转换脚本旁边，用 --detector-repo-path 指向 clone 下来的目录

这个适配层只做"检测 + 透视矫正"，不加载、不使用该仓库的识别模型(plate_rec_model)，
因为我们已经有 CBLPRD 提供的 ground-truth plate_text，不需要它再识别一遍。

重要说明（诚实告知，没有回避）：
    - four_point_transform / order_points 是从该仓库 detect_plate.py 原样抄过来的，
      已经用合成的倾斜四边形单独测试过，矫正逻辑本身没问题。
    - 但 Yolov5FaceLandmarkDetector 这个类依赖该仓库自带的 models/、utils/（尤其是
      non_max_suppression_face，这是标准 yolov5 utils 里没有的，是该仓库自己加的）,
      以及 PyTorch 和它提供的权重文件。这几样在我这边的沙盒里无法安装/下载（沙盒不能
      联网、也没有对应权重），所以这个类本身没有被我实际跑通过，只是严格照着源码抄的。
      建议你在自己环境里先用少量图片跑一遍 --self-check 确认能正常检测，再上全量数据。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 几何变换：原样抄自 detect_plate.py，已用合成数据单独验证过
# --------------------------------------------------------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """四个点按 左上、右上、右下、左下 排列（仓库源码里保留但未启用，这里同样保留以备用）。"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """按 4 个角点(左上,右上,右下,左下)做透视变换，得到摆正的车牌小图。"""
    rect = pts.astype("float32")
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


# --------------------------------------------------------------------------
# 检测结果 & 检测器接口
# --------------------------------------------------------------------------
@dataclass
class PlateDetection:
    landmarks: np.ndarray  # shape (4,2)，顺序：左上、右上、右下、左下
    class_id: int          # 0 = 单层，1 = 双层（来自检测模型自己的分类头）
    conf: float

class _legacy_full_model_load:
    def __init__(self, torch_module):
        self._torch = torch_module
        self._original_load = torch_module.load
 
    def __enter__(self):
        original_load = self._original_load
 
        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)
 
        self._torch.load = patched_load
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._torch.load = self._original_load
        return False
    
    
class Yolov5FaceLandmarkDetector:
    """封装 we0091234/Chinese_license_plate_detection_recognition 的检测模型推理。

    只暴露一个方法 detect(img_bgr) -> PlateDetection | None，
    取置信度最高的一个检测框；一张图片理论上只有一块车牌（CBLPRD 已经是裁剪好的单车牌图）。
    """

    def __init__(
        self,
        repo_path: Path,
        weights_path: Path,
        device: str = "cpu",
        img_size: int = 640,
        conf_thres: float = 0.3,
        iou_thres: float = 0.5,
    ):
        repo_path = Path(repo_path).resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            import torch
            from models.experimental import attempt_load
            from utils.datasets import letterbox
            from utils.general import check_img_size, non_max_suppression_face, scale_coords
        except ImportError as e:
            raise ImportError(
                "加载检测模型依赖失败：请确认 --detector-repo-path 指向的是 "
                "we0091234/Chinese_license_plate_detection_recognition 的 clone 目录，"
                "并且已安装该仓库 requirements.txt 里的依赖（torch 等）。"
                f"原始错误：{e}"
            ) from e

        self._torch = torch
        self._letterbox = letterbox
        self._nms = non_max_suppression_face
        self._scale_coords = scale_coords

        self.device = torch.device(device)
        try:
            with _legacy_full_model_load(torch):
                self.model = attempt_load(str(weights_path), map_location=self.device)
        except Exception as e:
            raise RuntimeError(
                f"加载权重失败: {weights_path}\n"
                "如果报错信息里出现 UnpicklingError / weights_only，说明是 PyTorch>=2.6 "
                "默认行为变化导致的兼容问题，本应该已经被自动处理；如果仍然失败，"
                "可以手动改仓库里 models/experimental.py 第 118 行左右，"
                "把 torch.load(w, map_location=map_location) 改成 "
                "torch.load(w, map_location=map_location, weights_only=False)。"
                f"\n原始错误：{e}"
            ) from e
        self.img_size = check_img_size(img_size, s=self.model.stride.max())
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _scale_coords_landmarks(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        coords[:, [0, 2, 4, 6]] -= pad[0]
        coords[:, [1, 3, 5, 7]] -= pad[1]
        coords[:, :8] /= gain
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        coords[:, 4].clamp_(0, img0_shape[1])
        coords[:, 5].clamp_(0, img0_shape[0])
        coords[:, 6].clamp_(0, img0_shape[1])
        coords[:, 7].clamp_(0, img0_shape[0])
        return coords

    def detect(self, img_bgr: np.ndarray) -> "PlateDetection | None":
        torch = self._torch
        orgimg = img_bgr
        h0, w0 = orgimg.shape[:2]
        r = self.img_size / max(h0, w0)
        img0 = orgimg
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(orgimg, (int(w0 * r), int(h0 * r)), interpolation=interp)

        img = self._letterbox(img0, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()
        img_t = torch.from_numpy(img).to(self.device).float() / 255.0
        if img_t.ndimension() == 3:
            img_t = img_t.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_t)[0]
        pred = self._nms(pred, self.conf_thres, self.iou_thres)

        best = None
        for det in pred:
            if not len(det):
                continue
            det[:, :4] = self._scale_coords(img_t.shape[2:], det[:, :4], orgimg.shape).round()
            det[:, 5:13] = self._scale_coords_landmarks(img_t.shape[2:], det[:, 5:13], orgimg.shape).round()
            for j in range(det.size()[0]):
                conf = float(det[j, 4].cpu().numpy())
                if best is not None and conf <= best[0]:
                    continue
                landmarks_flat = det[j, 5:13].view(-1).tolist()
                class_num = int(det[j, 13].cpu().numpy())
                landmarks = np.array(
                    [[landmarks_flat[2 * i], landmarks_flat[2 * i + 1]] for i in range(4)],
                    dtype="float32",
                )
                best = (conf, landmarks, class_num)

        if best is None:
            return None
        conf, landmarks, class_num = best
        return PlateDetection(landmarks=landmarks, class_id=class_num, conf=conf)


def self_check(repo_path: Path, weights_path: Path, sample_image: Path) -> None:
    """给用户的最小自检脚本：确认检测器能在自己的环境里正常加载并跑通一张图。
    用法：python plate_detector_adapter.py --repo-path <repo> --weights <weights_path> --image <一张车牌图>
    """
    detector = Yolov5FaceLandmarkDetector(repo_path, weights_path)
    img = cv2.imread(str(sample_image))
    if img is None:
        raise SystemExit(f"读不到图片: {sample_image}")
    result = detector.detect(img)
    if result is None:
        print("未检测到车牌（在这张图上）。")
        return
    print(f"检测到车牌: class_id={result.class_id}({'双层' if result.class_id else '单层'}), "
          f"conf={result.conf:.3f}, landmarks={result.landmarks.tolist()}")
    rectified = four_point_transform(img, result.landmarks)
    out_path = sample_image.parent / f"rectified_{sample_image.name}"
    cv2.imwrite(str(out_path), rectified)
    print(f"矫正后的图片已保存到: {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="检测器自检工具")
    ap.add_argument("--repo-path", type=Path, default="Chinese_license_plate_detection_recognition")
    ap.add_argument("--weights", type=Path, default="Chinese_license_plate_detection_recognition/weights/plate_detect.pt")
    ap.add_argument("--image", type=Path, default="/disk01/zdg/plate/Chinese_license_plate_detection_recognition/imgs/double_yellow.jpg")
    args = ap.parse_args()
    self_check(args.repo_path, args.weights, args.image)