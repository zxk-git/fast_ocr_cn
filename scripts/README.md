# scripts/

训练、评估、推理和工作流脚本。

---

## `plate_workflow.py` — 统一工作流入口

```bash
# 环境安装
python3 scripts/plate_workflow.py setup [--skip-verify]

# 训练
python3 scripts/plate_workflow.py train --dataset-root /path/to/data --epochs 200 --lr 0.001 --gpu 0

# 微调
python3 scripts/plate_workflow.py fine-tune best.keras --dataset-root /path/to/new_data --epochs 50 --lr 1e-4 --gpu 0

# 导出 ONNX
python3 scripts/plate_workflow.py export best.keras
```

### train / fine-tune 共用参数

| 参数 | train 默认值 | fine-tune 默认值 | 说明 |
|------|-------------|------------------|------|
| `--dataset-root` | `.../asserts/FastOCRData` | 同 | 训练数据根目录（含 train/val） |
| `--model-config` | `models/cct_s_v2.yaml` | 同 | 模型结构配置 YAML |
| `--plate-config` | `config/cn_plate_config.yaml` | 同 | 字符集配置 YAML |
| `--epochs` | 200 | 50 | 训练轮数 |
| `--batch-size` | 1024 | 1024 | batch size |
| `--lr` | 0.001 | 0.0001 | 初始学习率 |
| `--early-stopping-patience` | 20 | 10 | 早停耐心值 |
| `--early-stopping-metric` | val_plate_acc | 同 | 早停监控指标 |
| `--workers` | 16 | 16 | 数据加载线程数 |
| `--gpu` | 无 | 无 | GPU 设备编号 |
| `--quick` | 否 | 否 | 3 个 epoch 快速验证 |

### train 独有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `trained_models/cblprd_cct_s_v2_torch` | 模型输出目录 |

### fine-tune 独有参数

| 参数 | 说明 |
|------|------|
| `model` | 预训练模型 .keras 路径（位置参数，不指定时从 `--search-root` 查找最新） |
| `--search-root` | 查找最新 best.keras 的根目录（默认 `trained_models/`） |
| `--output-dir` | 微调输出目录（默认 `trained_models/fine_tuned/`） |

### export 参数

| 参数 | 说明 |
|------|------|
| `model` | Keras 模型路径（位置参数，不指定时从 `--search-root` 查找最新） |
| `--search-root` | 查找最新 best.keras 的根目录 |
| `--plate-config` | plate_config.yaml 路径（默认模型同目录或 `config/cn_plate_config.yaml`） |
| `--output-dir` | ONNX 输出目录（默认模型同目录） |

---

## `test_single_image.py` — 单张图片推理

```bash
# ONNX 模型
python3 scripts/test_single_image.py \
  --image /path/to/cropped_plate.jpg \
  --model /path/to/model.onnx \
  --plate-config config/cn_plate_config.yaml \
  --device cuda

# Keras 模型
python3 scripts/test_single_image.py \
  --image /path/to/cropped_plate.jpg \
  --model /path/to/best.keras \
  --plate-config config/cn_plate_config.yaml \
  --keras-backend torch
```

| 参数 | 说明 |
|------|------|
| `--image` | 裁剪好的车牌图片路径 |
| `--model` | `.onnx` 或 `.keras` 模型路径 |
| `--plate-config` | 车牌配置文件，默认从模型同目录查找 |
| `--keras-backend` | Keras 后端：torch / tensorflow / jax |
| `--device` | ONNX 执行设备：auto / cpu / cuda |

---

## `compare_model_outputs.py` — Keras vs ONNX 输出对比

```bash
python3 scripts/compare_model_outputs.py \
  --image /path/to/cropped_plate.jpg \
  --keras-model /path/to/best.keras \
  --onnx-model /path/to/model.onnx \
  --plate-config config/cn_plate_config.yaml \
  --keras-backend torch --device cuda
```

| 参数 | 说明 |
|------|------|
| `--image` | 测试图片 |
| `--keras-model` | Keras 模型路径 |
| `--onnx-model` | ONNX 模型路径 |
| `--plate-config` | 车牌配置 |
| `--keras-backend` | Keras 后端 |
| `--device` | ONNX 设备 |
| `--atol` | 绝对容差 (默认 1e-5) |
| `--rtol` | 相对容差 (默认 1e-3) |

---

## `auto_label_plates.py` — 自动标注

```bash
# 标注单张图
python3 scripts/auto_label_plates.py --image /path/to/scene.jpg

# 批量标注目录
python3 scripts/auto_label_plates.py --image-dir /path/to/images/ --output labels.csv

# 保存裁剪的车牌
python3 scripts/auto_label_plates.py --image /path/to/scene.jpg --save-crops ./crops/
```

| 参数 | 说明 |
|------|------|
| `--image` | 单张原始场景图片 |
| `--image-dir` | 批量标注目录 |
| `--output` | 输出 CSV 路径 |
| `--save-crops` | 保存裁剪车牌图片的目录 |
| `--visualize` | 保存可视化标注结果 |
| `--model` | OCR 模型路径（ONNX） |
| `--plate-config` | 车牌配置 |
| `--detector-repo-path` | YOLO 检测器仓库路径 |
| `--detector-weights` | 检测器权重路径 |
| `--detector-device` | 检测器设备：cpu / cuda |

---

## `verify_pytorch_training_env.py` — 环境验证

```bash
KERAS_BACKEND=torch \
FAST_PLATE_OCR_DATASET_ROOT=/path/to/FastOCRData \
python3 scripts/verify_pytorch_training_env.py
```

验证 PyTorch backend、GPU 可用性、数据集完整性、字符集和一次模型前向计算。
