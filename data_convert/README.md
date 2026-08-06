# data_convert/

数据集转换、裁剪和合并脚本。将各类开源车牌数据集转换为统一的 fast-plate-ocr 训练格式。

---

## `CBLPRD2Fastocr.py` — CBLPRD-330K 数据处理

### separate 子命令：分离单/双层车牌

```bash
python3 data_convert/CBLPRD2Fastocr.py separate \
  --dataset-root /path/to/CBLPRD-330K \
  --out-dir /path/to/output
```

| 参数 | 说明 |
|------|------|
| `--dataset-root` | CBLPRD-330K 数据集根目录（含 `image/` + `label/`） |
| `--out-dir` | 输出目录 |
| `--max-plate-slots` | 过滤超过指定位数的车牌（默认不过滤） |

输出：`single_layer/` 和 `double_layer/` 两个 fastocr 格式目录，可直接用于训练。

### convert 子命令：转换为 fastocr 格式

```bash
python3 data_convert/CBLPRD2Fastocr.py convert \
  --source-dir /path/to/source \
  --out-dir /path/to/output \
  --plate-config config/cn_plate_config.yaml
```

| 参数 | 说明 |
|------|------|
| `--source-dir` | 包含 `train/annotations.csv` 和 `val/annotations.csv` 的源目录 |
| `--out-dir` | 输出目录 |
| `--max-plate-slots` | 过滤超长车牌 |
| `--plate-config` | 指定 plate_config.yaml（不指定则自动生成） |

---

## `CCPD2Fastocr.py` — CCPD2019 转换

```bash
python3 data_convert/CCPD2Fastocr.py \
  --dataset-root /path/to/CCPD2019 \
  --out-dir /path/to/output \
  --workers 16 --val-ratio 0.2
```

| 参数 | 说明 |
|------|------|
| `--dataset-root` | CCPD2019 根目录（含 `ccpd_base/` 等分组） |
| `--out-dir` | 输出目录 |
| `--workers` | 并行线程数（默认 16） |
| `--val-ratio` | 每组内验证集比例（默认 0.2） |

输出：每组内 `train/` + `val/` 子目录，含透视校正裁剪的图片和 annotations.csv。

---

## `CCPD20202Fastocr.py` — CCPD2020 绿牌转换

```bash
python3 data_convert/CCPD20202Fastocr.py \
  --dataset-root /path/to/CCPD2020 \
  --out-dir /path/to/output \
  --workers 16
```

| 参数 | 说明 |
|------|------|
| `--dataset-root` | CCPD2020 根目录（含 `ccpd_green/train/**/val/`） |
| `--out-dir` | 输出目录 |
| `--workers` | 并行线程数（默认 16） |
| `--plate-config` | plate_config.yaml 路径（默认 `config/cn_plate_config.yaml`） |

---

## `merge_cblprd_ccpd.py` — 多数据集合并

```bash
python3 data_convert/merge_cblprd_ccpd.py \
  --cblprd-root .../single_layer \
  --ccpd-root .../CCPD/fast-plate-ocr \
  --ccpd2020-root .../CCPD2020/fast-plate-ocr \
  --challenge-root .../challenge_data \
  --out-dir .../FastOCRData \
  --cblprd-ratio 0.5 \
  --ccpd-ratio 0.5 --ccpd-anhui-ratio 0.3 --ccpd-base-ratio 0.3 \
  --ccpd2020-ratio 1.0 \
  --challenge-ratio 1.0 --challenge-val-ratio 0.2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cblprd-root` | CBLPRD-330K/separate/single_layer | CBLPRD 单层车牌目录 |
| `--cblprd-ratio` | 1.0 | CBLPRD 添加比例 (0-1) |
| `--ccpd-root` | CCPD/fast-plate-ocr | CCPD2019 转换后目录 |
| `--ccpd-ratio` | 1.0 | CCPD2019 非安徽车牌添加比例 |
| `--ccpd-anhui-ratio` | 1.0 | CCPD2019 安徽车牌添加比例（独立控制） |
| `--ccpd-base-ratio` | 1.0 | ccpd_base 组总比例（覆盖 anhui-ratio，均匀抽样安徽和非安徽） |
| `--ccpd2020-root` | CCPD2020/fast-plate-ocr | CCPD2020 转换后目录 |
| `--ccpd2020-ratio` | 1.0 | CCPD2020 添加比例 |
| `--challenge-root` | 无 | challenge_data 目录 |
| `--challenge-ratio` | 0（不添加） | challenge_data 添加比例 |
| `--challenge-val-ratio` | 0.2 | challenge_data 验证集划分比例 |
| `--out-dir` | 必需 | 输出目录 |
| `--plate-config` | `config/cn_plate_config.yaml` | 车牌配置 |

---

## `export_ccpd2020_labels.py` — CCPD2020 标注导出

```bash
python3 data_convert/export_ccpd2020_labels.py \
  --dataset-root /path/to/CCPD2020 \
  --workers 8
```

| 参数 | 说明 |
|------|------|
| `--dataset-root` | CCPD2020 数据集根目录 |
| `--workers` | 并行解析线程数（默认 1） |

从文件名解析嵌入的多边形坐标和车牌文本，输出 per-split CSV 文件。

---

## `plate_detector_adapter.py` — 车牌检测器适配

```bash
python3 data_convert/plate_detector_adapter.py \
  --image /path/to/scene.jpg
```

| 参数 | 说明 |
|------|------|
| `--repo-path` | YOLO 检测器仓库路径 |
| `--weights` | 检测器权重文件 |
| `--image` | 测试图片 |

调用 YOLO 检测模型获取车牌 4 角坐标（左上/右上/右下/左下）和单/双层分类（class_id: 0=单层, 1=双层）。

---

## 典型工作流

```bash
# 1. 分离 CBLPRD 数据集
python3 data_convert/CBLPRD2Fastocr.py separate \
  --dataset-root .../CBLPRD-330K --out-dir .../cblprd_separated

# 2. 转换 CCPD2019
python3 data_convert/CCPD2Fastocr.py \
  --dataset-root .../CCPD2019 --out-dir .../ccpd_output --val-ratio 0.2

# 3. 转换 CCPD2020
python3 data_convert/CCPD20202Fastocr.py \
  --dataset-root .../CCPD2020 --out-dir .../ccpd2020_output

# 4. 合并训练数据（调整各数据源比例）
python3 data_convert/merge_cblprd_ccpd.py \
  --cblprd-root .../cblprd_separated/single_layer \
  --ccpd-root .../ccpd_output \
  --ccpd2020-root .../ccpd2020_output \
  --challenge-root .../challenge_data \
  --out-dir .../FastOCRData \
  --cblprd-ratio 0.5 --ccpd-ratio 0.5 --ccpd-anhui-ratio 0.3 --ccpd2020-ratio 1.0
```
