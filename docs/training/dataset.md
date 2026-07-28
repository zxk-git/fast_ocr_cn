# Dataset

This page describes the expected format for datasets used when training models with `fast-plate-ocr`.

---

## Expected File Format

Your dataset should be provided as a **CSV file** with the following structure:

| Column Name    | Type  | Description                                                |
|----------------|-------|------------------------------------------------------------|
| `image_path`   | `str` | Relative path to the license plate image                   |
| `plate_text`   | `str` | Ground-truth text on the plate (no padding)                |
| `plate_region` | `str` | Optional region/country label (must match `plate_regions`) |

The `plate_region` column is **optional**. If it is present **and** your plate config defines `plate_regions`,
training will enable the **region head** and learn region predictions alongside plate text.


!!! info "Relative Paths"
    Image paths in the CSV are resolved **relative to the location of the CSV file**, not the working directory.

---

### Dataset Structure Example

The dataset should include at least a **CSV file** (for training and validation) and a folder with corresponding
plate images.

```text
dataset/
├── train/
│   ├── annotations.csv
│   └── images/
│       ├── 00001.jpg
│       ├── 00002.jpg
│       ├── 00003.jpg
│       ├── 00004.jpg
│       └── 00005.jpg
└── val/
    ├── annotations.csv
    └── images/
        ├── 00006.jpg
        └── 00007.jpg
```

```csv title="train/annotations.csv (no region)"
image_path,plate_text
images/00001.jpg,KNN505
images/00002.jpg,J00NCW
images/00003.jpg,48593
images/00004.jpg,AB123CD
images/00005.jpg,17AB
```

```csv title="train/annotations.csv (with region)"
image_path,plate_text,plate_region
images/00001.jpg,KNN505,Argentina
images/00002.jpg,J00NCW,Argentina
images/00003.jpg,48593,Argentina
images/00004.jpg,AB123CD,Brazil
images/00005.jpg,17AB,Chile
```

```csv title="val/annotations.csv (with region)"
image_path,plate_text,plate_region
images/00006.jpg,NFM374,Argentina
images/00007.jpg,ZXC9871,Chile
```
