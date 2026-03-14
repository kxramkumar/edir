# EDIR — Project and data layout

This document summarizes the project structure and the **art** layout used by the notebooks. See also [README.md](../README.md) and `nbs/eye_disease_v3_506.ipynb` (Setup section).

## Data layout: art/rgb and art/grey

Data and outputs are organized by **variant** so RGB and greyscale pipelines stay separate:

| Path | Purpose |
|------|---------|
| `art/rgb/data/raw/train`, `art/rgb/data/raw/validate` | Raw RGB images (one subfolder per class). |
| `art/rgb/data/clean/...` | Cleaned/processed RGB images. |
| `art/rgb/csv/raw` | Manifest CSV for raw RGB (e.g. `image_manifest.csv`). |
| `art/rgb/csv/clean` | Manifest CSV for cleaned RGB (optional). |
| `art/rgb/report` | Reports and profiling for RGB. |
| `art/grey/data/raw/...` | Raw greyscale images. |
| `art/grey/data/clean/...` | Cleaned greyscale images. |
| `art/grey/csv/raw`, `art/grey/csv/clean` | CSV manifests for grey variant. |
| `art/grey/report` | Reports for grey variant. |

Each of **rgb** and **grey** has its own **data**, **csv**, and **report** (and any other) folders.

## Notebook

In `nbs/eye_disease_v3_506.ipynb`, set **`DATA_VARIANT = "rgb"`** or **`DATA_VARIANT = "grey"`** in the Setup section to point at the desired variant. All paths (raw_train_dir, raw_val_dir, clean_*, csv_raw_dir, report_dir) are derived from `ART_ROOT` and `DATA_VARIANT`.
