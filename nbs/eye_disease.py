#!/usr/bin/env python
# coding: utf-8

# # Automated Eye Disease Classification
# Single notebook for raw data exploration, cleaning (preprocess + augment), and before/after comparison for an eye-disease image dataset (e.g. cataract, diabetic_retinopathy, glaucoma, normal). **Flow**: Setup → Collection (raw manifest) → EDA → Preprocess (prep) → Augment (aug). Data under `res/_common/` (raw) and `res/<RUN_TAG>/` (prep, aug, meta).

# ## 1. Introduction — Problem statement and pipeline overview
#     
# **Problem statement**  
# Conditions such as cataract, glaucoma, diabetic retinopathy, and AMD often go undetected early due to subtle signs and limited access to specialists. Manual diagnosis is expert-dependent, slow, and variable—especially at scale—which can lead to delayed or missed treatment and preventable vision loss. Automated, image-based classification can support early detection, aid clinicians, and improve screening reach and consistency.
# 
# **Purpose of this pipeline**  
# This notebook prepares ocular image data to support such a system. It (1) **collects** a per-image manifest from raw train/validate class folders (paths, metadata, readability, dimensions, brightness, blur, and CleanVision issue flags); (2) **explores** the raw data via class balance, format, empty/corrupt files, image size and aspect ratio, exposure (dark/light), duplicates, blur, and low-information flags, plus RGB/greyscale histograms and edge/preprocessing samples; (3) **preprocesses** train images by removing exact duplicates, correcting odd sizes (crop black borders + resize), enhancing blur (unsharp mask) and dark (CLAHE + smooth); (4) **augments** by copying prep outputs to an aug tree and generating additional train images (rotation, zoom, horizontal flip) up to 2000 per class. All outputs live under `res/<RUN_TAG>/` (meta + data/prep, data/aug). Clean, well-documented data is essential for training a reliable classifier; this pipeline ensures consistent paths, issue tracking, and before/after checks at each step.
# 
# **Conventions:** Skip `.DS_Store`; manifest paths relative to **RES_ROOT**; run from `nbs/`.  
# **Prerequisites:** `res/_common/data/raw/train` and `res/_common/data/raw/validate` with class subfolders; optional **CleanVision** for issue detection.
# 

# ## 2. Setup — Imports and constants
# - **Imports:** stdlib, numpy, cv2, pandas, seaborn, matplotlib, plotly, PIL, tqdm, IPython, Keras ImageDataGenerator; optional **CleanVision** Imagelab, **imagehash**.
# - **Plot style:** seaborn whitegrid.
# - **Constants:** **COLOR_PALETTE**, **BLUR_BINS**, **get_blur_bin()** for Laplacian variance bins (very blurry → very sharp).

# In[2]:


# Standard library
import os
import sys
import shutil
import random
import shutil
import contextlib
import pickle as _pickle
from pathlib import Path

# Third-party
import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from tqdm.notebook import tqdm
from IPython.display import display, HTML
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split

try:
    from cleanvision import Imagelab
except ImportError:
    Imagelab = None
import imagehash 

# Plot style & palette
sns.set_style("whitegrid")

# Common color palette (single source of truth)
COLOR_PALETTE = [
    "#DC2626",  # red
    "#059669",  # emerald (green)
    "#D97706",  # amber (orange)
    "#7C3AED",  # violet (purple)
    "#DB2777",  # pink
    "#0891B2",  # cyan
    "#65A30D",  # lime (yellow-green)
    "#4F71B9",  # blue
]

# For Total in sunburst
TOTAL_COLOR = "#64748B"

# Distribution chart
DISTRIBUTION_COLOR = "#64748B" 

# Blur bins (Laplacian variance)
BLUR_BINS = [
    ("0–50 (very blurry)",  0,   50),
    ("50–150 (blurry)",     50,  150),
    ("150–300 (moderate)",  150, 300),
    ("300–500 (sharp)",     300, 500),
    ("500+ (very sharp)",   500, float("inf")),
]
# Blur order
BLUR_ORDER = [label for label, _, _ in BLUR_BINS]
# Blur function
def get_blur_bin(val):
    """Return blur bin label for a single Laplacian variance value."""
    for label, lo, hi in BLUR_BINS:
        if lo <= val < hi:
            return label
    return "unknown"


# ## 3. Paths — Resource root and run tag
# Define **RES_ROOT** (`../res`), **RUN_TAG**; validate `res/_common/data/raw/train` and `res/_common/data/raw/validate` exist with class subfolders. Set **common_raw_dir**, **raw_train_dir**, **raw_val_dir**; **prep_dir**; **aug_train_dir**, **aug_val_dir**, **aug_test_dir**; **meta_raw_dir**, **meta_prep_dir**, **meta_aug_dir**. **Raw / prep manifest (before §7.6):** **split** is only **train** or **validate** (latter = raw `validate/` folder). **After §7.6:** **train** | **validate** (10% stratified from train) | **test** (renamed from former holdout **validate**). Paths: **raw_manifest_path**, **prep_manifest_path**, **aug_manifest_path**; **aug_copy_lock_path**, **aug_generate_lock_path**; **aug_lock_path** (`augment.lock`, augment section lock); **rgb_hist_cache_path**, **gray_hist_cache_path**. Create all run-tag dirs. Display run summary (resource root, common raw, run data/meta).
# 
# - `res/_common/data/raw/train/<class>/..`       — original train images, shared
# - `res/_common/data/validate/<class>/..`  — original validate images, shared
# - `res/<run_tag>/data/prep/<step_dir>/train|validate/<class>/..` — intermediate per step
# - `res/<run_tag>/data/clean/train|validate/<class>/..` — final cleaned images
# - `res/<run_tag>/meta/raw/..`  — EDA outputs (manifest, caches)
# - `res/<run_tag>/meta/prep/..` — prep pipeline outputs
# - `res/<run_tag>/meta/clean/..` — cleaning pipeline outputs

# In[7]:


RUN_TAG  = "rgb_v1"
RES_ROOT = os.path.join("..", "res")

if not os.path.isdir(RES_ROOT):
    raise FileNotFoundError(
        f"Resource folder not found: {os.path.abspath(RES_ROOT)}\n"
        "Create it at project root (e.g. mkdir res)."
    )

# Common raw data (train and validate)
# Train: `res/_common/data/raw/train/<class>/..` ; Validate: `res/_common/data/validate/<class>/..`
common_raw_dir = os.path.join(RES_ROOT, "_common", "data", "raw")
raw_train_dir  = os.path.join(RES_ROOT, common_raw_dir, "train" )
raw_val_dir    = os.path.join(RES_ROOT,  common_raw_dir, "validate" )

for _dir, _label in [(raw_train_dir, "train"), (raw_val_dir, "validate")]:
    if not os.path.isdir(_dir):
        raise FileNotFoundError(
            f"Raw {_label} folder not found: {os.path.abspath(_dir)}\n"
            f"Expected: res/_common/data/raw/<class>/.. or res/_common/data/validate/<class>/.."
        )
    _files = [f for f in os.listdir(_dir) if not f.startswith(".")]
    if not _files:
        raise FileNotFoundError(
            f"Raw {_label} folder is empty: {os.path.abspath(_dir)}\n"
            f"Add class subfolders with images under res/_common/data/raw/ or res/_common/data/validate/"
        )

# Run-tag directories (data + meta)
# Prep: `res/<run_tag>/data/prep/<step_dir>/train|<validate>/<class>/..`
# Augment: `res/<run_tag>/data/aug/train|validate|test/<class>/..` (manifest **validate** = dev; **test** = held-out)
prep_dir        = os.path.join(RES_ROOT, RUN_TAG, "data", "prep")
aug_train_dir   = os.path.join(RES_ROOT, RUN_TAG, "data", "aug", "train")
aug_val_dir     = os.path.join(RES_ROOT, RUN_TAG, "data", "aug", "validate")
aug_test_dir    = os.path.join(RES_ROOT, RUN_TAG, "data", "aug", "test")
meta_raw_dir    = os.path.join(RES_ROOT, RUN_TAG, "meta", "raw")
meta_prep_dir   = os.path.join(RES_ROOT, RUN_TAG, "meta", "prep")
meta_aug_dir  = os.path.join(RES_ROOT, RUN_TAG, "meta", "aug")

# Manifest and cache paths
raw_manifest_path    = os.path.join(meta_raw_dir,  "image_manifest.csv")
prep_manifest_path  = os.path.join(meta_prep_dir, "image_manifest.csv")
aug_manifest_path  = os.path.join(meta_aug_dir, "image_manifest.csv")
aug_copy_lock_path     = os.path.join(meta_aug_dir, "aug_copy.lock")
aug_generate_lock_path = os.path.join(meta_aug_dir, "aug_generate.lock")
aug_lock_path          = os.path.join(meta_aug_dir, "augment.lock")
rgb_hist_cache_path  = os.path.join(meta_raw_dir,  "rgb_hist_cache.pkl")
gray_hist_cache_path = os.path.join(meta_raw_dir,  "gray_hist_cache.pkl")

# Create meta and data dirs (no report folder)
for _dir in [meta_raw_dir, meta_prep_dir, meta_aug_dir, prep_dir, aug_train_dir, aug_val_dir, aug_test_dir]:
    os.makedirs(_dir, exist_ok=True)

rows = [
    ("Resource root", os.path.abspath(RES_ROOT)),
    ("Common raw (train)", os.path.abspath(raw_train_dir)),
    ("Common validate",    os.path.abspath(raw_val_dir)),    
    ("Run data",   os.path.abspath(os.path.join(RES_ROOT, RUN_TAG, "data"))),
    ("Run meta",   os.path.abspath(os.path.join(RES_ROOT, RUN_TAG, "meta"))),
]
html = f"<p style='margin-bottom:6px'><strong>Run: <span style='color:#1e3a5f'>{RUN_TAG}</span></strong></p>"
html += "<ul style='font-family:monospace;font-size:13px;margin:0;padding-left:20px'>"
for k, v in rows:
    html += f"<li><span style='color:#555'>{k}:</span> {v}</li>"
html += "</ul>"
display(HTML(html))


# ## 4. Common methods
# Helper functions used across collection and EDA: **list_image_files(root_dir)** — walk class subfolders, return list of (absolute path, class_name); skip `.DS_Store` and non-image extensions. **path_from_root(full_path)** — path relative to **RES_ROOT**. **display_wide(df, …)** — render DataFrame as HTML table with fixed layout and optional **max_height**. **show_attr_charts(df, group_col)** — horizontal bar + sunburst + optional heatmap (when per-class columns exist). **show_class_distribution(df, attr_col, title, classes)** — one KDE (or bar if no variation) per class. **show_attr_samples(df, group_col, title, …)** — sample images in a grid grouped by **group_col**; images loaded from `RES_ROOT` + **path**.

# In[8]:


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
SKIP_NAMES = ('.DS_Store', 'Thumbs.db')

def list_image_files(root_dir):
    """
    Walk root_dir (e.g. raw/train or raw/validate); return list of (path, class_name).
    Skips .DS_Store and non-image files. Path is absolute.
    """
    files = []
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        return files
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname in SKIP_NAMES:
                continue
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                files.append((os.path.join(class_path, fname), class_name))
    return files

def path_from_root(full_path):
    """Return path relative to resource root (RES_ROOT); trim everything before that."""
    res_abs = os.path.abspath(os.path.normpath(RES_ROOT)).rstrip(os.sep)
    path_abs = os.path.normpath(full_path)
    if not os.path.isabs(path_abs):
        path_abs = os.path.abspath(path_abs)
    if path_abs.startswith(res_abs):
        rel = path_abs[len(res_abs):].lstrip(os.sep)
        return rel.replace("\\", "/")
    return os.path.normpath(full_path).replace("\\", "/")

def display_wide(df, col_widths=None, show_index=False, max_height=None):
    # Build HTML with inline styles
    html = '<table style="width:100%; table-layout:fixed; border-collapse:collapse;">'

    # Header row with blue background
    html += '<tr>'
    for i, col in enumerate(df.columns):
        align = 'left' if i == 0 else ('right' if i == len(df.columns)-1 else 'center')
        html += f'<th style="background-color:#4472C4; color:white; padding:8px; text-align:{align}">{col}</th>'
    html += '</tr>'

    # Data rows
    for _, row in df.iterrows():
        html += '<tr>'
        for i, val in enumerate(row):
            align = 'left' if i == 0 else ('right' if i == len(row)-1 else 'center')
            html += f'<td style="padding:8px; border-bottom:1px solid #ddd; text-align:{align}">{val}</td>'
        html += '</tr>'
    html += '</table>'

    if max_height:
        html = f'<div style="max-height:{max_height}px; overflow-y:auto;">{html}</div>'

    display(HTML(html))

def show_attr_charts(df, group_col):
    """Horizontal bar (left) + Sunburst (right) of value counts for group_col."""
    if df.empty or group_col not in df.columns:
        return

    labels = df[group_col].astype(str).tolist()
    values = df["total"].tolist() if "total" in df.columns else df[group_col].value_counts().tolist()
    colors = [COLOR_PALETTE[i % len(COLOR_PALETTE)] for i in range(len(labels))]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "bar"}, {"type": "sunburst"}]],
        horizontal_spacing=0.08,
    )
    max_val = max(values) if values else 1
    fig.add_trace(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors, marker_opacity=0.70,
        text=values, textposition="outside",
        textfont=dict(size=11),
        showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title_text="Count", range=[0, max_val * 1.08], row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    fig.add_trace(go.Sunburst(
        labels=["Total"] + labels,
        parents=[""] + ["Total"] * len(labels),
        values=[sum(values)] + values,
        branchvalues="total",
        marker=dict(colors=[TOTAL_COLOR] + colors),
        textinfo="value+percent parent",
        insidetextorientation="radial",
    ), row=1, col=2)

    fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20), height=320, autosize=True,
        hoverlabel=dict(namelength=0),
    )
    fig.show()

    # Heatmap: labels × classes — full width, only when per-class columns exist
    skip_cols = {"data", "split", "total", group_col}
    class_cols = [c for c in df.columns if c not in skip_cols]
    if class_cols:
        z = df[class_cols].values.tolist()
        text = [[str(v) for v in row] for row in z]
        hm = go.Figure(go.Heatmap(
            z=z, x=class_cols, y=labels,
            colorscale="YlOrRd",
            text=text, texttemplate="%{text}",
            showscale=True,
        ))
        hm.update_layout(
            margin=dict(t=10, b=30, l=20, r=20),
            height=max(160, 45 * len(labels)),
            autosize=True,
            yaxis=dict(autorange="reversed"),
            hoverlabel=dict(namelength=0),
        )
        hm.show()

def show_class_distribution(df, attr_col, title, classes, color=DISTRIBUTION_COLOR):
    """One KDE bell curve (or count bar if no variation) per class in a single row."""
    from scipy.stats import gaussian_kde
    n = len(classes)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=classes,
        shared_yaxes=False,
        horizontal_spacing=0.05,
    )
    for i, cls in enumerate(classes):
        values = df[df["class"] == cls][attr_col].dropna().values
        if len(values) < 2:
            continue
        if np.std(values) < 0.01:
            fig.add_trace(go.Bar(
                x=[f"{values[0]:.4f}"], y=[len(values)],
                marker_color=color, opacity=0.7,
                text=[f"n={len(values)}"], textposition="outside",
                showlegend=False, width=0.3,
            ), row=1, col=i+1)
        else:
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min() - 0.05, values.max() + 0.05, 300)
            fig.add_trace(go.Scatter(
                x=x_range, y=kde(x_range), mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", showlegend=False,
            ), row=1, col=i+1)
        fig.update_xaxes(title_text=title, row=1, col=i+1)

    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_layout(
        height=300, autosize=True,
        margin=dict(t=30, b=40, l=40, r=10),
        hoverlabel=dict(namelength=0),
    )
    fig.show()

def show_attr_samples(df, group_col, title, n_cols=5, order=None, max_groups=10, title_col=None):
    """Plot sample images grouped by group_col. If title_col is set, use it for per-cell title (e.g. to show size)."""
    if df.empty:
        return
    if order is not None:
        present = set(df[group_col].dropna().unique())
        uniq_vals = [v for v in order if v in present]
    elif isinstance(df[group_col].dtype, pd.CategoricalDtype):
        uniq_vals = [c for c in df[group_col].cat.categories if c in df[group_col].values]
    else:
        uniq_vals = sorted(df[group_col].unique())
    if len(uniq_vals) > max_groups:
        # Keep top groups by frequency
        top = df[group_col].value_counts().head(max_groups).index.tolist()
        uniq_vals = [v for v in uniq_vals if v in top]
        title = f"{title} (top {max_groups} of {len(df[group_col].unique())})"
    n_rows = len(uniq_vals)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(n_cols * 3, 18), 3 * n_rows))
    fig.patch.set_facecolor("#F3F4F6")
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    use_title_col = title_col and title_col in df.columns
    for i, val in enumerate(uniq_vals):
        samples = df[df[group_col] == val].head(n_cols)
        for j in range(n_cols):
            ax = axes[i, j]
            if j < len(samples):
                img_path = os.path.join(RES_ROOT, samples.iloc[j]["path"].replace("/", os.sep))
                img = cv2.imread(img_path)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="equal")
                    cell_title = str(samples.iloc[j][title_col]) if use_title_col else str(samples.iloc[j][group_col])
                    ax.set_title(cell_title[:20], fontsize=7)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(str(val), fontsize=8, rotation=0, ha="right", va="center")
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()


# ## 5. Collection — Raw manifest
# Build one row per raw image: **path** (relative to **RES_ROOT**), run, split, class, format, **file_size**, **is_corrupt**; if readable: height, width, aspect_ratio, resolution, brightness, **blur** (Laplacian variance). If **raw_manifest_path** exists, load and skip; else iterate **raw_train_dir** and **raw_val_dir** via **list_image_files()**, compute blur with `cv2.Laplacian(gray).var()`, then merge **CleanVision** (Imagelab) issue columns per split/class via `Imagelab(data_path=class_path).find_issues()`, plus **near_duplicates_group** (nullable int per cluster from `lab.info["near_duplicates"]["sets"]`, unique ids across folders). Save to `res/<RUN_TAG>/meta/raw/image_manifest.csv`.

# In[9]:


# Output: res/<RUN_TAG>/meta/raw
if not os.path.isfile(raw_manifest_path):
    rows = []
    for root_dir, split in [(raw_train_dir, "train"), (raw_val_dir, "validate")]:
        for path, class_name in tqdm(list_image_files(root_dir), desc=f"Collect {split}", leave=False, file=sys.stdout):
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            if ext == "jpg":
                ext = "jpeg"
            file_size = os.path.getsize(path)
            rel_path = path_from_root(path)
            img = cv2.imread(path)
            if img is None:
                rows.append({
                    "path": rel_path,
                    "run": RUN_TAG,
                    "split": split,
                    "class": class_name,
                    "format": ext,
                    "file_size": file_size,
                    "is_corrupt": True,
                    "height": None,
                    "width": None,
                    "aspect_ratio": None,
                    "resolution": None,
                    "brightness": None,
                    "blur": None,
                })
            else:
                h, w = img.shape[0], img.shape[1]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                rows.append({
                    "path": rel_path,
                    "run": RUN_TAG,
                    "split": split,
                    "class": class_name,
                    "format": ext,
                    "file_size": file_size,
                    "is_corrupt": False,
                    "height": h,
                    "width": w,
                    "aspect_ratio": round(h / w, 4),
                    "resolution": h * w,
                    "brightness": int(img.mean()),
                    "blur": round(blur_var, 2),
                })
    df_manifest = pd.DataFrame(rows)
    # CleanVision (Imagelab): run per split/class, merge issue columns on path
    if Imagelab is not None:
        _near_dup_gid_state = {"next": 0}

        def run_imagelab_for_split(base_dir, split_name):
            """Run Imagelab once per split/class; return issues with path. Suppress lib logs/tqdm."""
            lab = Imagelab(data_path=base_dir)
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    lab.find_issues()
            issues = lab.issues.copy()
            issues["near_duplicates_group"] = pd.Series(pd.NA, index=issues.index, dtype="Int64")
            nd_sets = lab.info.get("near_duplicates", {}).get("sets", []) or []
            for group_indices in nd_sets:
                gid = _near_dup_gid_state["next"]
                _near_dup_gid_state["next"] += 1
                for idx in group_indices:
                    if idx in issues.index:
                        issues.loc[idx, "near_duplicates_group"] = gid

            def to_abs(p):
                p = str(p)
                return os.path.abspath(os.path.join(base_dir, p)) if not os.path.isabs(p) else p
            issues["path"] = issues.index.map(to_abs).map(path_from_root)
            return issues
        class_names = sorted(
            d for d in os.listdir(raw_train_dir)
            if os.path.isdir(os.path.join(raw_train_dir, d)) and d not in SKIP_NAMES
        )
        issues_list = []
        for root_dir, split in [(raw_train_dir, "train"), (raw_val_dir, "validate")]:
            for class_name in class_names:
                class_path = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                issues = run_imagelab_for_split(class_path, f"{split}/{class_name}")
                issues_list.append(issues)
        issues_all = pd.concat(issues_list, ignore_index=True)
        imagelab_cols = [c for c in issues_all.columns if c not in ("path", "class")]
        df_manifest = df_manifest.merge(issues_all[["path"] + imagelab_cols], on="path", how="left")
        if "near_duplicates_score" in df_manifest.columns and "near_duplicates_group" in df_manifest.columns:
            _cols = list(df_manifest.columns)
            _cols.remove("near_duplicates_group")
            _ins = _cols.index("near_duplicates_score") + 1
            _cols.insert(_ins, "near_duplicates_group")
            df_manifest = df_manifest[_cols]
        print(f"Merged {len(imagelab_cols)} CleanVision columns")
    df_manifest.to_csv(raw_manifest_path, index=False)
    print(f"Saved {len(df_manifest)} rows to {raw_manifest_path}")
else:
    df_manifest = pd.read_csv(raw_manifest_path)
    print(f"Loaded {len(df_manifest)} rows from {raw_manifest_path}")


# ## 6. EDA — Exploratory Data Analysis (raw)
# All EDA uses the **raw manifest**; corrupt images excluded where relevant. Sections below: class counts, format, empty/corrupt files, image size (**CleanVision** odd_size), aspect ratio (CleanVision odd_aspect_ratio), dark/light (CleanVision), near/exact duplicates (CleanVision), blur (**Laplacian** bins), low information (CleanVision), CleanVision summary, RGB/greyscale histograms, edge detection, retina preprocessing.

# ### 6.1 Class labels and counts
# Per-split class counts from **raw manifest**; summary table (data, split, class columns, total); bar + sunburst via **show_attr_charts**(df_train_chart, `"class"`). Train only (validate counts commented out).

# In[10]:


# Class distribution per split from manifest
classes = sorted(df_manifest["class"].unique())
train_counts = df_manifest[df_manifest["split"] == "train"]["class"].value_counts().to_dict()
val_counts = df_manifest[df_manifest["split"] == "validate"]["class"].value_counts().to_dict()

# Summary table: one row per split, columns = classes + total
df_counts = pd.DataFrame([
    {"data": "raw", "split": "train", **{c: train_counts.get(c, 0) for c in classes}, "total": sum(train_counts.values())},
    #{"data": "raw", "split": "validate", **{c: val_counts.get(c, 0) for c in classes}, "total": sum(val_counts.values())},
])
df_counts = df_counts[["data", "split"] + list(classes) + ["total"]]

# Per-class chart dataframes for each split
df_train_chart = pd.DataFrame({"class": classes, "total": [train_counts.get(c, 0) for c in classes]})
df_val_chart = pd.DataFrame({"class": classes, "total": [val_counts.get(c, 0) for c in classes]})

display(HTML("<strong>Train - Data Distribution</strong>"))
display_wide(df_counts)
show_attr_charts(df_train_chart, "class")
# display(HTML("<strong>Validate (raw holdout) - Data Distribution</strong>"))
# show_attr_charts(df_val_chart, "class")


# ### 6.2 File format and counts
# Format distribution per split: pivot (format × class counts), total column; **display_wide** + **show_attr_charts**(fmt_pivot, `"format"`). Surfaces jpeg/png/etc. mix per split.

# In[6]:


# Format distribution per split from manifest
classes = sorted(df_manifest["class"].unique())
all_formats = sorted(df_manifest["format"].unique())

for split_name, split_label in [("train", "Train")]:
    df_split = df_manifest[df_manifest["split"] == split_name]

    # Pivot: rows = format, columns = class, values = count
    fmt_pivot = df_split.groupby(["format", "class"]).size().unstack(fill_value=0).reset_index()
    for c in classes:
        if c not in fmt_pivot.columns:
            fmt_pivot[c] = 0
    fmt_pivot["total"] = fmt_pivot[classes].sum(axis=1)
    fmt_pivot["data"] = "raw"
    fmt_pivot["split"] = split_name
    fmt_pivot = fmt_pivot[["data", "split", "format"] + classes + ["total"]]

    display(HTML(f"<strong>{split_label} - Format Distribution</strong>"))
    display_wide(fmt_pivot)
    show_attr_charts(fmt_pivot, "format")


# ### 6.3 Empty files and counts
# Filter manifest where **file_size** == 0; per-split per-class counts; summary table; if any empty files: bar/sunburst and list of affected file **paths** (location) per split.

# In[7]:


# Filter manifest for empty (zero-byte) files
df_empty_all = df_manifest[df_manifest["file_size"] == 0]

# Per-split, per-class counts
empty_counts = {}
for split_name in ["train", "validate"]:
    df_split = df_empty_all[df_empty_all["split"] == split_name]
    empty_counts[split_name] = df_split.groupby("class").size().to_dict()

# Summary table (always displayed, even when all zeros)
df_empty = pd.DataFrame([
    {"data": "raw", "split": s, **{c: empty_counts[s].get(c, 0) for c in classes}, "total": sum(empty_counts[s].values())}
    for s in ["train", "validate"]
])
df_empty = df_empty[["data", "split"] + classes + ["total"]]
display(HTML("<strong>Empty Files Distribution</strong>"))
display_wide(df_empty)

# Charts and file locations (only when empty files exist)
total_empty = len(df_empty_all)
if total_empty == 0:
    display(HTML("<p>No empty files found - charts skipped.</p>"))
else:    
    for split_name, split_label in [("train", "Train")]:
        df_split = df_empty_all[df_empty_all["split"] == split_name]
        if len(df_split) == 0:
            continue
        # Per-class chart
        df_chart = pd.DataFrame({"class": classes, "total": [empty_counts[split_name].get(c, 0) for c in classes]})
        display(HTML(f"<strong>{split_label} Empty Files</strong>"))
        show_attr_charts(df_chart, "class")
        # Affected file locations
        df_files = df_split[["path", "class"]].reset_index(drop=True)
        df_files.insert(0, "sno", range(1, len(df_files) + 1))
        df_files.insert(1, "data", "raw")
        df_files.insert(2, "split", split_name)
        df_files = df_files.rename(columns={"path": "location"})
        display(HTML(f"<strong>{split_label} - Empty Files Locations</strong>"))
        display_wide(df_files, max_height=300)


# ### 6.4 Corrupt / unreadable files and counts
# Filter manifest where **is_corrupt** == True; per-split per-class counts; summary table; if any: bar/sunburst and list of affected **paths**. Surfaces images that failed to open (e.g. `cv2.imread` returned None).

# In[8]:


# Corrupt files from manifest (is_corrupt == True)
df_corrupt_all = df_manifest[df_manifest["is_corrupt"] == True]

# Per-split, per-class counts
corrupt_counts = {}
for split_name in ["train", "validate"]:
    df_split = df_corrupt_all[df_corrupt_all["split"] == split_name]
    corrupt_counts[split_name] = df_split.groupby("class").size().to_dict()

# Summary table (always displayed, even when all zeros)
df_corrupt = pd.DataFrame([
    {"data": "raw", "split": s, **{c: corrupt_counts[s].get(c, 0) for c in classes}, "total": sum(corrupt_counts[s].values())}
    for s in ["train", "validate"]
])
df_corrupt = df_corrupt[["data", "split"] + classes + ["total"]]
display(HTML("<strong>Corrupt Files Distribution</strong>"))
display_wide(df_corrupt)

# Charts and file locations (only when corrupt files exist)
total_corrupt = len(df_corrupt_all)
if total_corrupt == 0:
    display(HTML("<p>No corrupt files found - charts skipped.</p>"))
else:
    for split_name, split_label in [("train", "Train")]:
        df_split = df_corrupt_all[df_corrupt_all["split"] == split_name]
        if len(df_split) == 0:
            continue
        # Per-class chart
        df_chart = pd.DataFrame({"class": classes, "total": [corrupt_counts[split_name].get(c, 0) for c in classes]})
        display(HTML(f"<strong>{split_label} - Corrupt Files</strong>"))
        show_attr_charts(df_chart, "class")
        # Affected file locations
        df_files = df_split[["path", "class"]].reset_index(drop=True)
        df_files.insert(0, "sno", range(1, len(df_files) + 1))
        df_files.insert(1, "data", "raw")
        df_files.insert(2, "split", split_name)
        df_files = df_files.rename(columns={"path": "location"})
        display(HTML(f"<strong>{split_label} - Corrupt Files Locations</strong>"))
        display_wide(df_files, max_height=200)


# ### 6.5 Image size
# Use **CleanVision** **is_odd_size_issue** / **odd_size_score**. Add **odd_size_bin** (odd_size / normal); build per-split table (rows = bin, columns = classes + total); bar/sunburst; **odd_size_score** describe() and **show_class_distribution**; **show_attr_samples** by odd_size_bin. Corrupt excluded.

# In[36]:


# Odd size distribution per split from manifest (using Imagelab odd_size_score / is_odd_size_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
ODD_SIZE_ORDER = ["odd_size", "normal"]

def add_odd_size_bin(df):
    """Bin from Imagelab flag: odd_size / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["odd_size_bin"] = "normal"
    if "is_odd_size_issue" in out.columns:
        out.loc[out["is_odd_size_issue"] == True, "odd_size_bin"] = "odd_size"
    out["odd_size_bin"] = pd.Categorical(out["odd_size_bin"], categories=ODD_SIZE_ORDER, ordered=True)
    return out

def build_odd_size_table(df, split_name):
    """Pivot: rows = odd_size_bin (odd_size/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in ODD_SIZE_ORDER:
        df_bin = df[df["odd_size_bin"] == label]
        row = {"data": "raw", "split": split_name, "odd_size": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_odd_size_bin(df_valid[df_valid["split"] == split_name])
    df_odd_size_table = build_odd_size_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Image Size Distribution (CleanVision)</strong>"))
    display_wide(df_odd_size_table)
    show_attr_charts(df_odd_size_table, "odd_size")

    # odd_size_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Odd Size Score per Class</strong>"))
    display_wide(df_split.groupby("class")["odd_size_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "odd_size_score", "Odd Size Score", classes)

    # Sample images per odd size bin
    if not df_split.empty:
        show_attr_samples(df_split, "odd_size_bin", f"{split_label} - Odd Size", order=ODD_SIZE_ORDER)


# ### 6.6 Aspect ratio
# Use **CleanVision** **is_odd_aspect_ratio_issue** / **odd_aspect_ratio_score**. Add **odd_aspect_bin**; build table; bar/sunburst; score describe() and KDE per class; sample images per bin. Corrupt excluded.

# In[10]:


# Odd aspect ratio distribution per split from manifest (using Imagelab odd_aspect_ratio_score / is_odd_aspect_ratio_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
ODD_ASPECT_ORDER = ["odd_aspect_ratio", "normal"]

def add_odd_aspect_bin(df):
    """Bin from Imagelab flag: odd_aspect_ratio / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["odd_aspect_bin"] = "normal"
    if "is_odd_aspect_ratio_issue" in out.columns:
        out.loc[out["is_odd_aspect_ratio_issue"] == True, "odd_aspect_bin"] = "odd_aspect_ratio"
    out["odd_aspect_bin"] = pd.Categorical(out["odd_aspect_bin"], categories=ODD_ASPECT_ORDER, ordered=True)
    return out

def build_odd_aspect_table(df, split_name):
    """Pivot: rows = odd_aspect_bin (odd_aspect_ratio/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in ODD_ASPECT_ORDER:
        df_bin = df[df["odd_aspect_bin"] == label]
        row = {"data": "raw", "split": split_name, "odd_aspect_ratio": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_odd_aspect_bin(df_valid[df_valid["split"] == split_name])
    df_odd_aspect_table = build_odd_aspect_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Odd Aspect Ratio Distribution</strong>"))
    display_wide(df_odd_aspect_table)
    show_attr_charts(df_odd_aspect_table, "odd_aspect_ratio")

    # odd_aspect_ratio_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Odd Aspect Ratio Score per Class</strong>"))
    display_wide(df_split.groupby("class")["odd_aspect_ratio_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "odd_aspect_ratio_score", "Odd Aspect Score", classes)

    # Sample images per odd aspect bin
    if not df_split.empty:
        show_attr_samples(df_split, "odd_aspect_bin", f"{split_label} - Odd Aspect Ratio", order=ODD_ASPECT_ORDER)


# ### 6.7 Dark
# Use **CleanVision** **is_dark_issue** / **dark_score**. Add **dark_bin** (dark / normal); table; bar/sunburst; **dark_score** describe() and **show_class_distribution**; **show_attr_samples** per dark bin. For **underexposed** images.

# In[11]:


# Dark distribution per split from manifest (using Imagelab dark_score / is_dark_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
DARK_ORDER = ["dark", "normal"]

def add_dark_bin(df):
    """Bin from Imagelab flag: dark / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["dark_bin"] = "normal"
    if "is_dark_issue" in out.columns:
        out.loc[out["is_dark_issue"] == True, "dark_bin"] = "dark"
    out["dark_bin"] = pd.Categorical(out["dark_bin"], categories=DARK_ORDER, ordered=True)
    return out

def build_dark_table(df, split_name):
    """Pivot: rows = dark_bin (dark/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in DARK_ORDER:
        df_bin = df[df["dark_bin"] == label]
        row = {"data": "raw", "split": split_name, "dark": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_dark_bin(df_valid[df_valid["split"] == split_name])
    df_dark_table = build_dark_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Dark Distribution</strong>"))
    display_wide(df_dark_table)
    show_attr_charts(df_dark_table, "dark")

    # dark_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Dark Score per Class</strong>"))
    display_wide(df_split.groupby("class")["dark_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "dark_score", "0=darkest, 1=brightest", classes)

    # Sample images per dark bin
    if not df_split.empty:
        show_attr_samples(df_split, "dark_bin", f"{split_label} - Dark", order=DARK_ORDER)


# ### 6.8 Light
# Use **CleanVision** **is_light_issue** / **light_score**. Add **light_bin** (light / normal); table; bar/sunburst; **light_score** describe() and **show_class_distribution**; **show_attr_samples** per light bin. For **overexposed** images.

# In[12]:


# Light distribution per split from manifest (using Imagelab light_score / is_light_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
LIGHT_ORDER = ["light", "normal"]

def add_light_bin(df):
    """Bin from Imagelab flag: light / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["light_bin"] = "normal"
    if "is_light_issue" in out.columns:
        out.loc[out["is_light_issue"] == True, "light_bin"] = "light"
    out["light_bin"] = pd.Categorical(out["light_bin"], categories=LIGHT_ORDER, ordered=True)
    return out

def build_light_table(df, split_name):
    """Pivot: rows = light_bin (light/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in LIGHT_ORDER:
        df_bin = df[df["light_bin"] == label]
        row = {"data": "raw", "split": split_name, "light": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_light_bin(df_valid[df_valid["split"] == split_name])
    df_light_table = build_light_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Light Distribution</strong>"))
    display_wide(df_light_table)
    show_attr_charts(df_light_table, "light")

    # light_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Light Score per Class</strong>"))
    display_wide(df_split.groupby("class")["light_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "light_score", "0=brightest, 1=darkest", classes)

    # Sample images per light bin
    if not df_split.empty:
        show_attr_samples(df_split, "light_bin", f"{split_label} - Light", order=LIGHT_ORDER)


# ### 6.9 Near duplicates
# Use **CleanVision** **is_near_duplicates_issue** / **near_duplicates_score**. Add **near_dup_bin**; table; bar/sunburst; score describe() and KDE per class. Corrupt excluded.

# In[13]:


# Near duplicates distribution per split from manifest (using Imagelab near_duplicates_score / is_near_duplicates_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
NEAR_DUP_ORDER = ["near_duplicate", "normal"]

def add_near_dup_bin(df):
    """Bin from Imagelab flag: near_duplicate / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["near_dup_bin"] = "normal"
    if "is_near_duplicates_issue" in out.columns:
        out.loc[out["is_near_duplicates_issue"] == True, "near_dup_bin"] = "near_duplicate"
    out["near_dup_bin"] = pd.Categorical(out["near_dup_bin"], categories=NEAR_DUP_ORDER, ordered=True)
    return out

def build_near_dup_table(df, split_name):
    """Pivot: rows = near_dup_bin (near_duplicate/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in NEAR_DUP_ORDER:
        df_bin = df[df["near_dup_bin"] == label]
        row = {"data": "raw", "split": split_name, "near_duplicates": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_near_dup_bin(df_valid[df_valid["split"] == split_name])
    df_near_dup_table = build_near_dup_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Near Duplicates Distribution</strong>"))
    display_wide(df_near_dup_table)
    show_attr_charts(df_near_dup_table, "near_duplicates")

    # near_duplicates_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Near Duplicates Score per Class</strong>"))
    display_wide(df_split.groupby("class")["near_duplicates_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "near_duplicates_score", "0=most similar, 1=unique", classes)

    # Sample images: near duplicates
    #if not df_split.empty:
        #show_attr_samples(df_split, "near_dup_bin", f"{split_label} - Near Duplicates", order=NEAR_DUP_ORDER)


# ### 6.10 Exact duplicates
# Use **CleanVision** **is_exact_duplicates_issue** / **exact_duplicates_score**. Add **exact_dup_bin**; table; bar/sunburst; score describe() and KDE; **show_attr_samples** per bin. Corrupt excluded.

# In[14]:


# Exact duplicates distribution per split from manifest (using Imagelab exact_duplicates_score / is_exact_duplicates_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
EXACT_DUP_ORDER = ["exact_duplicate", "normal"]

def add_exact_dup_bin(df):
    """Bin from Imagelab flag: exact_duplicate / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["exact_dup_bin"] = "normal"
    if "is_exact_duplicates_issue" in out.columns:
        out.loc[out["is_exact_duplicates_issue"] == True, "exact_dup_bin"] = "exact_duplicate"
    out["exact_dup_bin"] = pd.Categorical(out["exact_dup_bin"], categories=EXACT_DUP_ORDER, ordered=True)
    return out

def build_exact_dup_table(df, split_name):
    """Pivot: rows = exact_dup_bin (exact_duplicate/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in EXACT_DUP_ORDER:
        df_bin = df[df["exact_dup_bin"] == label]
        row = {"data": "raw", "split": split_name, "exact_duplicates": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_exact_dup_bin(df_valid[df_valid["split"] == split_name])
    df_exact_dup_table = build_exact_dup_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Exact Duplicates Distribution</strong>"))
    display_wide(df_exact_dup_table)
    show_attr_charts(df_exact_dup_table, "exact_duplicates")

    # exact_duplicates_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Exact Duplicates Score per Class</strong>"))
    display_wide(df_split.groupby("class")["exact_duplicates_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "exact_duplicates_score", "Exact Dup. Score", classes)

    # Sample images: exact duplicates
    if not df_split.empty:
        show_attr_samples(df_split, "exact_dup_bin", f"{split_label} - Exact Duplicates", order=EXACT_DUP_ORDER)


# ### 6.11 Blur
# Blur from manifest column **blur** (**Laplacian** variance); higher = sharper. **add_blur_bin()** uses **get_blur_bin()** and **BLUR_ORDER** (5 bins). Table; bar/sunburst; **blur** describe() per class; **show_class_distribution**(blur); **show_attr_samples** per **blur_bin**. Corrupt excluded.

# In[15]:


# Blur (Laplacian variance) distribution per split from manifest
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()

def add_blur_bin(df):
    """Add blur_bin column using shared get_blur_bin() helper."""
    if df.empty:
        return df
    out = df.copy()
    out["blur_bin"] = out["blur"].apply(get_blur_bin)
    out["blur_bin"] = pd.Categorical(out["blur_bin"], categories=BLUR_ORDER, ordered=True)
    return out

def build_blur_table(df, split_name):
    """Pivot: rows = blur_bin, columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in BLUR_ORDER:
        df_bin = df[df["blur_bin"] == label]
        row = {"data": "raw", "split": split_name, "blur": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_blur_bin(df_valid[df_valid["split"] == split_name])
    df_blur_table = build_blur_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Blur Distribution (Laplacian Variance)</strong>"))
    display_wide(df_blur_table)
    show_attr_charts(df_blur_table, "blur")

    # Descriptive stats per class
    display(HTML(f"<strong>{split_label} - Laplacian Variance per Class</strong>"))
    display_wide(df_split.groupby("class")["blur"].describe().round(2).reset_index())

    # KDE bell curve per class
    show_class_distribution(df_split, "blur", "higher = sharper", classes)

    # Sample images per blur bin
    if not df_split.empty:
        show_attr_samples(df_split, "blur_bin", f"{split_label} - Blur (Laplacian)", order=BLUR_ORDER)


# ### 6.12 Low information
# Use **CleanVision** **is_low_information_issue** / **low_information_score**. Add **low_info_bin** (low_information / normal); table; bar/sunburst; score describe() and KDE; **show_attr_samples** per bin. **Low-information** images are those the library flags as having little usable signal. Corrupt excluded.

# In[16]:


# Low information distribution per split from manifest (using Imagelab low_information_score / is_low_information_issue)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
LOW_INFO_ORDER = ["low_information", "normal"]

def add_low_info_bin(df):
    """Bin from Imagelab flag: low_information / normal."""
    if df.empty:
        return df
    out = df.copy()
    out["low_info_bin"] = "normal"
    if "is_low_information_issue" in out.columns:
        out.loc[out["is_low_information_issue"] == True, "low_info_bin"] = "low_information"
    out["low_info_bin"] = pd.Categorical(out["low_info_bin"], categories=LOW_INFO_ORDER, ordered=True)
    return out

def build_low_info_table(df, split_name):
    """Pivot: rows = low_info_bin (low_information/normal), columns = classes + total."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for label in LOW_INFO_ORDER:
        df_bin = df[df["low_info_bin"] == label]
        row = {"data": "raw", "split": split_name, "low_information": label}
        for cls in classes:
            row[cls] = len(df_bin[df_bin["class"] == cls])
        row["total"] = len(df_bin)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = add_low_info_bin(df_valid[df_valid["split"] == split_name])
    df_low_info_table = build_low_info_table(df_split, split_name)

    # Per-bin count table + bar/sunburst/deviation
    display(HTML(f"<strong>{split_label} - Low Information Distribution</strong>"))
    display_wide(df_low_info_table)
    show_attr_charts(df_low_info_table, "low_information")

    # low_information_score stats + bell curve per class
    display(HTML(f"<strong>{split_label} - Low Information Score per Class</strong>"))
    display_wide(df_split.groupby("class")["low_information_score"].describe().round(4).reset_index())
    show_class_distribution(df_split, "low_information_score", "Low Info Score", classes)

    # Sample images per low info bin
    if not df_split.empty:
        show_attr_samples(df_split, "low_info_bin", f"{split_label} - Low Information", order=LOW_INFO_ORDER)


# ### 6.13 CleanVision issues summary
# Consolidated view: **CV_ISSUE_COLS** = dark, light, blurry, low_information, odd_aspect_ratio, odd_size, near_duplicates, exact_duplicates. (1) Table: one row per issue type, columns = classes + total. (2) Per-class table: one row per class, columns = issue counts + **any_issue**. Bar/sunburst for **issue_type** and for **any_issue** per class.

# In[17]:


# Consolidated CleanVision issues summary per class
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
CV_ISSUE_COLS = [
    ("dark", "is_dark_issue"),
    ("light", "is_light_issue"),
    ("blurry", "is_blurry_issue"),
    ("low_information", "is_low_information_issue"),
    ("odd_aspect_ratio", "is_odd_aspect_ratio_issue"),
    ("odd_size", "is_odd_size_issue"),
    ("near_duplicates", "is_near_duplicates_issue"),
    ("exact_duplicates", "is_exact_duplicates_issue"),
]

def build_cv_issues_summary(df, split_name):
    """One row per issue type, columns = classes + total. Shows count of flagged images."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for issue_label, issue_col in CV_ISSUE_COLS:
        if issue_col not in df.columns:
            continue
        df_flagged = df[df[issue_col] == True]
        row = {"data": "raw", "split": split_name, "issue_type": issue_label}
        for cls in classes:
            row[cls] = len(df_flagged[df_flagged["class"] == cls])
        row["total"] = len(df_flagged)
        rows.append(row)
    return pd.DataFrame(rows)

for split_name, split_label in [("train", "Train")]:
    df_split = df_valid[df_valid["split"] == split_name]
    df_cv_summary = build_cv_issues_summary(df_split, split_name)

    # Summary table
    display(HTML(f"<strong>{split_label} - CleanVision Issues Summary</strong>"))
    display_wide(df_cv_summary)
    show_attr_charts(df_cv_summary, "issue_type")

    # Per-class breakdown: one row per class, columns = issue types
    rows_per_class = []
    for cls in classes:
        df_cls = df_split[df_split["class"] == cls]
        row = {"data": "raw", "split": split_name, "class": cls, "images": len(df_cls)}
        for issue_label, issue_col in CV_ISSUE_COLS:
            if issue_col not in df_cls.columns:
                continue
            row[issue_label] = int(df_cls[issue_col].sum())
        row["any_issue"] = int(df_cls[[col for _, col in CV_ISSUE_COLS if col in df_cls.columns]].any(axis=1).sum())
        rows_per_class.append(row)
    df_per_class = pd.DataFrame(rows_per_class)

    display(HTML(f"<strong>{split_label} - Issues per Class</strong>"))
    display_wide(df_per_class)

    # Per-class chart: how many images in each class have any issue
    df_any_issue = pd.DataFrame({
        "class": classes,
        "total": [r["any_issue"] for r in rows_per_class],
    })
    display(HTML(f"<strong>{split_label} - Images with Any Issue per Class</strong>"))
    show_attr_charts(df_any_issue, "class")


# ### 6.14 Class-wise RGB histogram
# Per-class accumulated R,G,B histograms (bins 0–255) over all **train** images: load image, `cv2.split(RGB)`, `np.histogram` per channel, add to class accumulators. Cache at `meta_raw_dir/rgb_hist_cache.pkl`. Plot: proportion per channel, bins 0–(**DARK_THRESH**-1) excluded; one subplot per class, overlay R/G/B bars.

# In[18]:


DARK_THRESH = 10  # exclude bins 0–9 (black border + near-black gradient pixels)
bins = np.arange(257)  # 0..256 edges → 256 bins
x_plot = np.arange(DARK_THRESH, 256)  # skip bins 0–(DARK_THRESH-1)
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()

if os.path.isfile(rgb_hist_cache_path):
    with open(rgb_hist_cache_path, "rb") as _f:
        hist_data = _pickle.load(_f)
    print(f"RGB histogram cache loaded — {rgb_hist_cache_path}")
else:
    hist_data = {}
    class_pbar = tqdm(classes, total=len(classes), desc="RGB histogram (computing)", file=sys.stdout)
    for cls in class_pbar:
        class_pbar.set_postfix(cls=cls)
        df_cls = df_valid[(df_valid["split"] == "train") & (df_valid["class"] == cls)]
        r_acc = np.zeros(256, dtype=np.float64)
        g_acc = np.zeros(256, dtype=np.float64)
        b_acc = np.zeros(256, dtype=np.float64)
        count = 0
        for _, row in tqdm(df_cls.iterrows(), total=len(df_cls), desc=f"  [{cls}] {len(df_cls)} imgs", leave=False, file=sys.stdout):
            full_path = os.path.join(RES_ROOT, row["path"].replace("/", os.sep))
            img = cv2.imread(full_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            r, g, b = cv2.split(img_rgb)
            r_acc += np.histogram(r, bins=bins)[0]
            g_acc += np.histogram(g, bins=bins)[0]
            b_acc += np.histogram(b, bins=bins)[0]
            count += 1
        if count == 0:
            continue
        hist_data[cls] = {"R": r_acc, "G": g_acc, "B": b_acc}
    # Persist to pickle
    with open(rgb_hist_cache_path, "wb") as _f:
        _pickle.dump(hist_data, _f)
    print(f"RGB histogram cache saved — {rgb_hist_cache_path}")

n = len(classes)
fig = make_subplots(
    rows=1, cols=n, subplot_titles=classes,
    horizontal_spacing=0.06, shared_yaxes=True,
)
for i, cls in enumerate(classes):
    if cls not in hist_data:
        continue
    r_acc = hist_data[cls]["R"]
    g_acc = hist_data[cls]["G"]
    b_acc = hist_data[cls]["B"]
    # Normalise to proportion (sum to 1), then drop bins 0–(DARK_THRESH-1)
    r_prop = (r_acc / r_acc.sum())[DARK_THRESH:]
    g_prop = (g_acc / g_acc.sum())[DARK_THRESH:]
    b_prop = (b_acc / b_acc.sum())[DARK_THRESH:]
    fig.add_trace(go.Bar(x=x_plot, y=r_prop, name="R", marker_color="rgba(239, 85, 59, 0.5)", marker_line_width=0, showlegend=(i == 0)), row=1, col=i+1)
    fig.add_trace(go.Bar(x=x_plot, y=g_prop, name="G", marker_color="rgba(0, 204, 150, 0.5)", marker_line_width=0, showlegend=(i == 0)), row=1, col=i+1)
    fig.add_trace(go.Bar(x=x_plot, y=b_prop, name="B", marker_color="rgba(99, 110, 250, 0.5)", marker_line_width=0, showlegend=(i == 0)), row=1, col=i+1)
    fig.update_xaxes(title_text=f"Channel value ({DARK_THRESH}–255)", range=[DARK_THRESH, 255], row=1, col=i+1)
fig.update_yaxes(title_text="Proportion of pixels", row=1, col=1)
fig.update_layout(
    barmode="overlay",
    bargap=0,
    height=420,
    autosize=True,
    margin=dict(t=50, b=80, l=50, r=30),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),    
    hoverlabel=dict(namelength=0),
)
fig.show()
print(f"RGB histogram done — channel value proportion, channel values 0–{DARK_THRESH-1} excluded.")


# ### 6.15 Class-wise Greyscale histogram
# Per-class accumulated grayscale histogram (bins 0–255) over **train** images: `cv2.COLOR_BGR2GRAY`, `np.histogram`, add to accumulator. Cache at `meta_raw_dir/gray_hist_cache.pkl`. Plot proportion, exclude bins 0–(**DARK_THRESH**-1); one subplot per class.

# In[19]:


DARK_THRESH = 10  # exclude bins 0–9 (black border + near-black gradient pixels)
gray_x_plot = np.arange(DARK_THRESH, 256) 
df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
gray_hist_cache_path = os.path.join(meta_raw_dir, "gray_hist_cache.pkl")  # meta_raw_dir set above

if os.path.isfile(gray_hist_cache_path):
    with open(gray_hist_cache_path, "rb") as _f:
        gray_hist_data = _pickle.load(_f)
    print(f"Grayscale histogram cache loaded — {gray_hist_cache_path}")
else:
    gray_hist_data = {}
    class_pbar = tqdm(classes, total=len(classes), desc="Gray histogram (computing)", file=sys.stdout)
    for cls in class_pbar:
        class_pbar.set_postfix(cls=cls)
        df_cls = df_valid[(df_valid["split"] == "train") & (df_valid["class"] == cls)]
        g_acc = np.zeros(256, dtype=np.float64)
        count = 0
        for _, row in tqdm(df_cls.iterrows(), total=len(df_cls), desc=f"  [{cls}] {len(df_cls)} imgs", leave=False, file=sys.stdout):
            full_path = os.path.join(RES_ROOT, row["path"].replace("/", os.sep))
            img = cv2.imread(full_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g_acc += np.histogram(gray, bins=np.arange(257))[0]
            count += 1
        if count == 0:
            continue
        gray_hist_data[cls] = g_acc
    with open(gray_hist_cache_path, "wb") as _f:
        _pickle.dump(gray_hist_data, _f)
    print(f"Grayscale histogram cache saved — {gray_hist_cache_path}")

n = len(classes)
fig = make_subplots(
    rows=1, cols=n, subplot_titles=classes,
    horizontal_spacing=0.06, shared_yaxes=True,
)
for i, cls in enumerate(classes):
    if cls not in gray_hist_data:
        continue
    g_acc = gray_hist_data[cls]
    g_prop = (g_acc / g_acc.sum())[DARK_THRESH:]
    fig.add_trace(
        go.Bar(x=gray_x_plot, y=g_prop, name=cls,
               marker_color="#808080", marker_line_width=0, showlegend=True),
        row=1, col=i+1,
    )
    fig.update_xaxes(title_text=f"Pixel intensity ({DARK_THRESH}–255)", range=[DARK_THRESH, 255], row=1, col=i+1)
fig.update_yaxes(title_text="Proportion of pixels", row=1, col=1)
fig.update_layout(
    bargap=0,
    height=420,
    autosize=True,
    margin=dict(t=50, b=80, l=50, r=30),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),    
    hoverlabel=dict(namelength=0),
)
fig.show()
print(f"Grayscale histogram done — normalised proportion, bins 0–{DARK_THRESH-1} excluded.")


# ### 6.16 Edge detection
# Per **blur** bin and class: one sample. **plot_retina_edges()**: **CLAHE** on each channel, `GaussianBlur(3,3)`, `cv2.Canny(30,100)`. Panels: Original RGB | Canny Green | Canny Blue | Canny Red | Canny Gray. **Green channel** emphasizes vessel detail in fundus images.

# In[20]:


df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
def plot_retina_edges(image_path, klass, blur_val=None):
    """
    Original RGB + Canny edges on all channels (Green, Blue, Red, Grayscale) — CLAHE enhanced.
    Panels: Original RGB | Canny Green | Canny Blue | Canny Red | Canny Gray
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Could not read image", ha="center", va="center", fontsize=12)
        ax.set_title(klass); ax.axis("off"); plt.show()
        return

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _canny(ch):
        enhanced = clahe.apply(ch)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return cv2.Canny(blurred, 30, 100)

    edges_g = _canny(image_bgr[:, :, 1])
    edges_b = _canny(image_bgr[:, :, 0])
    edges_r = _canny(image_bgr[:, :, 2])
    edges_gray = _canny(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))

    blur_info = f"  |  blur={blur_val:.1f}" if blur_val is not None else ""
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.5))
    fig.patch.set_facecolor("#F3F4F6")
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original RGB\n{klass}")
    axes[1].imshow(edges_g, cmap="Greens")
    axes[1].set_title("Canny — Green ch.")
    axes[2].imshow(edges_b, cmap="Blues")
    axes[2].set_title("Canny — Blue ch.")
    axes[3].imshow(edges_r, cmap="Reds")
    axes[3].set_title("Canny — Red ch.")
    axes[4].imshow(edges_gray, cmap="gray")
    axes[4].set_title("Canny — Gray")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"{klass}{blur_info}", fontsize=9)
    plt.tight_layout()
    plt.show()

df_train_valid_edges = df_valid[df_valid["split"] == "train"].copy()
for bin_label, lo, hi in BLUR_BINS:
    df_bin = df_train_valid_edges[
        (df_train_valid_edges["blur"] >= lo) & (df_train_valid_edges["blur"] < hi)
    ]
    if df_bin.empty:
        continue
    display(HTML(f"<strong>Canny edges — blur bin: {bin_label}</strong>"))
    for cls in classes:
        df_cls = df_bin[df_bin["class"] == cls]
        if df_cls.empty:
            continue
        sample_row = df_cls.iloc[0]
        full_path = os.path.join(RES_ROOT, sample_row["path"].replace("/", os.sep))
        plot_retina_edges(full_path, cls, blur_val=sample_row["blur"])


# ### 6.17 Retina preprocessing
# **preprocess_retina_image()**: per-channel **CLAHE** → **GaussianBlur** → threshold mask → largest contour **ROI** crop → resize to **target_size** (224,224). Returns dict `{bgr_orig, green, blue, red, gray}`. Per **blur** bin and class: one sample showing Original RGB | Green | Blue | Red | Grayscale (all CLAHE+crop).

# In[21]:


df_valid = df_manifest[df_manifest["is_corrupt"] == False].copy()
# Blur bins reuse the Laplacian ranges defined earlier in this file.
def preprocess_retina_image(image_path, target_size=(224, 224)):
    """
    Per-channel CLAHE → Gaussian blur → ROI crop (largest contour) → resize.
    Returns dict {bgr_orig, green, blue, red, gray} (all np.ndarray), or None on read failure.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    def _process_channel(ch):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(ch)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped = blurred[y:y+h, x:x+w]
        else:
            cropped = blurred
        return cv2.resize(cropped, target_size)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "bgr_orig": img_bgr,
        "green":    _process_channel(img_bgr[:, :, 1]),
        "blue":     _process_channel(img_bgr[:, :, 0]),
        "red":      _process_channel(img_bgr[:, :, 2]),
        "gray":     _process_channel(gray),
    }
df_train_valid = df_valid[df_valid["split"] == "train"].copy()
for bin_label, lo, hi in BLUR_BINS:
    df_bin = df_train_valid[(df_train_valid["blur"] >= lo) & (df_train_valid["blur"] < hi)]
    if df_bin.empty:
        continue
    display(HTML(f"<strong>Brightness bin: {bin_label}</strong>"))
    for cls in classes:
        df_cls = df_bin[df_bin["class"] == cls]
        if df_cls.empty:
            continue
        sample_row = df_cls.iloc[0]
        full_path = os.path.join(RES_ROOT, sample_row["path"].replace("/", os.sep))
        result = preprocess_retina_image(full_path)
        if result is None:
            continue
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
        fig.patch.set_facecolor("#F3F4F6")
        axes[0].imshow(cv2.cvtColor(result["bgr_orig"], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original RGB\n{cls}")
        axes[1].imshow(result["green"], cmap="Greens_r")
        axes[1].set_title("Green (CLAHE+crop)")
        axes[2].imshow(result["blue"], cmap="Blues_r")
        axes[2].set_title("Blue (CLAHE+crop)")
        axes[3].imshow(result["red"], cmap="Reds_r")
        axes[3].set_title("Red (CLAHE+crop)")
        axes[4].imshow(result["gray"], cmap="gray")
        axes[4].set_title("Grayscale (CLAHE+crop)")
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"{bin_label} — {cls}  |  blur={sample_row['blur']:.1f}", fontsize=9)
        plt.tight_layout()
        plt.show()


# ## 7. Preprocess (prep)
# Copy **raw manifest** to **prep manifest**; add columns **has_odd_size**, **has_dark**, **has_blur**, **has_exact_dup** (from ImageLab / Laplacian bins), **issues** (comma-separated), **prep_path**, **prep_status**. Only **train** non-corrupt rows can have issues. Output: `res/<RUN_TAG>/meta/prep/image_manifest.csv`. Then run **Step 1–4**; each step updates **prep_path** for affected rows and writes outputs under `res/<RUN_TAG>/data/prep/`.

# ### 7.1 Preprocess manifest
# **BLUR_CLEAN_BINS** = very blurry (0–50) and blurry (50–150). Copy **df_manifest** to **df_prep**; set **has_odd_size**, **has_dark**, **has_blur**, **has_exact_dup** from **is_odd_size_issue**, **is_dark_issue**, **is_exact_duplicates_issue** and **get_blur_bin()**; only **train** non-corrupt get issues. Build **issues** string; **prep_path** = `""`, **prep_status** = `""`. Save to **prep_manifest_path**; print issue summary (**total_train**, **has_odd_size**, **has_dark**, **has_blur**, **has_exact_dup**, **multi_issue**, **no_issue**); **show_attr_charts** for **issue_type**.

# In[61]:


# Blur bins flagged for cleaning: 0–50 (very blurry) and 50–150 (blurry).
BLUR_CLEAN_BINS = {"0–50 (very blurry)", "50–150 (blurry)"}  # bins that require enhancement

# Copy raw manifest and add lifted columns: has_*, issues, prep_path (blank if none)
df_prep = df_manifest.copy()

# has_odd_size, has_dark, has_exact_dup: from ImageLab columns; has_blur from Laplacian bins
is_train_ok = (df_prep["split"] == "train") & (df_prep["is_corrupt"] == False)
for col, src in [("has_odd_size", "is_odd_size_issue"), ("has_dark", "is_dark_issue"), ("has_exact_dup", "is_exact_duplicates_issue")]:
    df_prep[col] = df_prep[src].fillna(False).astype(bool) if src in df_prep.columns else False
df_prep["has_blur"] = df_prep["blur"].apply(
    lambda v: get_blur_bin(v) in BLUR_CLEAN_BINS if pd.notna(v) else False
)
# Only train non-corrupt rows can have issues; validate (raw holdout) / corrupt stay False
df_prep.loc[~is_train_ok, ["has_odd_size", "has_dark", "has_blur", "has_exact_dup"]] = False

def _build_issues(row):
    issues = []
    if row["has_odd_size"]:  issues.append("odd_size")
    if row["has_dark"]:      issues.append("dark")
    if row["has_blur"]:      issues.append("blur")
    if row["has_exact_dup"]: issues.append("exact_dup")
    return ",".join(issues) if issues else ""

df_prep["issues"] = df_prep.apply(_build_issues, axis=1)
# prep_path: blank until a step writes an output path; updated on each run
df_prep["prep_path"] = ""
df_prep["prep_status"] = ""  # "" = included, "removed" = excluded (e.g. exact dup)

# Save to prep manifest (res/<run_tag>/meta/prep/image_manifest.csv); overwrites if exists
_prep_path = os.path.abspath(prep_manifest_path)
df_prep.to_csv(_prep_path, index=False)
df_train_prep = df_prep[(df_prep["split"] == "train") & (df_prep["is_corrupt"] == False)]
n_flagged = (df_train_prep["issues"] != "").sum()
print(f"Saved to {prep_manifest_path} — {len(df_prep)} rows ({len(df_train_prep)} train), {n_flagged} with issues")

summary = {
    "total_train":   len(df_train_prep),
    "has_odd_size":  int(df_train_prep["has_odd_size"].sum()),
    "has_dark":      int(df_train_prep["has_dark"].sum()),
    "has_blur":      int(df_train_prep["has_blur"].sum()),
    "has_exact_dup": int(df_train_prep["has_exact_dup"].sum()),
    "multi_issue":   int((df_train_prep[["has_odd_size", "has_dark", "has_blur", "has_exact_dup"]].sum(axis=1) > 1).sum()),
    "no_issue":      int((df_train_prep["issues"] == "").sum()),
}
display(HTML("<strong>Prep Manifest — Issue Summary (train)</strong>"))
display_wide(pd.DataFrame([summary]))

_issue_flags = [("odd_size", "has_odd_size"), ("dark", "has_dark"), ("blur", "has_blur"), ("exact_dup", "has_exact_dup")]
_issue_rows = []
for issue_label, flag_col in _issue_flags:
    row = {"issue_type": issue_label}
    for cls in classes:
        row[cls] = int(df_train_prep[df_train_prep["class"] == cls][flag_col].sum())
    row["total"] = int(df_train_prep[flag_col].sum())
    _issue_rows.append(row)
df_issue_chart = pd.DataFrame(_issue_rows)
show_attr_charts(df_issue_chart, "issue_type")


# ### 7.2 Step 1 — Exact duplicate removal
# Among rows with **has_exact_dup**: group by (split, class, blur), **keep one** (head(1)), set **prep_status** = `"removed"` for the rest. Save **prep manifest**; display summary (flagged, removed, active).

# In[23]:


# Keep one per (split, class, blur); mark the rest prep_status='removed'.
df_cm = pd.read_csv(prep_manifest_path)
if "prep_status" not in df_cm.columns:
    df_cm["prep_status"] = ""
df_cm["prep_status"] = df_cm["prep_status"].astype(object).fillna("")  # ensure string dtype

dup = df_cm[df_cm["has_exact_dup"]].sort_values("path")
keep_idx = dup.groupby(["split", "class", "blur"], sort=False).head(1).index
remove_idx = dup.index.difference(keep_idx)
df_cm.loc[remove_idx, "prep_status"] = "removed"

df_cm.to_csv(prep_manifest_path, index=False)
print(f"Step 1 — exact_dup: {len(remove_idx)} removed, {len(keep_idx)} kept")

summary = {
    "step": "1 — exact_dup",
    "flagged": int(df_cm["has_exact_dup"].sum()),
    "removed": int((df_cm["prep_status"] == "removed").sum()),
    "active":  int((df_cm["prep_status"] != "removed").sum()),
}
display(HTML("<strong>Step 1 — Exact Duplicate Removal</strong>"))
display_wide(pd.DataFrame([summary]))


# ### 7.3 Step 2 — Odd size correction
# **Target size** = mode of (width×height) of **train** non-corrupt non–odd_size images from **raw manifest** (else 512×512). **_crop_black_then_resize()**: threshold gray at 40, find non-zero bounding rect, crop, resize to target (**INTER_LANCZOS4**). For each row with **has_odd_size** and **prep_status** != `"removed"`: read from **prep_path** or **path**, write to `res/<RUN_TAG>/data/prep/step2/<class>/<fname>`, set **prep_path**. Before/After sample grid and comparison figure.

# In[62]:


def _input_path(row):
    p = str(row.get("prep_path") or "").strip()
    return p if p else row["path"]

def _crop_black_then_resize(img, target_w, target_h, black_thresh=40, interpolation=cv2.INTER_LANCZOS4):
    """Crop away black/dark borders (pixels below black_thresh), then resize to (target_w, target_h)."""
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    # Pixels above thresh = content; below = black border. Fundus borders are often 25–40, so 40 trims them.
    _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) == 0:
        return cv2.resize(img, (target_w, target_h), interpolation=interpolation)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y : y + h, x : x + w]
    return cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)

df_cm = pd.read_csv(prep_manifest_path)
df_cm["prep_path"] = df_cm["prep_path"].astype(object).fillna("")
df_cm["prep_status"] = df_cm["prep_status"].astype(object).fillna("")

# Target size: mode WxH of non-odd-size train images
_normal = df_manifest[(df_manifest["split"] == "train") & (df_manifest["is_corrupt"] == False)]
if "is_odd_size_issue" in df_manifest.columns:
    _normal = _normal[_normal["is_odd_size_issue"] != True]
if not _normal.empty and "width" in _normal.columns:
    _mode = (_normal["width"].astype(str) + "x" + _normal["height"].astype(str)).mode()[0]
    target_w, target_h = (int(x) for x in _mode.split("x"))
else:
    target_w, target_h = 512, 512
print(f"Step 2 — target size: {target_w}x{target_h}")

step_dir = os.path.join(prep_dir, "step2")
step_rel = os.path.relpath(step_dir, RES_ROOT).replace("\\", "/")
active = df_cm[df_cm["has_odd_size"] & (df_cm["prep_status"] != "removed")]
df_cm["_before_path"] = df_cm.apply(_input_path, axis=1)

done = 0
for idx, row in tqdm(active.iterrows(), total=len(active), desc="Step 2 odd_size", leave=False, file=sys.stdout):
    src_rel = _input_path(row)
    src = os.path.join(RES_ROOT, src_rel.replace("/", os.sep))
    img = cv2.imread(src)
    if img is None:
        continue
    out_dir = os.path.join(step_dir, row["class"])
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(src_rel)
    out_img = _crop_black_then_resize(img, target_w, target_h)
    cv2.imwrite(os.path.join(out_dir, fname), out_img)
    df_cm.at[idx, "prep_path"] = f"{step_rel}/{row['class']}/{fname}"
    done += 1

display(HTML("<strong>Step 2 — Odd Size Correction</strong>"))
display_wide(pd.DataFrame([{"stage": "Before", "flagged": len(active), "processed": 0, "target_size": f"{target_w}x{target_h}"}, {"stage": "After", "flagged": len(active), "processed": done, "target_size": f"{target_w}x{target_h}"}]))
display(HTML(f"<p><em>Before</em> = original. <em>After</em> = black borders cropped, then resized to {target_w}×{target_h}.</p>"))

# Before/After samples: Before = original raw image (path); After = Step 2 output (prep_path)
n_show = min(5, done)
if n_show > 0:
    sample_idx = list(active.index)[:n_show]
    records = []
    for idx in sample_idx:
        r = df_cm.loc[idx]
        before_sz = f"{int(r['width'])}x{int(r['height'])}" if pd.notna(r.get("width")) and pd.notna(r.get("height")) else "original"
        records.append({"stage": "Before", "path": r["path"], "cell_title": f"{r['class']} — {before_sz}"})
    for idx in sample_idx:
        r = df_cm.loc[idx]
        records.append({"stage": "After", "path": r["prep_path"], "cell_title": f"{r['class']} — {target_w}x{target_h}"})
    show_attr_samples(pd.DataFrame(records), "stage", "Step 2 — Before vs After (size in title)", n_cols=n_show, order=["Before", "After"], title_col="cell_title")

df_cm.drop(columns=["_before_path"], inplace=True)
df_cm.to_csv(prep_manifest_path, index=False)
print(f"Step 2 — odd_size: {done} resized, prep_path updated for {done} rows")


# ### 7.4 Step 3 — Blur enhancement
# **Unsharp mask** only inside eyeball: **_eyeball_mask()** (threshold, erode, feather); **_unsharp_masked()** = unsharp inside mask, original outside. **_unsharp_mask()**: GaussianBlur + addWeighted. For each **has_blur**, **prep_status** != `"removed"`: read image, write **_unsharp_masked**(img) to `res/<RUN_TAG>/data/prep/step3/<class>/<fname>`, set **prep_path**. Optional **_blurry_score_for_image()** for before/after (CleanVision or Laplacian fallback). Before/After samples and 2-row comparison figure.

# In[63]:


def _input_path(row):
    p = str(row.get("prep_path") or "").strip()
    return p if p else row["path"]

def _unsharp_mask(img, sigma=1.2, strength=1.25):
    """Unsharp mask: GaussianBlur + addWeighted (OpenCV). Enhances edges with moderate strength to avoid amplifying noise (white dots)."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, strength, blurred, 1 - strength, 0)

def _eyeball_mask(gray, thresh=30, erode_px=15, feather_px=20):
    """Mask for fundus (eyeball) region: gray > thresh, erode well inside to avoid boundary, then feather for smooth blend."""
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        mask = cv2.erode(mask, kernel)
    if feather_px > 0:
        mask = cv2.GaussianBlur(mask, (feather_px * 2 + 1, feather_px * 2 + 1), 0)
    return mask

def _unsharp_masked(img, sigma=1.2, strength=1.25, mask_thresh=30, mask_erode=15, mask_feather=20):
    """Simple blur removal: unsharp only inside eyeball; boundary and background stay original to avoid white border."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _eyeball_mask(gray, thresh=mask_thresh, erode_px=mask_erode, feather_px=mask_feather)
    enhanced = _unsharp_mask(img, sigma=sigma, strength=strength)
    m = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    out = (enhanced.astype(np.float32) * m + img.astype(np.float32) * (1 - m)).astype(np.uint8)
    return out

def _blurry_score_for_image(img_bgr, normalizing_factor=1.0):
    """Compute CleanVision-style blurry_score (0–1, higher = sharper) for one image. Falls back to Laplacian-based score if cleanvision unavailable."""
    try:
        from PIL import Image
        from cleanvision.issue_managers.image_property import BlurrinessProperty
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        prop = BlurrinessProperty()
        raw = prop.calculate(pil_img)
        raw_blur = raw.get("blurriness")
        if raw_blur is not None:
            return float(np.clip(1 - np.exp(-raw_blur * normalizing_factor), 0, 1))
    except Exception:
        pass
    # Fallback: Laplacian variance → 0–1 (higher = sharper), comparable scale
    if img_bgr is None:
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(np.clip(1 - np.exp(-lap_var / 100.0), 0, 1))
    except Exception:
        return None

df_cm = pd.read_csv(prep_manifest_path)
df_cm["prep_path"] = df_cm["prep_path"].astype(object).fillna("")
df_cm["prep_status"] = df_cm["prep_status"].astype(object).fillna("")

step_dir = os.path.join(prep_dir, "step3")
step_rel = os.path.relpath(step_dir, RES_ROOT).replace("\\", "/")
active = df_cm[df_cm["has_blur"] & (df_cm["prep_status"] != "removed")]
df_cm["_before_path"] = df_cm.apply(_input_path, axis=1)

done = 0
for idx, row in tqdm(active.iterrows(), total=len(active), desc="Step 3 blur", file=sys.stdout):
    src_rel = _input_path(row)
    src = os.path.join(RES_ROOT, src_rel.replace("/", os.sep))
    img = cv2.imread(src)
    if img is None:
        continue
    out_dir = os.path.join(step_dir, row["class"])
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(src_rel)
    cv2.imwrite(os.path.join(out_dir, fname), _unsharp_masked(img))
    df_cm.at[idx, "prep_path"] = f"{step_rel}/{row['class']}/{fname}"
    done += 1

display(HTML("<strong>Step 3 — Blur Enhancement (Unsharp mask / OpenCV)</strong>"))
display_wide(pd.DataFrame([{"stage": "Before", "flagged": len(active), "processed": 0}, {"stage": "After", "flagged": len(active), "processed": done}]))

# Before/After samples: score first in title so it's visible (show_attr_samples truncates to 20 chars)
n_show = min(5, done)
if n_show > 0:
    sample_idx = list(active.index)[:n_show]
    records = []
    for idx in sample_idx:
        r = df_cm.loc[idx]
        bs_before = r.get("blurry_score")
        if pd.notna(bs_before) and isinstance(bs_before, (int, float)):
            cell_title = f"blurry={bs_before:.2f} {r['class']}"
        else:
            cell_title = str(r["class"])
        records.append({"stage": "Before", "path": r["path"], "cell_title": cell_title})
    for idx in sample_idx:
        r = df_cm.loc[idx]
        prep_full = os.path.join(RES_ROOT, str(r["prep_path"]).replace("/", os.sep))
        img_after = cv2.imread(prep_full) if str(r.get("prep_path") or "").strip() else None
        bs_after = _blurry_score_for_image(img_after) if img_after is not None else None
        if bs_after is not None:
            cell_title = f"blurry={bs_after:.2f} {r['class']}"
        else:
            cell_title = f"unsharp {r['class']}"
        records.append({"stage": "After", "path": r["prep_path"], "cell_title": cell_title})
    show_attr_samples(pd.DataFrame(records), "stage", "Step 3 — Before vs After (Unsharp mask; blurry_score)", n_cols=n_show, order=["Before", "After"], title_col="cell_title")

# Three samples: Before on top row, respective After on bottom row — big size
n_compare = min(3, done)
if n_compare > 0:
    fig2, axes = plt.subplots(2, n_compare, figsize=(5 * n_compare, 10))
    fig2.patch.set_facecolor("#F3F4F6")
    if n_compare == 1:
        axes = axes.reshape(2, 1)
    sample_idx = list(active.index)[:n_compare]
    for j, idx in enumerate(sample_idx):
        r = df_cm.loc[idx]
        path_before = os.path.join(RES_ROOT, str(r["path"]).replace("/", os.sep))
        path_after = os.path.join(RES_ROOT, str(r["prep_path"]).replace("/", os.sep))
        img_before = cv2.imread(path_before)
        img_after = cv2.imread(path_after)
        bs_before = r.get("blurry_score")
        bs_after = _blurry_score_for_image(img_after) if img_after is not None else None
        # Top row: Before
        ax_b = axes[0, j]
        if img_before is not None:
            ax_b.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB), aspect="equal")
        ax_b.set_title(f"Before — {r['class']} blurry={bs_before:.2f}" if pd.notna(bs_before) and isinstance(bs_before, (int, float)) else f"Before — {r['class']}", fontsize=11)
        ax_b.set_axis_off()
        # Bottom row: After
        ax_a = axes[1, j]
        if img_after is not None:
            ax_a.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB), aspect="equal")
        ax_a.set_title(f"After — blurry={bs_after:.2f}" if bs_after is not None else "After (unsharp)", fontsize=11)
        ax_a.set_axis_off()
    axes[0, 0].set_ylabel("Before", fontsize=12, rotation=0, ha="right")
    axes[1, 0].set_ylabel("After", fontsize=12, rotation=0, ha="right")
    fig2.suptitle("Step 3 — Sharpness comparison (Before top, After bottom)", fontsize=12)
    plt.tight_layout()
    plt.show()

df_cm.drop(columns=["_before_path"], inplace=True)
df_cm.to_csv(prep_manifest_path, index=False)
print(f"Step 3 — blur: {done} enhanced with unsharp mask (OpenCV), prep_path updated for {done} rows")



# ### 7.5 Step 4 — Dark enhancement
# **_enhance_dark()**: per-channel **CLAHE** (clipLimit=2, tile 8×8) then **GaussianBlur**(3,3). For each **has_dark**, **prep_status** != `"removed"`: read from **prep_path** or **path**, write enhanced to `res/<RUN_TAG>/data/prep/step4/<class>/<fname>`, set **prep_path**. **_dark_score_for_image()**: 5th percentile of gray / 255. Before/After samples and 2-row comparison figure (**dark_score** in title).

# In[64]:


# Clean (dark → brighten): CLAHE + GaussianBlur per channel
df_cm = pd.read_csv(prep_manifest_path)
df_cm["prep_path"] = df_cm["prep_path"].astype(object).fillna("")
df_cm["prep_status"] = df_cm["prep_status"].astype(object).fillna("")

def _input_path(row):
    """First preference: prep_path; if empty, use path (both relative to RES_ROOT)."""
    p = str(row.get("prep_path") or "").strip()
    return p if p else row["path"]

def _dark_score_for_image(img_bgr):
    """Brightness score 0–1 (higher = brighter). Uses 5th percentile like CleanVision dark_score so After can show improvement."""
    if img_bgr is None:
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        p5 = np.percentile(gray, 5)
        return float(np.clip(p5 / 255.0, 0, 1))
    except Exception:
        return None

def _enhance_dark(img, clip_limit=2.0, tile_size=8):
    """Per-channel CLAHE then GaussianBlur (3x3) — matches eye_disease_v3: contrast boost then smooth to reduce grain."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = np.zeros_like(img)
    for i in range(3):
        ch = clahe.apply(img[:, :, i])
        out[:, :, i] = cv2.GaussianBlur(ch, (3, 3), 0)
    return out

step_dir = os.path.join(prep_dir, "step4")
step_rel = os.path.relpath(step_dir, RES_ROOT).replace("\\", "/")
active = df_cm[df_cm["has_dark"] & (df_cm["prep_status"] != "removed")]

done = 0
for idx, row in tqdm(active.iterrows(), total=len(active), desc="Step 4 dark", file=sys.stdout):
    src_rel = _input_path(row)
    src = os.path.join(RES_ROOT, src_rel.replace("/", os.sep))
    img = cv2.imread(src)
    if img is None:
        continue
    enhanced = _enhance_dark(img)
    out_dir = os.path.join(step_dir, row["class"])
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.basename(src_rel)
    cv2.imwrite(os.path.join(out_dir, fname), enhanced)
    df_cm.at[idx, "prep_path"] = f"{step_rel}/{row['class']}/{fname}"
    done += 1

display(HTML("<strong>Step 4 — Clean (dark → CLAHE + smooth, ref: eye_disease_v3)</strong>"))
display_wide(pd.DataFrame([{"stage": "Before", "flagged": len(active), "processed": 0}, {"stage": "After", "flagged": len(active), "processed": done}]))

# Before/After samples: show_attr_samples with dark_score in title
n_show = min(5, done)
if n_show > 0:
    sample_idx = list(active.index)[:n_show]
    records = []
    for idx in sample_idx:
        r = df_cm.loc[idx]
        ds = r.get("dark_score")
        cell_title = f"dark={ds:.2f} {r['class']}" if pd.notna(ds) and isinstance(ds, (int, float)) else str(r["class"])
        records.append({"stage": "Before", "path": r["path"], "cell_title": cell_title})
    for idx in sample_idx:
        r = df_cm.loc[idx]
        prep_full = os.path.join(RES_ROOT, str(r["prep_path"]).replace("/", os.sep))
        img_after = cv2.imread(prep_full) if str(r.get("prep_path") or "").strip() else None
        ds_after = _dark_score_for_image(img_after)
        cell_title = f"dark={ds_after:.2f} {r['class']}" if ds_after is not None else f"clean {r['class']}"
        records.append({"stage": "After", "path": r["prep_path"], "cell_title": cell_title})
    show_attr_samples(pd.DataFrame(records), "stage", "Step 4 — Before vs After (Clean; Before=manifest dark_score, After=computed)", n_cols=n_show, 
                      order=["Before", "After"], title_col="cell_title")

# Three samples: Before on top row, respective After on bottom row — minimal white borders
n_compare = min(3, done)
if n_compare > 0:
    fig4, axes = plt.subplots(2, n_compare, figsize=(5 * n_compare, 10))
    fig4.patch.set_facecolor("#374151")
    fig4.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.04, hspace=0.18)
    if n_compare == 1:
        axes = axes.reshape(2, 1)
    sample_idx = list(active.index)[:n_compare]
    for j, idx in enumerate(sample_idx):
        r = df_cm.loc[idx]
        path_before = os.path.join(RES_ROOT, str(r["path"]).replace("/", os.sep))
        path_after = os.path.join(RES_ROOT, str(r["prep_path"]).replace("/", os.sep))
        img_before = cv2.imread(path_before)
        img_after = cv2.imread(path_after)
        ds_before = r.get("dark_score")
        ds_after = _dark_score_for_image(img_after)
        ds_b_str = f" dark={ds_before:.2f}" if pd.notna(ds_before) and isinstance(ds_before, (int, float)) else ""
        ds_a_str = f" dark={ds_after:.2f}" if ds_after is not None else ""
        # Row 0 = Before (original, darker)
        ax_b = axes[0, j]
        ax_b.set_facecolor("#374151")
        if img_before is not None:
            ax_b.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB), aspect="equal")
        ax_b.set_title(f"Before — {r['class']}{ds_b_str}", fontsize=11, color="white")
        ax_b.set_axis_off()
        # Row 1 = After (CLAHE, brighter — score computed on enhanced image)
        ax_a = axes[1, j]
        ax_a.set_facecolor("#374151")
        if img_after is not None:
            ax_a.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB), aspect="equal")
        ax_a.set_title(f"After (Clean) — {r['class']}{ds_a_str}", fontsize=11, color="white")
        ax_a.set_axis_off()
    axes[0, 0].set_ylabel("Before", fontsize=12, rotation=0, ha="right", color="white")
    axes[1, 0].set_ylabel("After", fontsize=12, rotation=0, ha="right", color="white")
    fig4.suptitle("Step 4 — Clean (Before top, After bottom; ref: eye_disease_v3)", fontsize=12, color="white")
    plt.show()

df_cm.to_csv(prep_manifest_path, index=False)
print(f"Step 4 — clean: {done} enhanced (CLAHE + GaussianBlur), prep_path updated for {done} rows")

# ### 7.6 Data split — Train / validate / test
# Prep manifest in: **train** + **validate** (validate = raw `validate/` folder). Out: **train** (smaller) + **validate** (10% stratified from original train) + **test** (renamed from former **validate**). Skip if **test** already present.
#

# In[65]:


df_cm = pd.read_csv(prep_manifest_path)
if (df_cm["split"] == "test").any():
    display(HTML("<strong>§7.6 skipped — manifest already has <code>split=test</code> (split step already done).</strong>"))
else:
    # Snapshot original training rows before any label changes
    active_train = df_cm[df_cm["split"] == "train"].copy()
    holdout_count = int((df_cm["split"] == "validate").sum())
    # Step 1 — held-out folder: validate → test (final evaluation set)
    df_cm.loc[df_cm["split"] == "validate", "split"] = "test"
    # Step 2 — stratified train split: 90% train, 10% validate (model tuning)
    train_idx, validate_idx = train_test_split(
        active_train.index,
        test_size=0.10,
        stratify=active_train["class"],
        random_state=42,
    )
    done_split = 0
    for idx in tqdm(validate_idx, desc="Train → stratified validate", file=sys.stdout):
        df_cm.at[idx, "split"] = "validate"
        done_split += 1
    display(HTML("<strong>Manifest split update — holdout → test; train → train + validate</strong>"))
    stats_split = [
        {"stage": "Original train (before stratify)", "count": len(active_train)},
        {"stage": "New train (after stratify)", "count": len(train_idx)},
        {"stage": "New validate (10% from train)", "count": done_split},
        {"stage": "Test (was raw validate folder)", "count": holdout_count},
    ]
    display_wide(pd.DataFrame(stats_split))
    val_check = df_cm[df_cm["split"] == "validate"]["class"].value_counts().to_frame().reset_index()
    val_check.columns = ["class", "count in new validate"]
    print("\nClass distribution in new validation set (10% stratified split):")
    display(val_check)
    df_cm.to_csv(prep_manifest_path, index=False)
    print(f"\nSuccess: holdout relabeled validate→test ({holdout_count} rows); {done_split} train rows → validate.")

# Class balance by split (prep manifest: train / validate / test)
prep_classes_sorted = sorted(df_cm["class"].dropna().unique())
for split_key, split_title in [("train", "Train"), ("validate", "Validate"), ("test", "Test")]:
    sub = df_cm[df_cm["split"] == split_key]
    counts = sub["class"].value_counts()
    df_prep_split_chart = pd.DataFrame(
        {"class": prep_classes_sorted, "total": [int(counts.get(c, 0)) for c in prep_classes_sorted]}
    )
    display(HTML(f"<strong>§7.6 — Prep manifest: {split_title} (class distribution)</strong>"))
    show_attr_charts(df_prep_split_chart, "class")

# ## 8. Augmentation
# Two sub-steps, two locks: **aug_copy.lock**, **aug_generate.lock**. **augment.lock** (**aug_lock_path**) = augment section lock (both sub-steps treated complete). Output: `data/aug/train|validate|test/`; `meta/aug/image_manifest.csv`.

# ### 8.1 Copy files — Copy process files
# If **aug_copy.lock** or **aug_lock_path**: report from aug manifest only. Else: copy prep → aug, write **aug_copy.lock**.

# In[19]:


if os.path.isfile(aug_copy_lock_path) or os.path.isfile(aug_lock_path):
    aug_copy_skip_html = (
        "<strong>Augment copy already done.</strong> To re-run, delete:<br><code>"
        + os.path.abspath(aug_copy_lock_path)
        + "</code>"
    )
    if os.path.isfile(aug_lock_path) and not os.path.isfile(aug_copy_lock_path):
        aug_copy_skip_html += (
            "<br>Augment section lock (<code>augment.lock</code> — full augment section marked complete; copy skipped):<br><code>"
            + os.path.abspath(aug_lock_path)
            + "</code>"
        )
    display(HTML(aug_copy_skip_html))
    train_copied = val_copied = test_copied = 0
    train_class_counts = {}
    val_class_counts = {}
    test_class_counts = {}
    if os.path.isfile(aug_manifest_path):
        df_aug = pd.read_csv(aug_manifest_path)
        df_aug["aug_path"] = df_aug["aug_path"].astype(object).fillna("")

        def _aug_lock_stats(split_key):
            sub = df_aug[(df_aug["split"] == split_key) & (df_aug["aug_path"].str.strip() != "")]
            return len(sub), (sub["class"].value_counts().to_dict() if not sub.empty else {})

        train_copied, train_class_counts = _aug_lock_stats("train")
        val_copied, val_class_counts = _aug_lock_stats("validate")
        test_copied, test_class_counts = _aug_lock_stats("test")
    classes_aug = sorted(set(train_class_counts) | set(val_class_counts) | set(test_class_counts))
else:
    # copy prep manifest to aug
    shutil.copy2(prep_manifest_path, aug_manifest_path)
    print(f"Augment step 1 — copied prep manifest → {aug_manifest_path}")

    # Load aug manifest (paths come from here for copy step)
    df_aug = pd.read_csv(aug_manifest_path)
    df_aug["prep_path"] = df_aug["prep_path"].astype(object).fillna("")
    if "prep_status" not in df_aug.columns:
        df_aug["prep_status"] = ""
    else:
        df_aug["prep_status"] = df_aug["prep_status"].astype(object).fillna("")
    df_aug["aug_path"] = ""  # filled for each copied file (relative to RES_ROOT, like path/prep_path)

    def _aug_source_path(row):
        """Full path for an image: prep_path if non-empty, else path (relative to RES_ROOT)."""
        rel = str(row.get("prep_path") or "").strip()
        if not rel:
            rel = str(row.get("path") or "").strip()
        if not rel:
            return None
        return os.path.join(RES_ROOT, rel.replace("/", os.sep))

    def _aug_dest_rel(dest_abs):
        """Manifest aug_path relative to RES_ROOT — matches aug_train_dir / aug_val_dir / aug_test_dir layout."""
        return os.path.relpath(dest_abs, RES_ROOT).replace("\\", "/")

    def _copy_manifest_rows_to_aug(rows_df, aug_root, tqdm_desc):
        n = 0
        counts = {}
        for idx, row in tqdm(rows_df.iterrows(), total=len(rows_df), desc=tqdm_desc, file=sys.stdout):
            src = _aug_source_path(row)
            if src is None or not os.path.isfile(src):
                continue
            cls = row["class"]
            dest_dir = os.path.join(aug_root, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dest)
            df_aug.at[idx, "aug_path"] = _aug_dest_rel(dest)
            n += 1
            counts[cls] = counts.get(cls, 0) + 1
        return n, counts

    _aug_copy_jobs = [
        ((df_aug["split"] == "train") & (df_aug["prep_status"] != "removed"), aug_train_dir, "Augment copy train", "train"),
        (df_aug["split"] == "validate", aug_val_dir, "Augment copy validate", "validate"),
        (df_aug["split"] == "test", aug_test_dir, "Augment copy test", "test"),
    ]
    _aug_label_print = {"train": "train", "validate": "validate (dev)", "test": "test"}
    split_copied_and_counts = {}
    for mask, aug_dir, desc, label in _aug_copy_jobs:
        n, cts = _copy_manifest_rows_to_aug(df_aug[mask], aug_dir, desc)
        split_copied_and_counts[label] = (n, cts)
        print(f"Augment copy — {_aug_label_print[label]}: {n} files → {aug_dir}")
    train_copied, train_class_counts = split_copied_and_counts["train"]
    val_copied, val_class_counts = split_copied_and_counts["validate"]
    test_copied, test_class_counts = split_copied_and_counts["test"]

    # Write aug manifest back with aug_path column updated
    df_aug.to_csv(aug_manifest_path, index=False)
    print(f"Augment manifest updated with aug_path → {aug_manifest_path}")
    with open(aug_copy_lock_path, "w") as _f:
        _f.write("")
    print(f"Augment copy lock: {os.path.abspath(aug_copy_lock_path)}")

    classes_aug = sorted(set(train_class_counts) | set(val_class_counts) | set(test_class_counts))

# Single report block (from copy run or from manifest when lock present)
display(HTML("<strong>Augment copy — summary</strong>"))
df_aug_summary = pd.DataFrame([
    {"split": "train", "copied": train_copied, "path": aug_train_dir},
    {"split": "validate", "copied": val_copied, "path": aug_val_dir},
    {"split": "test", "copied": test_copied, "path": aug_test_dir},
])
display_wide(df_aug_summary)
if classes_aug:
    display(HTML("<strong>Augment copy — per class</strong>"))
    df_aug_per_class = pd.DataFrame([
        {
            "class": c,
            "train": train_class_counts.get(c, 0),
            "validate": val_class_counts.get(c, 0),
            "test": test_class_counts.get(c, 0),
        }
        for c in classes_aug
    ])
    display_wide(df_aug_per_class)


# ### 8.2 Augment images — Train only, 2000 per class
# If **aug_generate.lock** or **aug_lock_path**: skip. Else: generate train images, write **aug_generate.lock** (independent of aug copy lock).

# In[9]:


if os.path.isfile(aug_generate_lock_path) or os.path.isfile(aug_lock_path):
    aug_generate_skip_html = (
        "<strong>Augment generate already done.</strong> To re-run, delete:<br><code>"
        + os.path.abspath(aug_generate_lock_path)
        + "</code>"
    )
    if os.path.isfile(aug_lock_path) and not os.path.isfile(aug_generate_lock_path):
        aug_generate_skip_html += (
            "<br>Augment section lock (<code>augment.lock</code> — full augment section marked complete):<br><code>"
            + os.path.abspath(aug_lock_path)
            + "</code>"
        )
    display(HTML(aug_generate_skip_html))
else:
    TARGET_COUNT = 2000
    IMG_SIZE = (512, 512)
    # Get train source file list from aug_train_dir (files we already copied); one list per class, shuffled
    train_sources_by_class = {}
    if os.path.isdir(aug_train_dir):
        for cls in sorted(os.listdir(aug_train_dir)):
            class_dir = os.path.join(aug_train_dir, cls)
            if not os.path.isdir(class_dir) or cls.startswith("."):
                continue
            paths = []
            for f in os.listdir(class_dir):
                if f.startswith("."):
                    continue
                p = os.path.join(class_dir, f)
                if os.path.isfile(p):
                    paths.append(p)
            if paths:
                random.shuffle(paths)
                train_sources_by_class[cls] = paths
    classes_aug_step2 = sorted(train_sources_by_class.keys())

    # Three generators: rotation only, zoom only, horizontal_flip only (plan: only these three)
    datagen_rotation = ImageDataGenerator(
        rotation_range=15,
        fill_mode="constant",
        cval=0,
        zoom_range=0,
        horizontal_flip=False,
    )
    datagen_zoom = ImageDataGenerator(
        rotation_range=0,
        zoom_range=0.1,
        horizontal_flip=False,
    )
    datagen_flip = ImageDataGenerator(
        rotation_range=0,
        zoom_range=0,
        horizontal_flip=True,
    )
    datagens = [datagen_rotation, datagen_zoom, datagen_flip]

    for cls in tqdm(classes_aug_step2, desc="Augment classes", file=sys.stdout):
        source_paths = train_sources_by_class[cls]
        class_dir = os.path.join(aug_train_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        current_count = len(source_paths)
        needed = TARGET_COUNT - current_count
        if needed <= 0:
            continue

        for i in tqdm(range(needed), desc=f"Augment {cls}", leave=False, file=sys.stdout):
            src = np.random.choice(source_paths)
            stem = os.path.splitext(os.path.basename(src))[0]
            count_str = f"{i + 1:03d}"
            out_name = f"{stem}_aug{count_str}.png"
            out_path = os.path.join(class_dir, out_name)

            img = load_img(src, color_mode="rgb", target_size=IMG_SIZE)
            x = img_to_array(img)
            aug_type = i % 3
            x = datagens[aug_type].random_transform(x)

            if x.dtype != np.uint8:
                x = np.clip(x, 0, 255).astype(np.uint8)
            Image.fromarray(x).save(out_path, format="PNG")

    display(HTML("<strong>Augmentation done.</strong> Train split only; 2000 per class target."))

    with open(aug_generate_lock_path, "w") as _f:
        _f.write("")
    print(f"Augment generate lock: {os.path.abspath(aug_generate_lock_path)}")

# Augment — Class labels & counts (same style as raw "Class labels & counts" section)
def _count_files_in_dir(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and not f.startswith("."))

aug_train_counts = {}
aug_val_counts = {}
aug_test_counts = {}
for name in (os.listdir(aug_train_dir) if os.path.isdir(aug_train_dir) else []):
    p = os.path.join(aug_train_dir, name)
    if os.path.isdir(p) and not name.startswith("."):
        aug_train_counts[name] = _count_files_in_dir(p)
for name in (os.listdir(aug_val_dir) if os.path.isdir(aug_val_dir) else []):
    p = os.path.join(aug_val_dir, name)
    if os.path.isdir(p) and not name.startswith("."):
        aug_val_counts[name] = _count_files_in_dir(p)
for name in (os.listdir(aug_test_dir) if os.path.isdir(aug_test_dir) else []):
    p = os.path.join(aug_test_dir, name)
    if os.path.isdir(p) and not name.startswith("."):
        aug_test_counts[name] = _count_files_in_dir(p)
classes_aug_report = sorted(set(aug_train_counts) | set(aug_val_counts) | set(aug_test_counts))

df_aug_counts = pd.DataFrame([
    {"data": "aug", "split": "train", **{c: aug_train_counts.get(c, 0) for c in classes_aug_report}, "total": sum(aug_train_counts.values())},
    {"data": "aug", "split": "validate", **{c: aug_val_counts.get(c, 0) for c in classes_aug_report}, "total": sum(aug_val_counts.values())},
    {"data": "aug", "split": "test", **{c: aug_test_counts.get(c, 0) for c in classes_aug_report}, "total": sum(aug_test_counts.values())},
])
df_aug_counts = df_aug_counts[["data", "split"] + list(classes_aug_report) + ["total"]]

df_aug_train_chart = pd.DataFrame({"class": classes_aug_report, "total": [aug_train_counts.get(c, 0) for c in classes_aug_report]})
df_aug_val_chart = pd.DataFrame({"class": classes_aug_report, "total": [aug_val_counts.get(c, 0) for c in classes_aug_report]})
df_aug_test_chart = pd.DataFrame({"class": classes_aug_report, "total": [aug_test_counts.get(c, 0) for c in classes_aug_report]})

display(HTML("<strong>Augment — Class labels & counts</strong>"))
display_wide(df_aug_counts)
display(HTML("<strong>Augment — Train (class distribution)</strong>"))
show_attr_charts(df_aug_train_chart, "class")
display(HTML("<strong>Augment — Validate / dev (class distribution)</strong>"))
show_attr_charts(df_aug_val_chart, "class")
display(HTML("<strong>Augment — Test / held-out (class distribution)</strong>"))
show_attr_charts(df_aug_test_chart, "class")


# ## 9. Modeling
# Classifiers are trained from a **frozen copy** of the aug tree under  
# **`res/<RUN_TAG>/data/model/<model_name>/train|validate|test/<class>/..`** so **aug/** is never modified.
# Final **`state_dict`** for each model is saved under that same **`data/model/<model_name>/`** as **`<model_name>.pth`**; if that file exists, the model’s **training** cell skips fitting and only loads for evaluation.
#
# - **§9.1 Common methods:** path layout, copy **aug → model**, generic train/validate loop, eval helper — no input modality (RGB vs medical) assumptions.
# - **§9.2** **`effnet_b4_rgb`**: **RGB** — unchanged aug images copied into **`data/model/effnet_b4_rgb/`**, ImageNet-style resize/normalize, EfficientNet-B4 (v5 aug-track).
# - **§9.3** **`effnet_b4_g`**: **green-channel** pipeline — **`medical_step_b4`** (v5 med-track) writes pseudo-RGB PNGs into **`data/model/effnet_b4_g/`**; loaders use **ToTensor + normalize** only (380×380 on disk).

# ### 9.1 Common methods
# Code cell used by all §9 models. **Does not** write under **aug/**; reads **aug** and writes only under **`data/model/<model_name>/`**.
#
# **Imports:** **PyTorch** core (`nn`, `optim`, **`DataLoader`**); **sklearn** classification report + confusion matrix for **`display_classification_eval_charts`**. Per-model cells add **torchvision** / **`ssl`** as needed.
#
# **Path helper:** **`model_image_roots(model_name)`** — returns dict with **`root`**, **`train`**, **`validate`**, **`test`** paths under `res/<RUN_TAG>/data/model/<model_name>/`.
#
# **Copy from aug:** **`copy_split_class_tree_to_model`** copies one aug split root (class subfolders + images per **IMAGE_EXTENSIONS**) into the matching model split folder. **`copy_all_aug_splits_to_model`** runs train / validate / test from **`aug_train_dir`**, **`aug_val_dir`**, **`aug_test_dir`**; writes **`from_aug_copy.lock`** in the model **`root`** when done. **`skip_if_lock=True`** skips recopy when the lock exists (delete lock to refresh after aug changes).
#
# **Training:** **`train_model_validate_best`** — each epoch runs **`train`** then **`validate`**; records loss/acc history; when **validate** accuracy improves, saves **`state_dict`** to **`best_weights_path`**.
#
# **Evaluation:** **`evaluate_on_split`** — runs model on a loader, returns **`y_true`**, **`y_pred`**. **`display_classification_eval_charts`** — sklearn **`classification_report`** as **`display_wide`** HTML table (same look as augment / prep summaries); then **`show_attr_charts`** for bar + sunburst + confusion heatmap.

# In[90]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# Resolve **root/train/validate/test** paths under `res/<RUN_TAG>/data/model/<model_name>/`.
def model_image_roots(model_name):
    """
    Per-model directory layout under the run tag (image files only; checkpoints sit alongside).
    """
    root = os.path.join(RES_ROOT, RUN_TAG, "data", "model", model_name)
    return {
        "root": root,
        "train": os.path.join(root, "train"),
        "validate": os.path.join(root, "validate"),
        "test": os.path.join(root, "test"),
    }


# Bitwise copy of allowed image extensions from one aug split tree into a model split folder.
def copy_split_class_tree_to_model(aug_split_dir, model_split_dir, tqdm_label):
    """
    Copy `<class>/*` image files from one aug split root into the model split root.
    Uses **IMAGE_EXTENSIONS**; does not alter **aug_split_dir**.
    """
    if not os.path.isdir(aug_split_dir):
        print(f"copy_split_class_tree_to_model: missing source {aug_split_dir}")
        return 0
    total = 0
    for klass in sorted(x for x in os.listdir(aug_split_dir) if not x.startswith(".")):
        kp = os.path.join(aug_split_dir, klass)
        if not os.path.isdir(kp):
            continue
        out_k = os.path.join(model_split_dir, klass)
        os.makedirs(out_k, exist_ok=True)
        files = [
            f
            for f in os.listdir(kp)
            if not f.startswith(".") and f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        for fn in tqdm(files, desc=f"{tqdm_label} {klass}", file=sys.stdout, leave=False):
            shutil.copy2(os.path.join(kp, fn), os.path.join(out_k, fn))
            total += 1
    return total


# Copy all aug splits into **model_name** layout; **from_aug_copy.lock** can skip repeat work.
def copy_all_aug_splits_to_model(model_name, skip_if_lock=False):
    """
    Copy **aug** train / validate / test class folders into  
    **`data/model/<model_name>/train|validate|test`**.

    Optional **from_aug_copy.lock** in the model folder skips repeat copies (delete lock to refresh).
    """
    layout = model_image_roots(model_name)
    os.makedirs(layout["root"], exist_ok=True)
    lock_path = os.path.join(layout["root"], "from_aug_copy.lock")

    if skip_if_lock and os.path.isfile(lock_path):
        display(
            HTML(
                "<strong>Model data copy skipped (lock).</strong> Delete to recopy from aug:<br><code>"
                + os.path.abspath(lock_path)
                + "</code>"
            )
        )
        return None

    n_train = copy_split_class_tree_to_model(aug_train_dir, layout["train"], "Model copy train")
    n_val = copy_split_class_tree_to_model(aug_val_dir, layout["validate"], "Model copy validate")
    n_test = copy_split_class_tree_to_model(aug_test_dir, layout["test"], "Model copy test")

    with open(lock_path, "w") as _lk:
        _lk.write("")
    summary = pd.DataFrame(
        [
            {"split": "train", "copied": n_train, "path": layout["train"]},
            {"split": "validate", "copied": n_val, "path": layout["validate"]},
            {"split": "test", "copied": n_test, "path": layout["test"]},
        ]
    )
    display(HTML("<strong>Copied aug → model folder</strong>"))
    display_wide(summary)
    print(f"Model copy lock: {os.path.abspath(lock_path)}")
    return layout


# Epoch loop over **phases** (train + validate); persist best validate acc to **best_weights_path**.
def train_model_validate_best(
    model,
    criterion,
    optimizer,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs,
    best_weights_path,
    phases=("train", "validate"),
):
    """
    Alternates **train** and **validate** each epoch; saves **state_dict** when validate accuracy improves.
    """
    best_acc = 0.0
    history = {f"{p}_loss": [] for p in phases}
    history.update({f"{p}_acc": [] for p in phases})

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for phase in phases:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], leave=False, file=sys.stdout):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())
            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "validate" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_weights_path)
                print(f"  → saved best weights ({best_acc:.4f}) → {best_weights_path}")

    return model, history


# Run **model** on one **DataLoader**; collect label indices for sklearn / reporting.
def evaluate_on_split(model, dataloader, device, title="evaluation"):
    """Return **y_true**, **y_pred** numpy lists for sklearn metrics."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=title, file=sys.stdout):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred


# Held-out test: HTML metric table via **display_wide** + **show_attr_charts** confusion/balance.
def display_classification_eval_charts(y_true, y_pred, class_names, model_label):
    """
    **Held-out test** evaluation: **`display_wide`** table built from sklearn **classification_report**
    (same styling as other notebook summaries), then **`show_attr_charts`** (bar + sunburst + confusion heatmap).
    """
    cn = list(class_names)
    rep = classification_report(
        y_true, y_pred, target_names=cn, output_dict=True, zero_division=0
    )
    rows = []
    for name in cn:
        r = rep[name]
        rows.append(
            {
                "Class": name,
                "Precision": round(float(r["precision"]), 4),
                "Recall": round(float(r["recall"]), 4),
                "F1-score": round(float(r["f1-score"]), 4),
                "Support": int(r["support"]),
            }
        )
    n_sup = sum(int(rep[c]["support"]) for c in cn)
    rows.append(
        {
            "Class": "accuracy",
            "Precision": "",
            "Recall": "",
            "F1-score": round(float(rep["accuracy"]), 4),
            "Support": n_sup,
        }
    )
    for agg in ("macro avg", "weighted avg"):
        r = rep[agg]
        rows.append(
            {
                "Class": agg,
                "Precision": round(float(r["precision"]), 4),
                "Recall": round(float(r["recall"]), 4),
                "F1-score": round(float(r["f1-score"]), 4),
                "Support": int(r["support"]),
            }
        )
    df_rep = pd.DataFrame(rows)
    display(HTML(f"<strong>Test set (held-out) classification report — {model_label}</strong>"))
    display_wide(df_rep)

    labels_ord = list(range(len(cn)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_ord)
    df_cm = pd.DataFrame(cm, columns=cn)
    df_cm.insert(0, "class", cn)
    df_cm["total"] = df_cm[cn].sum(axis=1).astype(int)
    display(
        HTML(f"<strong>{model_label} — held-out test: class balance + confusion counts</strong>")
    )
    show_attr_charts(df_cm, "class")


# ### 9.2 Model **`effnet_b4_rgb`** — EfficientNet-B4 on RGB aug copy
# Copies **`aug/train|validate|test/<class>/`** (read-only source) into **`data/model/effnet_b4_rgb/...`** via §9.1 **`copy_all_aug_splits_to_model`**. **`from_aug_copy.lock`** in that model root skips recopy when **`skip_if_lock=True`** (delete lock to refresh after aug changes).
#
# **Loaders:** **torchvision** **Resize** to B4 input size, **ToTensor**, ImageNet **mean/std** normalize; **`ImageFolder`** + **`DataLoader`** on the on-disk copy.
#
# **`ssl`** workaround may apply when downloading ImageNet backbone weights over HTTPS.
#
# **Checkpoint:** **`effnet_b4_rgb.pth`** in the model directory; if present, **training is skipped** (same pattern as §9.3). Test report + confusion charts: §9.1 **`display_classification_eval_charts`**.
#
# Uses §9.1: **`copy_all_aug_splits_to_model`**, **`train_model_validate_best`**, **`evaluate_on_split`**, **`display_classification_eval_charts`**.

# In[91]:


import ssl

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import torchvision.transforms as transforms


# --- Identity & training hyperparameters (RGB / aug-copy track) ---
MODEL_NAME_EFFNET_B4_RGB = "effnet_b4_rgb"
EFFNET_B4_IMG_SIZE = 380
EFFNET_B4_BATCH = 16
EFFNET_B4_EPOCHS = 15
EFFNET_B4_LR = 0.001
EFFNET_B4_NUM_WORKERS = 0  # 0: safe default for Jupyter on Windows


# **ImageFolder** + loaders for RGB on-disk trees: resize, tensor, ImageNet normalize.
def make_rgb_imagenet_loaders(
    layout,
    img_size=380,
    batch_size=16,
    num_workers=0,
):
    """
    **RGB / ImageNet-pretrained** pipeline: train / validate / test **ImageFolder** + **DataLoader**.
    **num_workers=0** avoids common multiprocessing issues in Jupyter on Windows.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_datasets = {
        "train": ImageFolder(root=layout["train"], transform=tfm),
        "validate": ImageFolder(root=layout["validate"], transform=tfm),
        "test": ImageFolder(root=layout["test"], transform=tfm),
    }
    dataloaders = {
        k: DataLoader(
            image_datasets[k],
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
        )
        for k in image_datasets
    }
    dataset_sizes = {k: len(image_datasets[k]) for k in image_datasets}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# Pretrained weight download over HTTPS (optional SSL workaround; same idea as §9.3)
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}; model={MODEL_NAME_EFFNET_B4_RGB}")

# Build `data/model/effnet_b4_rgb/{train,validate,test}/...` from aug (read-only source)
layout_effnet_rgb = copy_all_aug_splits_to_model(
    MODEL_NAME_EFFNET_B4_RGB,
    skip_if_lock=True,
)
if layout_effnet_rgb is None:
    # Lock told us to skip recopy; paths still come from **model_image_roots**
    layout_effnet_rgb = model_image_roots(MODEL_NAME_EFFNET_B4_RGB)

# Best checkpoint path: same basename as **MODEL_NAME_EFFNET_B4_RGB** (.pth)
effnet_b4_rgb_weights_path = os.path.join(
    layout_effnet_rgb["root"], f"{MODEL_NAME_EFFNET_B4_RGB}.pth"
)
skip_effnet_rgb_training = os.path.isfile(effnet_b4_rgb_weights_path)

dataloaders_effnet, sizes_effnet, class_names_effnet = make_rgb_imagenet_loaders(
    layout_effnet_rgb,
    img_size=EFFNET_B4_IMG_SIZE,
    batch_size=EFFNET_B4_BATCH,
    num_workers=EFFNET_B4_NUM_WORKERS,
)
num_classes_effnet = len(class_names_effnet)
print(
    f"model={MODEL_NAME_EFFNET_B4_RGB}; classes={num_classes_effnet} {class_names_effnet}; splits={sizes_effnet}"
)

# EfficientNet-B4: ImageNet backbone only when training (skip download if loading .pth only)
weights_b4_rgb = models.EfficientNet_B4_Weights.IMAGENET1K_V1
model_effnet_b4_rgb = models.efficientnet_b4(
    weights=None if skip_effnet_rgb_training else weights_b4_rgb,
)
for param in model_effnet_b4_rgb.parameters():
    param.requires_grad = False
num_ftrs_rgb = model_effnet_b4_rgb.classifier[1].in_features
model_effnet_b4_rgb.classifier[1] = nn.Linear(num_ftrs_rgb, num_classes_effnet)
model_effnet_b4_rgb = model_effnet_b4_rgb.to(device)

test_eval_title_rgb = f"Test eval — {MODEL_NAME_EFFNET_B4_RGB}"

# Train only when weights file missing; otherwise UI message and go straight to load + test
if skip_effnet_rgb_training:
    display(
        HTML(
            "<strong>Training skipped.</strong> Remove weights to retrain:<br><code>"
            + os.path.abspath(effnet_b4_rgb_weights_path)
            + "</code>"
        )
    )
else:
    criterion_effnet = nn.CrossEntropyLoss()
    optimizer_effnet = optim.Adam(
        model_effnet_b4_rgb.classifier[1].parameters(), lr=EFFNET_B4_LR
    )
    model_effnet_b4_rgb, history_effnet = train_model_validate_best(
        model_effnet_b4_rgb,
        criterion_effnet,
        optimizer_effnet,
        dataloaders_effnet,
        sizes_effnet,
        device,
        EFFNET_B4_EPOCHS,
        effnet_b4_rgb_weights_path,
        phases=("train", "validate"),
    )
    display(
        HTML(
            "<strong>Training finished.</strong> Weights:<br><code>"
            + os.path.abspath(effnet_b4_rgb_weights_path)
            + "</code>"
        )
    )

# Reload best **state_dict** (last epoch in memory may differ)
model_effnet_b4_rgb.load_state_dict(
    torch.load(effnet_b4_rgb_weights_path, map_location=device)
)
# Held-out **test** split only here (validate was used during training for model selection)
y_true_t, y_pred_t = evaluate_on_split(
    model_effnet_b4_rgb,
    dataloaders_effnet["test"],
    device,
    title=test_eval_title_rgb,
)
# **`display_wide`** classification_report table + balance/confusion **show_attr_charts**
display_classification_eval_charts(
    y_true_t, y_pred_t, class_names_effnet, MODEL_NAME_EFFNET_B4_RGB
)


# ### 9.3 Model **`effnet_b4_g`** — EfficientNet-B4 on green-channel (CLAHE) preprocessing
# Reads **aug/** (never modified). For each split, applies **`medical_step_b4`** from `eye_disease_v5.py`: green channel → CLAHE → blur → ROI crop → 380×380 → **pseudo-RGB** (3 identical planes), saves **PNG** under **`data/model/effnet_b4_g/train|validate|test/<class>/`**.
#
# **`from_aug_green_processed.lock`** in that model root skips reprocessing when **`skip_if_lock=True`** (delete lock — and optionally the split folders — to rebuild after aug changes).
#
# **Loaders:** no **Resize** in **transforms** (images are already 380×380); **ToTensor** + ImageNet **normalize** (same as v5 med dataloaders).
#
# **Checkpoint:** **`effnet_b4_g.pth`** in the model directory; if present, **training is skipped** (same pattern as §9.2). **ssl** workaround applies when downloading ImageNet backbone weights. Test report + confusion matrix: §9.1 **`display_classification_eval_charts`**.

# In[92]:


import ssl

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
import torchvision.transforms as transforms

# --- Identity & training hyperparameters (green / pseudo-RGB track) ---
MODEL_NAME_EFFNET_B4_G = "effnet_b4_g"
EFFNET_B4_G_IMG_SIZE = 380
EFFNET_B4_G_BATCH = 16
EFFNET_B4_G_EPOCHS = 15
EFFNET_B4_G_LR = 0.001
EFFNET_B4_G_NUM_WORKERS = 0  # 0: safe default for Jupyter on Windows


# Load one image path → 380×380×3 pseudo-RGB (green/CLAHE/ROI) BGR array, or None if read fails.
def medical_step_b4(image_path, target_size=(380, 380)):
    """Green-channel fundus-style prep for B4 (ref: eye_disease_v5.py)."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    green = img[:, :, 1]  # vessel contrast vs full RGB

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        blurred = blurred[y : y + h, x : x + w]

    resized = cv2.resize(blurred, target_size)
    # Stack grayscale plane ×3 so EfficientNet sees 3-channel input
    return cv2.merge([resized, resized, resized])


# Process every image under one aug split (e.g. train) into **model_split_dir** class folders as PNGs.
def process_aug_split_to_effnet_b4_g(
    aug_split_dir,
    model_split_dir,
    tqdm_label,
    target_size=(380, 380),
):
    """Walk aug split class folders; write green-processed PNGs into **model_split_dir**."""
    # One aug split (train / validate / test) → mirror class subdirs under model folder
    if not os.path.isdir(aug_split_dir):
        print(f"process_aug_split_to_effnet_b4_g: missing source {aug_split_dir}")
        return 0
    total = 0
    for klass in sorted(x for x in os.listdir(aug_split_dir) if not x.startswith(".")):
        kp = os.path.join(aug_split_dir, klass)
        if not os.path.isdir(kp):
            continue
        out_k = os.path.join(model_split_dir, klass)
        os.makedirs(out_k, exist_ok=True)
        files = [
            f
            for f in os.listdir(kp)
            if not f.startswith(".") and f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        for fn in tqdm(files, desc=f"{tqdm_label} {klass}", file=sys.stdout, leave=False):
            input_path = os.path.join(kp, fn)
            processed = medical_step_b4(input_path, target_size=target_size)
            if processed is not None:
                # Normalize extension: always PNG after CLAHE pipeline
                out_name = os.path.splitext(fn)[0] + ".png"
                cv2.imwrite(os.path.join(out_k, out_name), processed)
                total += 1
    return total


# Run green pipeline for train + validate + test aug trees into **data/model/<model_name>/** (optional lock).
def process_all_aug_splits_to_effnet_b4_g(model_name, skip_if_lock=False, target_size=(380, 380)):
    """Populate **data/model/<model_name>/** from aug via **`medical_step_b4`**."""
    layout = model_image_roots(model_name)
    os.makedirs(layout["root"], exist_ok=True)
    # Skip full reprocessing if user already ran this (delete lock to refresh after aug edits)
    lock_path = os.path.join(layout["root"], "from_aug_green_processed.lock")

    if skip_if_lock and os.path.isfile(lock_path):
        display(
            HTML(
                "<strong>Green preprocess skipped (lock).</strong> Delete to rebuild from aug:<br><code>"
                + os.path.abspath(lock_path)
                + "</code>"
            )
        )
        return None

    n_train = process_aug_split_to_effnet_b4_g(
        aug_train_dir, layout["train"], "Green train", target_size
    )
    n_val = process_aug_split_to_effnet_b4_g(
        aug_val_dir, layout["validate"], "Green validate", target_size
    )
    n_test = process_aug_split_to_effnet_b4_g(
        aug_test_dir, layout["test"], "Green test", target_size
    )

    with open(lock_path, "w") as _lk:
        _lk.write("")
    summary = pd.DataFrame(
        [
            {"split": "train", "processed": n_train, "path": layout["train"]},
            {"split": "validate", "processed": n_val, "path": layout["validate"]},
            {"split": "test", "processed": n_test, "path": layout["test"]},
        ]
    )
    display(HTML(f"<strong>Processed aug → {model_name}</strong>"))
    display_wide(summary)
    print(f"Green preprocess lock: {os.path.abspath(lock_path)}")
    return layout


# Build **DataLoader**s over **layout** PNGs: tensorize + ImageNet normalize (no resize).
def make_green_effnet_loaders(
    layout,
    batch_size=16,
    num_workers=0,
):
    """Already 380×380 on disk — only tensor + ImageNet normalize (v5 med-track)."""
    # No Resize: **medical_step_b4** fixed output size; normalize matches ImageNet-pretrained B4
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_datasets = {
        "train": ImageFolder(root=layout["train"], transform=tfm),
        "validate": ImageFolder(root=layout["validate"], transform=tfm),
        "test": ImageFolder(root=layout["test"], transform=tfm),
    }
    dataloaders = {
        k: DataLoader(
            image_datasets[k],
            batch_size=batch_size,
            shuffle=(k == "train"),
            num_workers=num_workers,
        )
        for k in image_datasets
    }
    dataset_sizes = {k: len(image_datasets[k]) for k in image_datasets}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


# Pretrained weight download over HTTPS (optional SSL workaround; same idea as RGB model cell)
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}; model={MODEL_NAME_EFFNET_B4_G}")

# Build `data/model/effnet_b4_g/{train,validate,test}/...` from aug (read-only source)
layout_effnet_g = process_all_aug_splits_to_effnet_b4_g(
    MODEL_NAME_EFFNET_B4_G,
    skip_if_lock=True,
    target_size=(EFFNET_B4_G_IMG_SIZE, EFFNET_B4_G_IMG_SIZE),
)
if layout_effnet_g is None:
    # Lock told us to skip disk copy; paths still come from **model_image_roots**
    layout_effnet_g = model_image_roots(MODEL_NAME_EFFNET_B4_G)

# Best checkpoint path: same basename as **MODEL_NAME_EFFNET_B4_G** (.pth)
effnet_b4_g_weights_path = os.path.join(
    layout_effnet_g["root"], f"{MODEL_NAME_EFFNET_B4_G}.pth"
)
skip_effnet_g_training = os.path.isfile(effnet_b4_g_weights_path)

dataloaders_g, sizes_g, class_names_g = make_green_effnet_loaders(
    layout_effnet_g,
    batch_size=EFFNET_B4_G_BATCH,
    num_workers=EFFNET_B4_G_NUM_WORKERS,
)
num_classes_g = len(class_names_g)
print(
    f"model={MODEL_NAME_EFFNET_B4_G}; classes={num_classes_g} {class_names_g}; splits={sizes_g}"
)

# EfficientNet-B4: ImageNet backbone only when training (skip download if loading .pth only)
weights_b4_g = models.EfficientNet_B4_Weights.IMAGENET1K_V1
model_effnet_b4_g = models.efficientnet_b4(
    weights=None if skip_effnet_g_training else weights_b4_g,
)
for param in model_effnet_b4_g.parameters():
    param.requires_grad = False
num_ftrs_g = model_effnet_b4_g.classifier[1].in_features
model_effnet_b4_g.classifier[1] = nn.Linear(num_ftrs_g, num_classes_g)
model_effnet_b4_g = model_effnet_b4_g.to(device)

test_eval_title_g = f"Test eval — {MODEL_NAME_EFFNET_B4_G}"

# Train only when weights file missing; otherwise UI message and go straight to load + test
if skip_effnet_g_training:
    display(
        HTML(
            "<strong>Training skipped.</strong> Remove weights to retrain:<br><code>"
            + os.path.abspath(effnet_b4_g_weights_path)
            + "</code>"
        )
    )
else:
    # Head-only Adam: frozen trunk, same pattern as **effnet_b4_rgb**
    criterion_g = nn.CrossEntropyLoss()
    optimizer_g = optim.Adam(
        model_effnet_b4_g.classifier[1].parameters(), lr=EFFNET_B4_G_LR
    )
    model_effnet_b4_g, _ = train_model_validate_best(
        model_effnet_b4_g,
        criterion_g,
        optimizer_g,
        dataloaders_g,
        sizes_g,
        device,
        EFFNET_B4_G_EPOCHS,
        effnet_b4_g_weights_path,
        phases=("train", "validate"),
    )
    display(
        HTML(
            "<strong>Training finished.</strong> Weights:<br><code>"
            + os.path.abspath(effnet_b4_g_weights_path)
            + "</code>"
        )
    )

# Reload best **state_dict** (last epoch in memory may differ)
model_effnet_b4_g.load_state_dict(
    torch.load(effnet_b4_g_weights_path, map_location=device)
)
# Held-out **test** split only here (validate was used during training for model selection)
y_true_g, y_pred_g = evaluate_on_split(
    model_effnet_b4_g,
    dataloaders_g["test"],
    device,
    title=test_eval_title_g,
)
# **display_classification_eval_charts**: **`display_wide`** **classification_report** table + balance/confusion **show_attr_charts**
display_classification_eval_charts(
    y_true_g, y_pred_g, class_names_g, MODEL_NAME_EFFNET_B4_G
)

