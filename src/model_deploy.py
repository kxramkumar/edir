"""
Multi-model Streamlit deploy: configure models in **MODEL_LIST** (name + optional weights path).

Each model uses the same EfficientNet-B4 head (**effnet_b4_g** training recipe): green → CLAHE →
blur → ROI crop → 380×380 pseudo-RGB.

Weights: set **``weights``** in **MODEL_LIST** (absolute path, or relative to this file’s directory —
**``src/``** when the script lives in **``src/``**). If **``weights``** is omitted, resolution falls back to:

1. Env **EDIR_<NAME_UPPER>_WEIGHTS** (absolute or repo-relative from **REPO_ROOT**).
2. **mode/<name>/<name>.pth**
3. **res/<RUN_TAG>/data/model/<name>/<name>.pth**

Layout: **Setup** (path) → **Result by CLASS** → **Summary** (tabs: each model, **ALL**, **PREVIEW** when multi-model). **PREVIEW** shows side-by-side images and predictions for rows selected in **Result by CLASS**. One **file** × all models per rerun until complete. Session cache until path set changes.

Run from repo root::

    uv sync
    uv run streamlit run src/model_deploy.py
"""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# Repo root = parent of ``src/``; deploy dir = directory of this file (local ``.pth`` siblings).
REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = Path(__file__).resolve().parent

# --- Models: ``name`` + ``weights`` path. Relative ``weights`` resolve under **DEPLOY_DIR** (same folder as this file). ---
MODEL_LIST: list[dict[str, str | None]] = [
    {"name": "effnet_b4_g", "weights": str(DEPLOY_DIR / "effnet_b4_g.pth")},
    {"name": "effnet_b4_rgb", "weights": str(DEPLOY_DIR / "effnet_b4_rgb.pth")},
]

IMG_SIZE = 380

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

# Two-letter UI codes for softmax text (cell **background** color via Pandas Styler).
CLASS_CODE_UI: dict[str, str] = {
    "cataract": "CA",
    "diabetic_retinopathy": "DR",
    "glaucoma": "GL",
    "normal": "NR",
}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

BATCH_PATH_INPUT_KEY = "batch_path_input"
PATH_FEEDBACK_STATE_KEY = "_path_feedback_norm"
# Must not match any **MODEL_LIST** ``name`` (reserved Summary-tab label).
SUMMARY_TAB_PREVIEW = "Preview"
SUMMARY_TAB_PICK_KEY = "summary_tab_pick"
COMPARE_PATHS_SIG_KEY = "_compare_paths_sig"

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TENSOR_TFM = transforms.Compose(
    [
        transforms.ToTensor(),
        IMAGENET_NORMALIZE,
    ]
)


def normalize_user_path(s: str) -> str:
    """Strip whitespace, quotes, and common invisible paste junk (Windows-friendly)."""
    s = s.strip().strip('"').strip("'").replace("\r", "").replace("\n", "")
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d"):
        s = s.replace(ch, "")
    return s.strip()


def path_from_user_input(raw: str) -> Path | None:
    """Normalize pasted paths: ``normpath`` + ``expanduser`` (helps Windows slash / typo recovery)."""
    s = normalize_user_path(raw)
    if not s:
        return None
    return Path(os.path.normpath(s)).expanduser()


def meta_key_slug(meta_key: str) -> str:
    return hashlib.sha256(meta_key.encode("utf-8", errors="replace")).hexdigest()[:12]


def _summary_segmented_pick(
    widget_return: object,
    session_value: object,
    valid_display: list[str],
) -> str:
    """Map ``st.segmented_control`` (single) return / session value to a display label (``str`` or one-element list)."""
    valid = set(valid_display)
    default = valid_display[0]
    for raw in (widget_return, session_value):
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            if len(raw) == 0:
                continue
            v = raw[0]
        else:
            v = raw
        s = str(v)
        if s in valid:
            return s
    return default


def infer_class_from_path(raw_name: str) -> str | None:
    path = Path(raw_name)
    class_set = set(CLASS_NAMES)
    for part in path.parts:
        stem = part.rsplit(".", 1)[0] if "." in part else part
        if stem in class_set:
            return stem
    return None


def list_image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def batch_files_signature(paths: list[Path]) -> str:
    """Stable hash for the current batch (paths + mtimes)."""
    if not paths:
        return "empty"
    parts: list[str] = []
    for p in sorted(paths, key=lambda x: str(x).lower()):
        try:
            parts.append(f"{p.resolve()}:{p.stat().st_mtime_ns}")
        except OSError:
            parts.append(str(p.resolve()))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _inject_deploy_like_css() -> None:
    st.markdown(
        """
<style>
h1, h2, h3 {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-weight: 700;
    color: #1e3d59;
}
[data-testid="stSidebar"] {
    display: none !important;
}
section.main > div {
    padding-top: 0.5rem;
    max-width: 1200px;
    margin: 0 auto;
}
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def resolve_weights_path(model_name: str, explicit: str | None) -> tuple[Path | None, list[Path]]:
    candidates: list[Path] = []
    weights_filename = f"{model_name}.pth"

    if explicit and explicit.strip():
        p = Path(explicit.strip())
        if not p.is_absolute():
            p = DEPLOY_DIR / p
        candidates.append(p.resolve())
    else:
        env_key = f"EDIR_{model_name.upper()}_WEIGHTS"
        env = os.environ.get(env_key, "").strip()
        if env:
            p = Path(env)
            if not p.is_absolute():
                p = REPO_ROOT / p
            candidates.append(p.resolve())

        candidates.append((REPO_ROOT / "mode" / model_name / weights_filename).resolve())
        candidates.extend(
            sorted(REPO_ROOT.glob(f"res/*/data/model/{model_name}/{weights_filename}"))
        )

    seen: set[Path] = set()
    uniq: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if c.is_file():
            return c, uniq
    return None, uniq


def preprocess_fundus_effnet_b4_g(pil_image: Image.Image) -> Image.Image:
    rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    green = rgb[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        blurred = blurred[y : y + h, x : x + w]

    resized = cv2.resize(blurred, (IMG_SIZE, IMG_SIZE))
    bgr = cv2.merge([resized, resized, resized])
    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


@st.cache_resource
def load_effnet_b4_g(weights_path: str) -> torch.nn.Module:
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b4(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, NUM_CLASSES)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_proba(model: torch.nn.Module, pil_prep: Image.Image) -> tuple[int, np.ndarray]:
    device = next(model.parameters()).device
    x = TENSOR_TFM(pil_prep).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def build_file_inventory_dataframe(folder: Path, paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in paths:
        rel = str(p.relative_to(folder)) if p.is_relative_to(folder) else p.name
        rows.append(
            {
                "relative_path": rel,
                "class_from_path": infer_class_from_path(str(p)) or "—",
                "full_path": str(p.resolve()),
            }
        )
    return pd.DataFrame(rows)


def predict_row_from_prepped(model: torch.nn.Module, prepped: Image.Image) -> dict[str, object]:
    pred_idx, probs = predict_proba(model, prepped)
    row: dict[str, object] = {
        "inference_status": "done",
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
    }
    for cname, prob in zip(CLASS_NAMES, probs):
        row[f"P({cname})"] = float(prob)
    return row


def predict_row_for_file(p: Path, model: torch.nn.Module) -> dict[str, object]:
    image = Image.open(p).convert("RGB")
    prepped = preprocess_fundus_effnet_b4_g(image)
    return predict_row_from_prepped(model, prepped)


def next_file_needing_predictions(
    files: list[Path],
    model_names: list[str],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> Path | None:
    """First image in **files** order that still needs at least one model."""
    for p in files:
        fp = str(p.resolve())
        if any(fp not in deploy_preds.get(n, {}) for n in model_names):
            return p
    return None


def run_deploy_predictions_for_one_file(
    p: Path,
    model_names: list[str],
    weights_by_name: dict[str, Path],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> None:
    """Fill all missing **model** rows for **one** file (shared preproc)."""
    fp = str(p.resolve())
    missing = [n for n in model_names if fp not in deploy_preds.get(n, {})]
    if not missing:
        return

    model_by_name: dict[str, torch.nn.Module | None] = {}
    for n in model_names:
        try:
            model_by_name[n] = load_effnet_b4_g(str(weights_by_name[n].resolve()))
        except Exception:
            model_by_name[n] = None

    try:
        pil_o = Image.open(p).convert("RGB")
        prepped = preprocess_fundus_effnet_b4_g(pil_o)
    except OSError as e:
        err = {"inference_status": "error", "error": str(e)}
        for n in missing:
            deploy_preds.setdefault(n, {})[fp] = dict(err)
        return

    for n in missing:
        model = model_by_name[n]
        if model is None:
            deploy_preds.setdefault(n, {})[fp] = {
                "inference_status": "error",
                "error": "failed to load model weights",
            }
        else:
            try:
                deploy_preds.setdefault(n, {})[fp] = predict_row_from_prepped(model, prepped)
            except Exception as e:
                deploy_preds.setdefault(n, {})[fp] = {
                    "inference_status": "error",
                    "error": str(e),
                }


def _result_public_label(internal_status: str) -> str:
    if internal_status == "done":
        return "success"
    if internal_status == "error":
        return "failed"
    return "pending"


def merge_inventory_with_preds(inventory: pd.DataFrame, preds: dict[str, dict[str, object]]) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for _, r in inventory.iterrows():
        fp = str(r["full_path"])
        pr = preds.get(fp, {})
        status = pr.get("inference_status", "pending")
        row: dict[str, object] = {
            "relative_path": r["relative_path"],
            "class_from_path": r["class_from_path"],
            "result": _result_public_label(str(status)),
            "full_path": fp,
        }
        if status == "done":
            row["predicted_class"] = pr.get("predicted_class")
            for cname in CLASS_NAMES:
                row[f"P({cname})"] = pr.get(f"P({cname})")
        elif status == "error":
            row["predicted_class"] = "—"
            row["error"] = pr.get("error", "failed")
            for cname in CLASS_NAMES:
                row[f"P({cname})"] = None
        else:
            row["predicted_class"] = "—"
            for cname in CLASS_NAMES:
                row[f"P({cname})"] = None
        out_rows.append(row)
    return pd.DataFrame(out_rows)


PREVIEW_IMAGE_WIDTH = 180
COMPARE_ROW_IMG_WIDTH = 130
DATAFRAME_VIEW_HEIGHT = 480


def _softmax_items_sorted(pr: dict[str, object]) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for cname in CLASS_NAMES:
        key = f"P({cname})"
        raw = pr.get(key)
        if raw is None:
            continue
        try:
            items.append((cname, float(raw)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda x: -x[1])
    return items


def format_model_cell_plain(bucket: dict[str, dict[str, object]], fp: str, actual: str) -> str:
    """Plain text for **st.dataframe**: pending / error / **CA:12% · NR:80% · …** (softmax, high→low)."""
    pr = bucket.get(fp, {})
    status = str(pr.get("inference_status", "pending"))
    if status == "pending":
        return "pending"
    if status == "error":
        return "error · fail"
    items = _softmax_items_sorted(pr)
    if not items:
        return str(pr.get("predicted_class", "—"))
    return " · ".join(f"{CLASS_CODE_UI[c]}:{p * 100:.1f}%" for c, p in items)


def render_compare_probability_cell(bucket: dict[str, dict[str, object]], fp: str, actual: str) -> None:
    """**probabilities** column: same **text color / weight** rules as **Result by CLASS** (Styler)."""
    pr = bucket.get(fp, {})
    cell_txt = format_model_cell_plain(bucket, fp, actual)
    css = _model_cell_style_css(pr, actual)
    st.markdown(
        f'<p style="{css}; margin:0; line-height:1.35; font-size:0.9rem;">'
        f"{html.escape(cell_txt)}</p>",
        unsafe_allow_html=True,
    )


def _model_cell_style_css(pr: dict[str, object], actual: str) -> str:
    """CSS for model columns only: **color** / **font-weight** (top-1 vs **actual**); no background."""
    status = str(pr.get("inference_status", "pending"))
    if status == "pending":
        return "color: #757575"
    if status == "error":
        return "color: #b71c1c; font-weight: 600"
    items = _softmax_items_sorted(pr)
    if not items:
        return "color: #616161"
    max_p = items[0][1]
    tops = {c for c, p in items if abs(p - max_p) < 1e-9}
    if actual in CLASS_NAMES:
        if tops & {actual}:
            return "color: #1b5e20; font-weight: 600"
        return "color: #c62828; font-weight: 600"
    return "color: #424242"


def confusion_matrix_labeled(
    files: list[Path],
    inventory: pd.DataFrame,
    bucket: dict[str, dict[str, object]],
) -> np.ndarray:
    """Counts[actual_idx, pred_idx] for files with **actual** in **CLASS_NAMES** and inference **done**."""
    n = len(CLASS_NAMES)
    mat = np.zeros((n, n), dtype=float)
    inv_by = {str(r["full_path"]): dict(r) for _, r in inventory.iterrows()}
    for p in files:
        fp = str(p.resolve())
        inv = inv_by.get(fp, {})
        acl = assigned_label_class(inv)
        if acl == "unlabeled":
            continue
        pr = bucket.get(fp, {})
        if str(pr.get("inference_status")) != "done":
            continue
        pred = str(pr.get("predicted_class", ""))
        if pred not in CLASS_NAMES:
            continue
        i = CLASS_NAMES.index(acl)
        j = CLASS_NAMES.index(pred)
        mat[i, j] += 1.0
    return mat


def per_class_recall_by_model_dataframe(
    model_names: list[str],
    files: list[Path],
    inventory: pd.DataFrame,
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> pd.DataFrame:
    """Rows = **actual** class; one column per model = recall (%); **n_actual** = count in batch."""
    inv_by = {str(r["full_path"]): dict(r) for _, r in inventory.iterrows()}
    mats = {m: confusion_matrix_labeled(files, inventory, deploy_preds.get(m, {})) for m in model_names}
    rows: list[dict[str, object]] = []
    for ci, cname in enumerate(CLASS_NAMES):
        row: dict[str, object] = {"actual": cname}
        n_act = sum(
            1
            for p in files
            if assigned_label_class(inv_by.get(str(p.resolve()), {})) == cname
        )
        row["n_actual"] = int(n_act)
        for m in model_names:
            mat = mats[m]
            rs = float(mat[ci, :].sum())
            if rs > 0:
                row[m] = round(100.0 * float(mat[ci, ci]) / rs, 1)
            else:
                row[m] = None
        rows.append(row)
    return pd.DataFrame(rows)


def render_confusion_heatmap(
    mat: np.ndarray,
    title: str,
    *,
    size: str = "compact",
) -> None:
    """Fixed logical size + no container stretch. **size**: full | compact | mini (left column in model summary)."""
    n = len(CLASS_NAMES)
    if size == "mini":
        cell = 0.5
        w = max(3.5, n * cell + 0.95)
        h = max(3.0, n * cell + 0.85)
        dpi = 88
        ann_sz, title_sz, tick_sz = 7, 9, 6
        cbar_shrink = 0.5
    elif size == "compact":
        cell = 0.88
        w = max(5.2, n * cell + 1.5)
        h = max(4.4, n * cell + 1.35)
        dpi = 96
        ann_sz, title_sz, tick_sz = 9, 11, 8
        cbar_shrink = 0.78
    else:
        cell = 1.35
        w = max(8.0, n * cell + 2.2)
        h = max(6.5, n * cell + 2.0)
        dpi = 100
        ann_sz, title_sz, tick_sz = 12, 13, 10
        cbar_shrink = 0.82
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi, layout="constrained")
    sns.heatmap(
        mat,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"label": "count", "shrink": cbar_shrink},
        linewidths=0.5,
        linecolor="0.85",
        annot_kws={"size": ann_sz},
        square=True,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=title_sz)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=tick_sz)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_sz)
    st.pyplot(fig, width="content")
    plt.close(fig)


def model_summary_table(
    model_names: list[str],
    files: list[Path],
    inventory: pd.DataFrame,
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> pd.DataFrame:
    """Per model: run stats; **labeled** / **correct** / **accuracy** use **actual** vs prediction."""
    file_resolved = [str(p.resolve()) for p in files]
    n = len(file_resolved)
    inv_by = {str(r["full_path"]): dict(r) for _, r in inventory.iterrows()}
    out: list[dict[str, object]] = []
    for m in model_names:
        bucket = deploy_preds.get(m, {})
        pending = sum(1 for fr in file_resolved if fr not in bucket)
        infer_ok = 0
        infer_fail = 0
        labeled_done = 0
        correct_vs_label = 0
        for fr in file_resolved:
            pr = bucket.get(fr, {})
            if not pr:
                continue
            stt = str(pr.get("inference_status", "pending"))
            if stt == "done":
                infer_ok += 1
            elif stt == "error":
                infer_fail += 1
            if stt != "done":
                continue
            inv = inv_by.get(fr, {})
            acl = assigned_label_class(inv)
            if acl == "unlabeled":
                continue
            labeled_done += 1
            pred = str(pr.get("predicted_class", ""))
            if pred == acl:
                correct_vs_label += 1
        acc = (correct_vs_label / labeled_done) if labeled_done > 0 else None
        out.append(
            {
                "model": m,
                "pending": pending,
                "inference_ok": infer_ok,
                "inference_err": infer_fail,
                "labeled": labeled_done,
                "correct": correct_vs_label,
                "accuracy": round(float(acc), 4) if acc is not None else None,
                "total": n,
            }
        )
    return pd.DataFrame(out)


def render_compare_blocks(
    paths: list[str],
    inventory: pd.DataFrame,
    deploy_preds: dict[str, dict[str, dict[str, object]]],
    model_names: list[str],
    *,
    img_width: int = COMPARE_ROW_IMG_WIDTH,
) -> None:
    """One horizontal row per (file, model): path, **actual**, model, softmax **probabilities**, images."""
    compare_hint = st.empty()
    if not paths:
        compare_hint.info(
            "**Nothing selected.** In **Result by CLASS**, select one or more rows in the table (multi-select), "
            "then open the **PREVIEW** tab under **Summary** to see each chosen image with **original** and **preprocessed** views "
            "and every model’s **probabilities**, with the same text colors as the table."
        )
        return
    compare_hint.empty()
    inv_map: dict[str, dict[str, object]] = {
        str(r["full_path"]): dict(r) for _, r in inventory.iterrows()
    }
    hdr = st.columns([2.0, 1.0, 1.2, 2.2, 1.0, 1.0])
    headers = ("relative path", "actual", "model", "probabilities", "original", "preproc")
    for col, label in zip(hdr, headers):
        with col:
            st.markdown(f"**{label}**")
    st.divider()

    for fp in paths:
        p = Path(fp)
        inv = inv_map.get(fp, {})
        rel = str(inv.get("relative_path", "—"))
        gt = assigned_label_class(inv)

        pil_o: Image.Image | None = None
        pil_p: Image.Image | None = None
        if p.is_file():
            try:
                pil_o = Image.open(p).convert("RGB")
                pil_p = preprocess_fundus_effnet_b4_g(pil_o)
            except OSError:
                pil_o = pil_p = None
        else:
            st.warning(f"Missing file: `{p}`")

        for m in model_names:
            c0, c1, c2, c3, c4, c5 = st.columns([2.0, 1.0, 1.2, 2.2, 1.0, 1.0])
            with c0:
                st.text(rel)
            with c1:
                st.text(gt)
            with c2:
                st.text(m)
            with c3:
                render_compare_probability_cell(deploy_preds.get(m, {}), fp, gt)
            with c4:
                if pil_o is not None:
                    st.image(pil_o, width=img_width)
                else:
                    st.caption("—")
            with c5:
                if pil_p is not None:
                    st.image(pil_p, width=img_width)
                else:
                    st.caption("—")
        st.divider()


def assigned_label_class(inv_row: dict[str, object]) -> str:
    """**Actual** label: known **CLASS_NAMES** or **unlabeled**."""
    c = str(inv_row.get("class_from_path", "—"))
    if c in CLASS_NAMES:
        return c
    return "unlabeled"


def build_multi_model_row(
    fp: str,
    rel: str,
    actual: str,
    deploy_preds: dict[str, dict[str, dict[str, object]]],
    model_names: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {"relative_path": rel, "actual": actual, "full_path": fp}
    for m in model_names:
        row[m] = format_model_cell_plain(deploy_preds.get(m, {}), fp, actual)
    return row


def pivot_for_files(
    inventory: pd.DataFrame,
    deploy_preds: dict[str, dict[str, dict[str, object]]],
    model_names: list[str],
    *,
    class_filter: str | None,
) -> pd.DataFrame:
    """Rows = files; **actual** column; model columns = plain **CA:% · …** (colors via Styler)."""
    rows: list[dict[str, object]] = []
    for _, r in inventory.iterrows():
        invd = dict(r)
        acl = assigned_label_class(invd)
        if class_filter is not None and acl != class_filter:
            continue
        fp = str(r["full_path"])
        rel = str(r["relative_path"])
        rows.append(build_multi_model_row(fp, rel, acl, deploy_preds, model_names))
    return pd.DataFrame(rows)


def pivot_display_columns(pivot: pd.DataFrame, model_names: list[str]) -> list[str]:
    cols = ["relative_path", "actual"]
    cols.extend(m for m in model_names if m in pivot.columns)
    return cols


def compare_full_paths_from_dataframe_event(pivot: pd.DataFrame, event) -> list[str]:
    """**full_path** values for the current **st.dataframe** selection (one class tab)."""
    sel = event.selection
    sel_rows = list(sel.rows) if sel is not None and sel.rows else []
    if not sel_rows or pivot.empty:
        return []
    out: list[str] = []
    for ri in sorted(sel_rows, key=int):
        ii = int(ri)
        if 0 <= ii < len(pivot):
            out.append(str(Path(pivot.iloc[ii]["full_path"])))
    return out


def merge_compare_paths_unique(groups: list[list[str]]) -> list[str]:
    """Stable de-dupe: first occurrence wins (order across class tabs left-to-right)."""
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for fp in g:
            if fp not in seen:
                seen.add(fp)
                out.append(fp)
    return out


def style_results_pivot(
    pivot: pd.DataFrame,
    model_names: list[str],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> pd.io.formats.style.Styler:
    """**st.dataframe** with default cell backgrounds; model columns tinted by **text color** only."""
    show_cols = pivot_display_columns(pivot, model_names)
    show_df = pivot.loc[:, show_cols]
    styler = show_df.style
    for m in model_names:
        if m not in show_df.columns:
            continue

        def make_col_styler(mn: str):
            def _apply(col: pd.Series) -> list[str]:
                out: list[str] = []
                for idx in col.index:
                    fp = str(pivot.loc[idx, "full_path"])
                    act = str(pivot.loc[idx, "actual"])
                    pr = deploy_preds.get(mn, {}).get(fp, {})
                    out.append(_model_cell_style_css(pr, act))
                return out

            return _apply

        styler = styler.apply(make_col_styler(m), axis=0, subset=[m])
    return styler


def main() -> None:
    st.set_page_config(
        page_title="Eye disease — multi-model",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_deploy_like_css()

    if not MODEL_LIST:
        st.error("**MODEL_LIST** is empty. Add at least one model entry.")
        st.stop()

    names = [str(e["name"]) for e in MODEL_LIST]
    if len(names) != len(set(names)):
        st.error("Duplicate model **name** values in **MODEL_LIST**.")
        st.stop()

    weights_by_name: dict[str, Path] = {}
    tried_by_missing: dict[str, list[Path]] = {}
    for entry in MODEL_LIST:
        name = str(entry["name"])
        explicit = entry.get("weights")
        explicit_s = str(explicit) if explicit else None
        wp, tried = resolve_weights_path(name, explicit_s)
        if wp is not None:
            weights_by_name[name] = wp
        else:
            tried_by_missing[name] = tried

    missing = [n for n in names if n not in weights_by_name]
    if missing:
        lines: list[str] = []
        for n in missing:
            tried = tried_by_missing.get(n, [])
            lines.append(f"**{n}** — tried:\n" + "\n".join(f"- `{p}`" for p in tried))
        st.error(
            "Missing weights for: **"
            + "**, **".join(missing)
            + "**.\n\n"
            + "\n\n".join(lines)
        )
        st.stop()

    first_model = load_effnet_b4_g(str(weights_by_name[names[0]]))
    device = next(first_model.parameters()).device

    if "compare_paths" not in st.session_state:
        st.session_state.compare_paths = []
    if "deploy_meta" not in st.session_state:
        st.session_state.deploy_meta = None
    if "deploy_preds" not in st.session_state:
        st.session_state.deploy_preds = {}

    st.title("Eye Disease Classification")
    models_list = " · ".join(f"**{n}**" for n in names)
    st.caption(f"Models: {models_list}. Device: `{device}`")

    with st.expander("**Setup** — paths & weights", expanded=True):
        with st.expander("Resolved weight files"):
            for name in names:
                st.markdown(f"**{name}**")
                st.code(str(weights_by_name[name]), language="text")

        path_raw = st.text_input(
            "Image folder or image file path",
            key=BATCH_PATH_INPUT_KEY,
            placeholder=r"Folder (scanned recursively) or one image file, e.g. C:\data\fundus or C:\data\a.png",
            help="Paste or type a full path. Folders: all images under it are included. Files: that image only.",
        )
        path_raw = (path_raw or "").strip()
        path_obj = path_from_user_input(path_raw) if path_raw else None
        path_norm = normalize_user_path(path_raw) if path_raw else ""

        path_feedback = st.empty()
        if st.session_state.get(PATH_FEEDBACK_STATE_KEY) != path_norm:
            st.session_state[PATH_FEEDBACK_STATE_KEY] = path_norm
            path_feedback.empty()

        folder: Path | None = None
        files: list[Path] = []
        folder_ok = False
        single_file_mode = False

        if path_obj is not None:
            if not path_obj.exists():
                with path_feedback.container():
                    st.warning("Path does not exist.")
                    st.caption(f"Checked: `{path_obj}`")
            elif path_obj.is_dir():
                folder = path_obj
                files = list_image_files(folder)
                folder_ok = True
            elif path_obj.is_file():
                if path_obj.suffix.lower() in IMAGE_EXT:
                    folder = path_obj.parent
                    files = [path_obj.resolve()]
                    folder_ok = True
                    single_file_mode = True
                else:
                    with path_feedback.container():
                        st.warning("File type is not a supported image.")
            else:
                with path_feedback.container():
                    st.warning("Path is not a valid folder or file.")

        if folder_ok and single_file_mode and files:
            one = files[0]
            try:
                image = Image.open(one).convert("RGB")
            except OSError as e:
                st.error(f"Could not open file: {e}")
                folder_ok = False
            else:
                prepped = preprocess_fundus_effnet_b4_g(image)
                st.caption(str(one.resolve()))
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.image(image, width=PREVIEW_IMAGE_WIDTH)
                with c2:
                    st.caption("Preprocessed")
                    st.image(prepped, width=PREVIEW_IMAGE_WIDTH)
                m0 = load_effnet_b4_g(str(weights_by_name[names[0]]))
                pred_idx, probs = predict_proba(m0, prepped)
                st.success(f"**{CLASS_NAMES[pred_idx]}** ({names[0]})")
                for cname, p in zip(CLASS_NAMES, probs):
                    st.progress(float(p), text=f"{cname}: {float(p) * 100:.1f}%")

        if folder_ok and not single_file_mode and not files:
            with path_feedback.container():
                st.info("Folder has no image files yet.")

    if not folder_ok:
        st.caption("Enter a valid **folder** or **image file** path in **Setup** to load results.")
        return

    if not files:
        st.caption("No images to process for this path.")
        return

    assert folder is not None
    sig = batch_files_signature(files)
    meta_key = f"{folder.resolve()}|{sig}"
    mk_slug = meta_key_slug(meta_key)
    if st.session_state.deploy_meta != meta_key:
        st.session_state.deploy_meta = meta_key
        st.session_state.deploy_preds = {}
        st.session_state.compare_paths = []
        st.session_state.pop(COMPARE_PATHS_SIG_KEY, None)
        st.session_state.pop(SUMMARY_TAB_PICK_KEY, None)

    inventory = build_file_inventory_dataframe(folder, files)
    if inventory.empty:
        st.info("No rows.")
        return

    dp = st.session_state.deploy_preds

    present_classes = {assigned_label_class(dict(r)) for _, r in inventory.iterrows()}
    class_tab_order = [c for c in CLASS_NAMES if c in present_classes]
    if "unlabeled" in present_classes:
        class_tab_order.append("unlabeled")

    st.subheader("Result by CLASS")
    compare_selection_groups: list[list[str]] = []
    if not class_tab_order:
        st.caption("No files to classify.")
    else:
        class_tab_display = [c.upper() for c in class_tab_order]
        class_tabs = st.tabs(class_tab_display)
        for tab, cls in zip(class_tabs, class_tab_order):
            with tab:
                pivot = pivot_for_files(inventory, dp, names, class_filter=cls)
                if pivot.empty:
                    st.caption("No files in this class.")
                    continue
                st.caption(
                    f"**{cls}** — {len(pivot)} file(s). "
                    "**CA** cataract · **DR** diabetic_retinopathy · **GL** glaucoma · **NR** normal."
                )
                styled = style_results_pivot(pivot, names, dp)
                ev = st.dataframe(
                    styled,
                    width="stretch",
                    height=DATAFRAME_VIEW_HEIGHT,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=f"df_class_{cls}_{mk_slug}",
                )
                compare_selection_groups.append(compare_full_paths_from_dataframe_event(pivot, ev))
    st.session_state.compare_paths = merge_compare_paths_unique(compare_selection_groups)

    st.subheader("Summary")
    summ_df = model_summary_table(names, files, inventory, dp)
    sum_labels = list(names)
    if len(names) > 1:
        sum_labels.append("all")
    sum_labels.append(SUMMARY_TAB_PREVIEW)
    sum_tab_display = [x.upper() for x in sum_labels]
    display_to_label = dict(zip(sum_tab_display, sum_labels))

    paths_now = list(st.session_state.get("compare_paths") or [])
    sig_now = tuple(sorted(paths_now))
    sig_prev = st.session_state.get(COMPARE_PATHS_SIG_KEY)
    if sig_prev != sig_now:
        st.session_state[COMPARE_PATHS_SIG_KEY] = sig_now
        if sig_prev is not None:
            st.session_state[SUMMARY_TAB_PICK_KEY] = SUMMARY_TAB_PREVIEW.upper()

    sel = st.segmented_control(
        " ",
        options=sum_tab_display,
        selection_mode="single",
        label_visibility="collapsed",
        key=SUMMARY_TAB_PICK_KEY,
        width="stretch",
    )
    picked = _summary_segmented_pick(
        sel,
        st.session_state.get(SUMMARY_TAB_PICK_KEY),
        sum_tab_display,
    )
    label = display_to_label[picked]

    if label == "all":
        st.caption("Cross-model summary table and **per-class recall**; confusion heatmaps are on each model tab.")
        st.dataframe(summ_df, width="stretch", hide_index=True)
        st.markdown("##### Per-class accuracy by model (recall, %)")
        pc_df = per_class_recall_by_model_dataframe(names, files, inventory, dp)
        st.dataframe(pc_df, width="stretch", hide_index=True)
        st.caption(
            "Each model column: **%** recall per **actual** (row total in that model’s confusion matrix, "
            "finished valid predictions). **n_actual** = images with that **actual** in the batch."
        )
    elif label == SUMMARY_TAB_PREVIEW:
        render_compare_blocks(
            paths_now,
            inventory,
            dp,
            names,
        )
    else:
        row = summ_df[summ_df["model"] == label]
        if row.empty:
            st.caption("No data.")
        else:
            r = row.iloc[0]
            mat = confusion_matrix_labeled(files, inventory, dp.get(label, {}))
            hm_col, stats_col = st.columns([0.36, 0.64], gap="small")
            with hm_col:
                render_confusion_heatmap(
                    mat,
                    f"{label}\nactual × predicted",
                    size="mini",
                )
            with stats_col:
                if int(r["total"]) > 0:
                    done = int(r["inference_ok"]) + int(r["inference_err"])
                    st.progress(
                        min(done / int(r["total"]), 1.0),
                        text=f"Inference: {done} / {int(r['total'])} files",
                    )
                st.caption(
                    "**Correct** and **Accuracy**: vs **actual**. "
                    f"Runs: {int(r['inference_ok'])} ok, {int(r['inference_err'])} errors."
                )
                a, b, c = st.columns(3)
                with a:
                    st.metric("Total files", int(r["total"]))
                with b:
                    st.metric("Pending", int(r["pending"]))
                with c:
                    st.metric("Labeled", int(r["labeled"]))
                d, e = st.columns(2)
                with d:
                    st.metric("Correct", int(r["correct"]))
                acc = r["accuracy"]
                with e:
                    if pd.isna(acc):
                        st.metric("Accuracy", "—")
                    else:
                        st.metric("Accuracy", f"{float(acc) * 100:.1f}%")

    p_run = next_file_needing_predictions(files, names, dp)
    if p_run is not None:
        run_deploy_predictions_for_one_file(p_run, names, weights_by_name, dp)
        st.rerun()


if __name__ == "__main__":
    main()
