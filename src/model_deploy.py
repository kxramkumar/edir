"""
Multi-model Streamlit deploy: **ResNet-50**, **DenseNet-121**, **EfficientNet-B0** using the same **build_model** and
**eval_transform** as ``final_eye_disease_classification.ipynb`` (torchvision **Resize** + **ToTensor** + **Normalize**).

Weights: set **``weights``** in **MODEL_LIST** (absolute path, or relative to this file’s directory —
**``src/``** when the script lives in **``src/``**). If **``weights``** is omitted, resolution falls back to:

1. Env **EDIR_<NAME_UPPER>_WEIGHTS** (absolute or repo-relative from **REPO_ROOT**).
2. **mode/<name>/<name>.pth**
3. **res/<RUN_TAG>/data/model/<name>/<name>.pth**

Layout: **Setup** (path) → **Result by CLASS** (updates after **each** file’s inference) → **Summary** only when the full batch is done. Results render inside one **fragment**; while inference is incomplete the fragment chains with ``st.rerun(scope="fragment")`` (no ``run_every`` timer), so the app **stops running** when the batch finishes.

Run from repo root::

    uv sync
    uv run streamlit run src/model_deploy.py
"""

from __future__ import annotations

import hashlib
import html
import os
from pathlib import Path

# PyTorch + NumPy BLAS can oversubscribe CPU threads on Windows; cap before heavy work.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit.errors import StreamlitAPIException

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
    {"name": "resnet_50", "weights": str(DEPLOY_DIR / "resnet_50.pth")},
    {"name": "densenet_121", "weights": str(DEPLOY_DIR / "densenet_121.pth")},
    {"name": "efficientnet_b0", "weights": str(DEPLOY_DIR / "efficientnet_b0.pth")},
]

# Same symbols as ``final_eye_disease_classification.ipynb`` (IMAGE_SIZE, DROPOUT).
IMAGE_SIZE = 224
DROPOUT = 0.3

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
CLASS_TAB_PICK_KEY = "result_class_tab_pick"
# Last **Result by CLASS** segment; switching clears that table’s selection and **Preview** paths for a clean reload.
LAST_RESULT_CLASS_KEY = "_last_result_class_tab"
# **Summary** tab body uses ``st.container(key=...)`` so switching tabs remounts content cleanly.
FRAG_INVENTORY_KEY = "_edir_frag_inventory"
FRAG_NAMES_KEY = "_edir_frag_names"
FRAG_MK_KEY = "_edir_frag_mk_slug"
FRAG_FILES_KEY = "_edir_frag_files"
FRAG_WEIGHTS_KEY = "_edir_frag_weights"
# **Summary** static tabs (ALL + per-model): cached while batch + preds unchanged; **PREVIEW** never uses this.
SUMMARY_STATIC_CACHE_KEY = "_edir_summary_static_fp"
SUMMARY_SUMM_DF_KEY = "_edir_summary_summ_df"
SUMMARY_PC_DF_KEY = "_edir_summary_pc_df"
SUMMARY_CM_STORE_KEY = "_edir_summary_cm_store"

IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Matches ``eval_transform`` in ``final_eye_disease_classification.ipynb``: default ``Resize(224)`` + ``ToTensor`` + ``Normalize``.
EVAL_RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
EVAL_TRANSFORM = transforms.Compose(
    [
        EVAL_RESIZE,
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


def _ensure_segmented_control_default(key: str, valid_display: list[str]) -> None:
    """If **st.session_state**[key] is missing or not in **valid_display**, set it to the first option (default tab)."""
    if not valid_display:
        return
    valid = set(valid_display)
    default = valid_display[0]
    raw = st.session_state.get(key)
    if raw is None:
        st.session_state[key] = default
        return
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0 or str(raw[0]) not in valid:
            st.session_state[key] = default
        return
    if str(raw) not in valid:
        st.session_state[key] = default


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


def validate_folder_has_class_layout(folder: Path, files: list[Path]) -> tuple[bool, str]:
    """Batch folders must follow ImageFolder-style paths: each image under a directory named in **CLASS_NAMES**."""
    if not files:
        return (
            False,
            "No images found. Expected subfolders named exactly: **"
            + "**, **".join(CLASS_NAMES)
            + "** (e.g. `…/cataract/img.png`), or use a **single image file** path.",
        )
    root = folder.resolve()
    bad: list[str] = []
    for p in files:
        try:
            rel_parts = p.resolve().relative_to(root).parts
        except ValueError:
            return False, "Could not resolve file paths relative to the selected folder."
        dir_parts = rel_parts[:-1]
        if not any(part in CLASS_NAMES for part in dir_parts):
            bad.append(str(p))
    if not bad:
        return True, ""
    show = bad[:3]
    tail = f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
    return (
        False,
        "Each image must be inside a subfolder named one of: **"
        + "**, **".join(CLASS_NAMES)
        + "**. "
        + f"{len(bad)} file(s) are not (examples: "
        + ", ".join(f"`{s}`" for s in show)
        + tail
        + "). Or choose **one image file** instead of a folder.",
    )


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
/* Tabs (Result by CLASS, etc.): left-aligned strip, not stretched to full main width */
div[data-testid="stTabs"] {
    width: fit-content !important;
    max-width: 100% !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"],
div[data-testid="stTabs"] [role="tablist"] {
    width: fit-content !important;
    justify-content: flex-start !important;
}
/* Summary panel switcher (segmented_control key summary_tab_pick): compact, left */
div.st-key-summary_tab_pick {
    width: fit-content !important;
    max-width: 100% !important;
}
/* Result by CLASS switcher (segmented_control key result_class_tab_pick): same as Summary */
div.st-key-result_class_tab_pick {
    width: fit-content !important;
    max-width: 100% !important;
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


def eval_transform_pil(pil_rgb: Image.Image) -> Image.Image:
    """PIL 224×224 for **display** only; same **EVAL_RESIZE** as inference."""
    return EVAL_RESIZE(pil_rgb.convert("RGB"))


def build_model(model_name: str, num_classes: int, dropout: float = DROPOUT) -> torch.nn.Module:
    """Same as ``build_model`` in ``final_eye_disease_classification.ipynb`` (``weights=None`` for deploy)."""
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def extract_state_dict_from_checkpoint(loaded: object, *, path: Path) -> dict[str, torch.Tensor]:
    """``.pth`` may be a raw ``state_dict`` or a dict with ``model_state_dict`` / ``state_dict``."""
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        sd = loaded["model_state_dict"]
    elif isinstance(loaded, dict) and "state_dict" in loaded:
        sd = loaded["state_dict"]
    elif isinstance(loaded, dict) and loaded:
        k0 = next(iter(loaded.keys()))
        if isinstance(k0, str) and (k0.startswith("features.") or k0.startswith("classifier.")):
            return loaded  # type: ignore[return-value]
        raise ValueError(f"Unrecognized checkpoint keys in {path}")
    else:
        raise ValueError(f"Expected a dict checkpoint: {path}")
    if not isinstance(sd, dict):
        raise ValueError(f"state dict is not a dict: {path}")
    return sd  # type: ignore[return-value]


def _notebook_build_model_name(deploy_name: str) -> str:
    """Map **MODEL_LIST** ``name`` to ``build_model`` ``model_name`` (``efficientnet_b0`` / ``densenet121`` / ``resnet50``)."""
    n = deploy_name.strip().lower().replace("-", "_")
    if n in ("resnet_50", "resnet50"):
        return "resnet50"
    if n in ("densenet_121", "densenet121"):
        return "densenet121"
    if n in ("efficientnet_b0", "effnet_b0"):
        return "efficientnet_b0"
    raise ValueError(f"Unknown model name (expected resnet_50, densenet_121, efficientnet_b0): {deploy_name!r}")


@st.cache_resource
def load_model_from_checkpoint(weights_path: str, model_name: str) -> torch.nn.Module:
    """``torch.load`` + ``model_state_dict`` into ``build_model`` — same pattern as loading best checkpoints in the notebook."""
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = _notebook_build_model_name(model_name)
    model = build_model(key, NUM_CLASSES, DROPOUT)
    for param in model.parameters():
        param.requires_grad = False
    raw = torch.load(str(path), map_location=device, weights_only=False)
    state = extract_state_dict_from_checkpoint(raw, path=path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_model_for_name(model_name: str, weights_path: str) -> torch.nn.Module:
    wp = str(Path(weights_path).resolve())
    return load_model_from_checkpoint(wp, model_name)


def device_from_loaded(loaded: torch.nn.Module) -> torch.device:
    return next(loaded.parameters()).device


def predict_proba(model: torch.nn.Module, x: torch.Tensor) -> tuple[int, np.ndarray]:
    """``x``: batch **(1, 3, H, W)** from **EVAL_TRANSFORM**(PIL RGB)."""
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def predict_deploy_row(loaded: torch.nn.Module, x: torch.Tensor) -> dict[str, object]:
    pred_idx, probs = predict_proba(loaded, x)
    row: dict[str, object] = {
        "inference_status": "done",
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
    }
    for cname, prob in zip(CLASS_NAMES, probs):
        row[f"P({cname})"] = float(prob)
    return row


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


def predict_row_for_file(model_name: str, p: Path, weights_path: str) -> dict[str, object]:
    loaded = load_model_for_name(model_name, str(weights_path))
    image = Image.open(p).convert("RGB")
    x = EVAL_TRANSFORM(image).unsqueeze(0)
    return predict_deploy_row(loaded, x)


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


def inference_files_completed(
    files: list[Path],
    model_names: list[str],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> tuple[int, int]:
    """Files that have an entry for **every** model (``done`` or ``error``). Returns ``(completed, total)``."""
    n = len(files)
    if n == 0:
        return 0, 0
    c = 0
    for p in files:
        fp = str(p.resolve())
        if all(fp in deploy_preds.get(m, {}) for m in model_names):
            c += 1
    return c, n


def _summary_static_fingerprint(
    mk_slug: str,
    files: list[Path],
    model_names: list[str],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> str:
    """Stable hash for **Summary** tables that only depend on batch + predictions (not row selection)."""
    lines: list[str] = [mk_slug]
    for p in sorted(files, key=lambda x: str(x).lower()):
        fp = str(p.resolve())
        lines.append(fp)
        for m in model_names:
            pr = deploy_preds.get(m, {}).get(fp, {})
            stt = str(pr.get("inference_status", ""))
            pred = str(pr.get("predicted_class", ""))
            lines.append(f"{m}:{stt}:{pred}")
    return hashlib.sha256("\n".join(lines).encode("utf-8", errors="replace")).hexdigest()


def run_deploy_predictions_for_one_file(
    p: Path,
    model_names: list[str],
    weights_by_name: dict[str, Path],
    deploy_preds: dict[str, dict[str, dict[str, object]]],
) -> None:
    """Fill all missing **model** rows for **one** file (shared RGB 224×224 prep per file)."""
    fp = str(p.resolve())
    missing = [n for n in model_names if fp not in deploy_preds.get(n, {})]
    if not missing:
        return

    loaded_by_name: dict[str, torch.nn.Module | None] = {}
    for n in model_names:
        try:
            loaded_by_name[n] = load_model_for_name(n, str(weights_by_name[n].resolve()))
        except Exception:
            loaded_by_name[n] = None

    try:
        pil_o = Image.open(p).convert("RGB")
    except OSError as e:
        err = {"inference_status": "error", "error": str(e)}
        for n in missing:
            deploy_preds.setdefault(n, {})[fp] = dict(err)
        return

    x_batch = EVAL_TRANSFORM(pil_o).unsqueeze(0)

    for n in missing:
        loaded = loaded_by_name[n]
        if loaded is None:
            deploy_preds.setdefault(n, {})[fp] = {
                "inference_status": "error",
                "error": "failed to load model weights",
            }
        else:
            try:
                deploy_preds.setdefault(n, {})[fp] = predict_deploy_row(loaded, x_batch)
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
        msg = str(pr.get("error", "failed")).replace("\n", " ").strip()
        if len(msg) > 100:
            msg = msg[:97] + "…"
        return f"error · {msg}"
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
    if not paths:
        st.info(
            "**Nothing selected.** In **Result by CLASS**, open the class you want, multi-select rows in **that** table only, "
            "then open **PREVIEW** under **Summary** (images, preproc, per-model **probabilities** — same colors as the table). "
            "Switching class clears that table’s selection and starts fresh for **Preview**."
        )
        return
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
        if p.is_file():
            try:
                pil_o = Image.open(p).convert("RGB")
            except OSError:
                pil_o = None
        else:
            st.warning(f"Missing file: `{p}`")

        pil_p = eval_transform_pil(pil_o) if pil_o is not None else None
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


def _class_segment_display_for_path(
    p: Path,
    inventory: pd.DataFrame,
    class_tab_order: list[str],
) -> str | None:
    """``st.segmented_control`` label (e.g. ``CATARACT``) for **p**'s class, or **None** if not in tabs."""
    fp = str(p.resolve())
    m = inventory[inventory["full_path"] == fp]
    if not m.empty:
        acl = assigned_label_class(dict(m.iloc[0]))
    else:
        ic = infer_class_from_path(str(p))
        acl = ic if ic in CLASS_NAMES else "unlabeled"
    if acl not in class_tab_order:
        return None
    return acl.upper()


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


def _event_row_indices(event: object) -> tuple[int, ...]:
    """Stable row-index tuple from **st.dataframe** ``on_select`` event (for change detection)."""
    if event is None:
        return ()
    sel = getattr(event, "selection", None)
    if sel is None:
        return ()
    rows = getattr(sel, "rows", None)
    if not rows:
        return ()
    try:
        return tuple(sorted(int(x) for x in list(rows)))
    except (TypeError, ValueError):
        return ()


def _dataframe_row_indices(widget_key: str, event: object) -> tuple[int, ...]:
    """Row selection: prefer **st.session_state** (authoritative) then event object."""
    raw = st.session_state.get(widget_key)
    if isinstance(raw, dict):
        sel = raw.get("selection")
        if isinstance(sel, dict):
            rows = sel.get("rows")
            if rows is not None:
                try:
                    return tuple(sorted(int(x) for x in list(rows)))
                except (TypeError, ValueError):
                    pass
    return _event_row_indices(event)


def compare_full_paths_from_dataframe_event(
    pivot: pd.DataFrame, event: object, *, widget_key: str
) -> list[str]:
    """**full_path** values for the current **st.dataframe** selection (one class tab)."""
    sel_rows = list(_dataframe_row_indices(widget_key, event))
    return paths_from_stored_row_indices(pivot, tuple(sel_rows))


def paths_from_stored_row_indices(pivot: pd.DataFrame, indices: tuple[int, ...]) -> list[str]:
    """Map row indices to **full_path** (shared helper for dataframe selection)."""
    if pivot.empty or not indices:
        return []
    out: list[str] = []
    for ii in indices:
        if 0 <= ii < len(pivot):
            out.append(str(Path(pivot.iloc[ii]["full_path"])))
    return out


def _summary_tab_container_key(label: str) -> str:
    """ASCII key for ``st.container`` so each Summary tab gets a fresh subtree when switching."""
    safe = "".join(ch if str(ch).isalnum() or ch in "._-" else "_" for ch in str(label))
    return f"edir_sm_{safe}"[:120]


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


def _render_results_and_summary() -> bool | None:
    """**Result by CLASS**, **Summary**, and one inference step per call. Returns **inference_complete** or **None** if setup state is missing."""
    inventory = st.session_state.get(FRAG_INVENTORY_KEY)
    if not isinstance(inventory, pd.DataFrame) or inventory.empty:
        return None

    names_raw = st.session_state.get(FRAG_NAMES_KEY)
    if not isinstance(names_raw, list) or not names_raw:
        return None
    names = [str(x) for x in names_raw]

    mk_slug = str(st.session_state.get(FRAG_MK_KEY, ""))
    files_raw = st.session_state.get(FRAG_FILES_KEY)
    if not isinstance(files_raw, list):
        return None
    files = [Path(str(p)) for p in files_raw]

    weights_raw = st.session_state.get(FRAG_WEIGHTS_KEY)
    if not isinstance(weights_raw, dict):
        return None
    weights_by_name: dict[str, Path] = {str(k): Path(str(v)) for k, v in weights_raw.items()}

    dp = st.session_state.deploy_preds
    p_next = next_file_needing_predictions(files, names, dp)
    p_focus = p_next
    if p_next is not None:
        run_deploy_predictions_for_one_file(
            p_next, names, weights_by_name, st.session_state.deploy_preds
        )
        dp = st.session_state.deploy_preds

    inference_complete = next_file_needing_predictions(files, names, dp) is None

    present_classes = {assigned_label_class(dict(r)) for _, r in inventory.iterrows()}
    class_tab_order = [c for c in CLASS_NAMES if c in present_classes]
    if "unlabeled" in present_classes:
        class_tab_order.append("unlabeled")

    st.subheader("Result by CLASS")
    done_batch, n_batch = inference_files_completed(files, names, dp)
    if n_batch > 0:
        st.progress(
            min(done_batch / n_batch, 1.0),
            text=f"Inference (all models, whole batch): {done_batch} / {n_batch} files",
        )
        if not inference_complete:
            st.caption(
                "**Summary** (metrics, PREVIEW, ALL) appears after every file has been run for every model. "
                "The class tab follows the file being inferred and is **locked** until then."
            )

    if not class_tab_order:
        st.caption("No files to classify.")
        st.session_state.compare_paths = []
    else:
        class_tab_display = [c.upper() for c in class_tab_order]
        display_to_class = dict(zip(class_tab_display, class_tab_order))

        if p_focus is not None:
            seg = _class_segment_display_for_path(p_focus, inventory, class_tab_order)
            if seg is not None:
                st.session_state[CLASS_TAB_PICK_KEY] = seg

        _ensure_segmented_control_default(CLASS_TAB_PICK_KEY, class_tab_display)
        ctab = st.segmented_control(
            " ",
            options=class_tab_display,
            selection_mode="single",
            label_visibility="collapsed",
            key=CLASS_TAB_PICK_KEY,
            width="content",
            disabled=not inference_complete,
        )
        picked_class_d = _summary_segmented_pick(
            ctab,
            st.session_state.get(CLASS_TAB_PICK_KEY),
            class_tab_display,
        )
        picked_cls = display_to_class[picked_class_d]

        prev_cls = st.session_state.get(LAST_RESULT_CLASS_KEY)
        # Stable key per (batch slug, class tab): changing it every inference remounts the grid and can flash blank.
        key_df = f"df_class_{picked_cls}_{mk_slug}"
        if prev_cls is not None and prev_cls != picked_cls:
            st.session_state.compare_paths = []
            st.session_state.pop(COMPARE_PATHS_SIG_KEY, None)
        st.session_state[LAST_RESULT_CLASS_KEY] = picked_cls

        pivot = pivot_for_files(inventory, dp, names, class_filter=picked_cls)
        if pivot.empty:
            st.caption("No files in this class.")
            st.session_state.compare_paths = []
        else:
            _prev_hint = (
                "**Preview** uses only rows selected in **this** class (switching class clears selection)."
                if inference_complete
                else "Class tab **tracks the file being inferred**; you can switch classes after the batch completes."
            )
            st.caption(
                f"**{picked_cls}** — {len(pivot)} file(s). "
                "**CA** cataract · **DR** diabetic_retinopathy · **GL** glaucoma · **NR** normal. "
                + _prev_hint
            )
            styled = style_results_pivot(pivot, names, dp)
            with st.container(border=False):
                ev = st.dataframe(
                    styled,
                    width="stretch",
                    height=DATAFRAME_VIEW_HEIGHT,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key=key_df,
                )
            st.session_state.compare_paths = compare_full_paths_from_dataframe_event(
                pivot, ev, widget_key=key_df
            )

    if inference_complete:
        st.subheader("Summary")
        _sum_fp = _summary_static_fingerprint(mk_slug, files, names, dp)
        if st.session_state.get(SUMMARY_STATIC_CACHE_KEY) != _sum_fp:
            st.session_state[SUMMARY_STATIC_CACHE_KEY] = _sum_fp
            st.session_state[SUMMARY_SUMM_DF_KEY] = model_summary_table(
                names, files, inventory, dp
            )
            st.session_state[SUMMARY_PC_DF_KEY] = per_class_recall_by_model_dataframe(
                names, files, inventory, dp
            )
            st.session_state[SUMMARY_CM_STORE_KEY] = {}
        summ_df = st.session_state[SUMMARY_SUMM_DF_KEY]
        pc_df_cached = st.session_state[SUMMARY_PC_DF_KEY]
        cm_store: dict[str, np.ndarray] = st.session_state[SUMMARY_CM_STORE_KEY]

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

        _ensure_segmented_control_default(SUMMARY_TAB_PICK_KEY, sum_tab_display)
        sel = st.segmented_control(
            " ",
            options=sum_tab_display,
            selection_mode="single",
            label_visibility="collapsed",
            key=SUMMARY_TAB_PICK_KEY,
            width="content",
        )
        picked = _summary_segmented_pick(
            sel,
            st.session_state.get(SUMMARY_TAB_PICK_KEY),
            sum_tab_display,
        )
        label = display_to_label[picked]

        with st.container(key=_summary_tab_container_key(label), border=False):
            if label == "all":
                st.caption(
                    "Cross-model summary table and **per-class recall**; confusion heatmaps are on each model tab."
                )
                st.dataframe(summ_df, width="stretch", hide_index=True)
                st.markdown("##### Per-class accuracy by model (recall, %)")
                st.dataframe(pc_df_cached, width="stretch", hide_index=True)
                st.caption(
                    "Each model column: **%** recall per **actual** (row total in that model’s confusion matrix, "
                    "finished valid predictions). **n_actual** = images with that **actual** in the batch."
                )
            elif label == SUMMARY_TAB_PREVIEW:
                render_compare_blocks(
                    list(st.session_state.get("compare_paths") or []),
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
                    if label not in cm_store:
                        cm_store[label] = confusion_matrix_labeled(
                            files, inventory, dp.get(label, {})
                        )
                    mat = cm_store[label]
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

    return inference_complete


@st.fragment()
def _render_results_fragment() -> None:
    """Results + Summary; chain inference with ``st.rerun`` (fragment scope) from this decorated function only."""
    status = _render_results_and_summary()
    if status is False:
        try:
            st.rerun(scope="fragment")
        except StreamlitAPIException:
            st.rerun()


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

    first_loaded = load_model_for_name(names[0], str(weights_by_name[names[0]]))
    device = device_from_loaded(first_loaded)

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
            placeholder=r"Folder with class subfolders (cataract, …) or one .png/.jpg file",
            help=(
                "**Folder:** must contain subfolders named exactly: cataract, diabetic_retinopathy, glaucoma, normal — "
                "with images inside (any depth under the chosen root). "
                "A plain folder with images but no class-named parents is rejected. "
                "**File:** a single image path is always allowed."
            ),
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
                layout_ok, layout_msg = validate_folder_has_class_layout(folder, files)
                if layout_ok:
                    folder_ok = True
                else:
                    with path_feedback.container():
                        st.warning("Invalid path for batch inference.")
                        st.markdown(layout_msg)
                    folder_ok = False
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
                prepped = eval_transform_pil(image)
                x_preview = EVAL_TRANSFORM(image).unsqueeze(0)
                st.caption(str(one.resolve()))
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Original")
                    st.image(image, width=PREVIEW_IMAGE_WIDTH)
                with c2:
                    st.caption("Preprocessed (224, same resize as inference)")
                    st.image(prepped, width=PREVIEW_IMAGE_WIDTH)
                m0 = load_model_for_name(names[0], str(weights_by_name[names[0]]))
                row = predict_deploy_row(m0, x_preview)
                st.success(f"**{row['predicted_class']}** ({names[0]})")
                for cname in CLASS_NAMES:
                    pv = float(row[f"P({cname})"])
                    st.progress(pv, text=f"{cname}: {pv * 100:.1f}%")

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
        st.session_state.pop(CLASS_TAB_PICK_KEY, None)
        st.session_state.pop(LAST_RESULT_CLASS_KEY, None)
        for _fk in (
            FRAG_INVENTORY_KEY,
            FRAG_NAMES_KEY,
            FRAG_MK_KEY,
            FRAG_FILES_KEY,
            FRAG_WEIGHTS_KEY,
        ):
            st.session_state.pop(_fk, None)
        for _sk in (
            SUMMARY_STATIC_CACHE_KEY,
            SUMMARY_SUMM_DF_KEY,
            SUMMARY_PC_DF_KEY,
            SUMMARY_CM_STORE_KEY,
        ):
            st.session_state.pop(_sk, None)

    inventory = build_file_inventory_dataframe(folder, files)
    if inventory.empty:
        st.info("No rows.")
        return

    st.session_state[FRAG_INVENTORY_KEY] = inventory
    st.session_state[FRAG_NAMES_KEY] = list(names)
    st.session_state[FRAG_MK_KEY] = mk_slug
    st.session_state[FRAG_FILES_KEY] = [str(p.resolve()) for p in files]
    st.session_state[FRAG_WEIGHTS_KEY] = {n: str(weights_by_name[n].resolve()) for n in names}

    _render_results_fragment()


if __name__ == "__main__":
    main()
