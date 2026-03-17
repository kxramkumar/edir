# EDIR

**E**ye **D**isease **I**ntelligent **R**ecognition

EDIR is an automated eye disease classification system that analyzes ocular images to support early detection and screening (e.g. cataract, glaucoma, diabetic retinopathy).

## What you need

**uv** — Package and project manager ([docs](https://docs.astral.sh/uv/)). Install directly (see Setup below). No need to install Python first.

*Once uv is installed, the Setup steps will install Python, Jupyter, and all project dependencies for you.*

## Project layout

| Folder | Description |
|--------|-------------|
| **res** | **Resource** root: all input data and run outputs. Shared raw data lives under `res/_common/`; per-run data (preprocess and augment) under `res/<RUN_TAG>/`. See [Data layout (res/)](#data-layout-res) below. |
| **doc** | Project documentation (notes, reports, specs). |
| **nbs** | Jupyter notebooks for exploration, preprocessing, and experiments. |
| **ref** | Reference material (e.g. `brain_tumour.ipynb` as reference for this project). |
| **src** | Reusable Python package and scripts; importable from notebooks and elsewhere after `uv sync`. |

### Data layout (res/)

- **res** = resource root (all data and outputs).
- **prep** = preprocess: cleaned/resized images and step outputs.
- **aug** = augment: copied + augmented images (train up to 2000/class; validate copied only).

Shared raw input is under `res/_common/`. Each run is identified by **RUN_TAG** (e.g. `rgb_v1`); preprocess and augment outputs go under `res/<RUN_TAG>/`.

| Path | Purpose |
|------|---------|
| `res/_common/data/raw/train`, `res/_common/data/raw/validate` | Raw images (one subfolder per class). |
| `res/<RUN_TAG>/data/prep/...` | Preprocess outputs (per-step train/validate). |
| `res/<RUN_TAG>/data/aug/train`, `res/<RUN_TAG>/data/aug/validate` | Augment: train (copied + augmented), validate (copied only). |
| `res/<RUN_TAG>/meta/raw` | Manifest CSV for raw (e.g. `image_manifest.csv`). |
| `res/<RUN_TAG>/meta/prep` | Preprocess manifest. |
| `res/<RUN_TAG>/meta/aug` | Augment manifest and lock (`image_augmented.lock`). |

In the notebook Setup, set `RUN_TAG` (e.g. `"rgb_v1"`) and `RES_ROOT` (e.g. `os.path.join("..", "res")`). The notebook checks that `res/` and the raw folders exist and are non-empty before running.

## Setup


### 1. Install uv (direct install)

Install uv using the official installer. No base Python required.

**Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal (or add uv to your PATH) and run `uv --version` to confirm.

### 2. Clone the repo and install dependencies

From the project root:

```bash
cd edir
uv sync
```

This creates a virtual environment, installs Python 3.11 if needed, and installs all dependencies (including Jupyter).

**Set up dependencies for Jupyter:** From the project root, run:

```bash
uv sync
```

This installs everything in `pyproject.toml` (Jupyter, numpy, pandas, matplotlib, opencv, seaborn, etc.). The Jupyter kernel will then have access to all of them when you run `uv run jupyter lab`.

**Adding a new dependency:** Jupyter only sees packages that are in the project. To use a new library in your notebooks, add it with `uv add <package-name>` (this updates `pyproject.toml`), then run `uv sync`. Restart the Jupyter kernel or restart Jupyter Lab so it picks up the new package.

### 3. Set up the resource directory and data

The project expects a **resource** root **res/** at the project root (or the path you set as `RES_ROOT` in the notebook). Shared raw data lives under `res/_common/data/raw/`; preprocess (**prep**) and augment (**aug**) outputs go under `res/<RUN_TAG>/`.

**Understanding: paths and run tag**

In the notebook Setup you set:

- **RES_ROOT** — Resource root (e.g. `os.path.join("..", "res")`). Must exist.
- **RUN_TAG** — Run identifier (e.g. `"rgb_v1"`). Prep and aug outputs go under `res/<RUN_TAG>/data/prep`, `res/<RUN_TAG>/data/aug`, and manifests under `res/<RUN_TAG>/meta/`.

The notebook resolves paths like this:

- Common raw: `res/_common/data/raw/train`, `res/_common/data/raw/validate` (one subfolder per class).
- Preprocess: `res/<RUN_TAG>/data/prep/...`, manifest at `res/<RUN_TAG>/meta/prep/image_manifest.csv`.
- Augment: `res/<RUN_TAG>/data/aug/train`, `res/<RUN_TAG>/data/aug/validate`, manifest at `res/<RUN_TAG>/meta/aug/image_manifest.csv`, lock at `res/<RUN_TAG>/meta/aug/image_augmented.lock`.

It checks that `res/` exists and that the raw train/validate folders exist and contain at least one class subfolder before running.

**Directory layout:**

| Folder | Purpose |
|--------|---------|
| `res/_common/data/raw/train` | Training images (one subfolder per class). |
| `res/_common/data/raw/validate` | Validation images (one subfolder per class). |
| `res/<RUN_TAG>/data/prep/...` | Preprocess outputs (populated by the notebook). |
| `res/<RUN_TAG>/data/aug/...` | Augment outputs (populated by the notebook). |
| `res/<RUN_TAG>/meta/raw`, `meta/prep`, `meta/aug` | Manifest CSVs and aug lock. |

**Get the data (links shared by mentor):**

1. **Train data** — [Download](https://drive.google.com/drive/folders/16uCYQS4-AiZrz4Jp6dXt9meM2QHh4JRv) and put the contents into `res/_common/data/raw/train` (class folders e.g. `cataract`, `glaucoma`, `diabetic_retinopathy`, `normal` directly inside).
2. **Validate data** — [Download](https://drive.google.com/drive/folders/1FwfXqTUIVcTZtf3bxIbDF5L0B2JeT5mv) and put the contents into `res/_common/data/raw/validate`.

**Create the folders if they don’t exist:**

```bash
# Linux / macOS / Git Bash
mkdir -p res/_common/data/raw/train res/_common/data/raw/validate
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path res/_common/data/raw/train, res/_common/data/raw/validate
```

Then copy or extract the downloaded data into the correct folders. The notebook creates `res/<RUN_TAG>/data/prep`, `res/<RUN_TAG>/data/aug`, and `res/<RUN_TAG>/meta/...` when you run it.

## Run Jupyter Lab

From the project root:

```bash
uv run jupyter lab
```

This runs Jupyter Lab using the project’s virtual environment and dependencies.
