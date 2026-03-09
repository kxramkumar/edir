# EDIR

**E**ye **D**isease **I**ntelligent **R**ecognition

EDIR is an automated eye disease classification system that analyzes ocular images to support early detection and screening (e.g. cataract, glaucoma, diabetic retinopathy).

## What you need

**uv** — Package and project manager ([docs](https://docs.astral.sh/uv/)). Install directly (see Setup below). No need to install Python first.

*Once uv is installed, the Setup steps will install Python, Jupyter, and all project dependencies for you.*

## Project layout

| Folder | Description |
|--------|-------------|
| **art** | Artifacts: input and output. Datasets (`art/data/raw`, `art/data/clean` with `train`/`validate`), reports (`art/report/profiling`), and later figures, models, logs. |
| **doc** | Project documentation (notes, reports, specs). |
| **nbs** | Jupyter notebooks for exploration, preprocessing, and experiments. |
| **ref** | Reference material (e.g. `brain_tumour.ipynb` as reference for this project). |
| **src** | Reusable Python package and scripts; importable from notebooks and elsewhere after `uv sync`. |

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

### 3. Set up the artifact directory and data

The project expects input data under `art/data/raw/`. You must create the folders and add the dataset.

**Directory layout:**

| Folder | Purpose |
|--------|---------|
| `art/data/raw/train` | Training images (one subfolder per class). |
| `art/data/raw/validate` | Validation images (one subfolder per class). |
| `art/data/clean` | Cleaned/processed images (populated later). |
| `art/report/profiling` | ydata-profiling HTML reports. |

**Get the data (links shared by mentor):**

1. **Train data** — [Download](https://drive.google.com/drive/folders/16uCYQS4-AiZrz4Jp6dXt9meM2QHh4JRv) and put the contents into `art/data/raw/train` (so that class folders e.g. `cataract`, `glaucoma`, `diabetic_retinopathy`, `normal` are directly inside `art/data/raw/train`).
2. **Validate data** — [Download](https://drive.google.com/drive/folders/1FwfXqTUIVcTZtf3bxIbDF5L0B2JeT5mv) and put the contents into `art/data/raw/validate`.

**Create the folders if they don’t exist:**

```bash
# Linux / macOS / Git Bash
mkdir -p art/data/raw/train art/data/raw/validate
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force -Path art/data/raw/train, art/data/raw/validate
```

Then copy or extract the downloaded data into the correct folders.

## Run Jupyter Lab

From the project root:

```bash
uv run jupyter lab
```

This runs Jupyter Lab using the project’s virtual environment and dependencies.
