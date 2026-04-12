# Eye disease notebook — Sections 6.7–6.17  
**Reference for print (library, parameters, plain-language summary)**

*Source: `nbs/eye_disease.py` / notebook EDA pipeline.*

---

## How to use this document

- **Libraries** = packages and the main API you call.  
- **Parameters** = numbers or choices in *your* code, or defaults *inside* a library. These are **not** neural-network training hyperparameters unless you later tune them for a model; they are **measurement / visualization / detector settings**.  
- **Layman** = what it does in simple terms (viva-style).

---

## 6.7 Dark (underexposure)

| | |
|:---|:---|
| **Library / functions** | **CleanVision**: `Imagelab`, `find_issues()` → columns `is_dark_issue`, `dark_score` (merged into manifest; analyzed with **pandas**). |
| **Parameters** | **Your code:** none passed to `find_issues()` — **CleanVision defaults** control thresholds and scoring. |
| **Layman** | Automatically marks photos that look **too dark / underexposed** and gives a **relative darkness score** so you can see counts per class. (Plot captions: interpret score axis as documented in the notebook — e.g. 0 = darkest, 1 = brightest for dark_score.) |

---

## 6.8 Light (overexposure)

| | |
|:---|:---|
| **Library / functions** | **CleanVision**: `Imagelab`, `find_issues()` → `is_light_issue`, `light_score`. |
| **Parameters** | **CleanVision defaults** (no extra arguments in your pipeline). |
| **Layman** | Flags images that look **too bright / washed out** and scores them for exploration by class. (Caption axis for light_score as in your notebook.) |

---

## 6.9 Near duplicates

| | |
|:---|:---|
| **Library / functions** | **CleanVision**: `Imagelab`, `find_issues()` → `is_near_duplicates_issue`, `near_duplicates_score`; grouping info from `lab.info["near_duplicates"]["sets"]`. |
| **Parameters** | **Internal to CleanVision** (perceptual similarity / hashing-style logic — cite their docs if asked for internals). |
| **Layman** | Finds images that are **almost the same** (not necessarily byte-identical), so you see **redundant** training samples. |

---

## 6.10 Exact duplicates

| | |
|:---|:---|
| **Library / functions** | **CleanVision**: `find_issues()` → `is_exact_duplicates_issue`, `exact_duplicates_score`. |
| **Parameters** | **CleanVision defaults** for exact-duplicate detection. |
| **Layman** | Marks images treated as **identical** by the library’s exact-duplicate detector. |

---

## 6.11 Blur (manifest column — Laplacian variance)

| | |
|:---|:---|
| **Library / functions** | **OpenCV**: `cv2.imread`, `cv2.cvtColor` (BGR→GRAY), `cv2.Laplacian`, `.var()`; your helpers **`get_blur_bin`**, **`BLUR_BINS`**, **`BLUR_ORDER`**. |
| **Parameters** | **Laplacian:** `cv2.CV_64F`; default **3×3** Laplacian kernel (OpenCV). **Bin edges (variance):** `[0, 50)`, `[50, 150)`, `[150, 300)`, `[300, 500)`, `[500, ∞)` — labels e.g. very blurry → very sharp. **Your chosen EDA thresholds**, not ML hyperparameters. |
| **Layman** | One **sharpness number** per image (higher variance → sharper); then sorted into **blur buckets** for tables, plots, and sampling. *This subsection uses manifest blur, not the CleanVision “blurry” flag.* |

---

## 6.12 Low information

| | |
|:---|:---|
| **Library / functions** | **CleanVision**: `find_issues()` → `is_low_information_issue`, `low_information_score`. |
| **Parameters** | **CleanVision defaults** for what counts as low information. |
| **Layman** | Flags images with **little usable visual signal** (criterion defined by CleanVision’s implementation). |

---

## 6.13 CleanVision issues summary

| | |
|:---|:---|
| **Library / functions** | **pandas**: filtering on boolean columns, counting, row-wise **`.any(axis=1)`**; your constant **`CV_ISSUE_COLS`** (pairs of label + `is_*_issue` column names). |
| **Parameters** | **Which issues are included:** fixed list (e.g. dark, light, blurry, low_information, odd_aspect_ratio, odd_size, near_duplicates, exact_duplicates). **No numeric tunables** — this is **reporting configuration**. |
| **Layman** | **Per issue type:** how many images are flagged, **per class** and total. **`any_issue`:** image has **at least one** of those issues. |

---

## 6.14 Class-wise RGB histogram

| | |
|:---|:---|
| **Library / functions** | **OpenCV**: `cv2.imread`, `cv2.cvtColor` (BGR→RGB), `cv2.split`. **NumPy**: `np.histogram`, `np.arange`. **pickle**: save/load `rgb_hist_cache.pkl`. **Plotly**: bar charts (in notebook). |
| **Parameters** | **Bins:** `np.arange(257)` → **256 bins** for pixel values 0–255. **Display:** `DARK_THRESH = 10` → exclude bins **0–9** in the plot (de-emphasize black borders). **Normalization:** counts ÷ sum → **proportions** per channel. |
| **Layman** | Builds a **color brightness fingerprint** per class: how much of each red/green/blue level appears across all train images; cached so you don’t recompute every run. |

---

## 6.15 Class-wise greyscale histogram

| | |
|:---|:---|
| **Library / functions** | **OpenCV**: `cv2.cvtColor` (BGR→GRAY). **NumPy**: `np.histogram`. **pickle**: `gray_hist_cache.pkl`. Plotting as in 6.14. |
| **Parameters** | **256 bins** (`np.arange(257)`). **`DARK_THRESH = 10`** for plot exclusion. **Proportion** normalization. |
| **Layman** | Same idea as RGB but **single channel** — overall **brightness distribution** per class. |

---

## 6.16 Edge detection (Canny visualization)

| | |
|:---|:---|
| **Library / functions** | **OpenCV**: `cv2.createCLAHE`, `.apply`, `cv2.GaussianBlur`, `cv2.Canny`, `cv2.cvtColor`. **Matplotlib**: `imshow`, subplots. |
| **Parameters** | **CLAHE:** `clipLimit=2.0`, `tileGridSize=(8, 8)`. **Gaussian blur:** kernel `(3, 3)`, `sigma=0` (OpenCV derives σ). **Canny:** `threshold1=30`, `threshold2=100`. **Hand-picked** for visualization. |
| **Layman** | **Boosts local contrast**, **smooths noise**, then finds **edges** (outlines). Run on **B, G, R and gray** to compare which channel shows structure (e.g. vessels in fundus images). |

---

## 6.17 Retina preprocessing demo (`preprocess_retina_image`)

| | |
|:---|:---|
| **Library / functions** | **OpenCV**: `cv2.imread`, `cv2.createCLAHE`, `cv2.GaussianBlur`, `cv2.threshold`, `cv2.findContours`, `cv2.boundingRect`, `cv2.resize`, `cv2.cvtColor`. |
| **Parameters** | **CLAHE:** `clipLimit=2.0`, `tileGridSize=(8, 8)`. **Gaussian blur:** `(3, 3)`, `sigma=0`. **Threshold:** `thresh=10`, `maxval=255`, `cv2.THRESH_BINARY`. **Contours:** `cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`. **Resize:** default `target_size=(224, 224)`. |
| **Layman** | Per channel: **enhance contrast** → **blur** → **threshold** to separate bright regions → **largest blob** → **crop its box** → **resize** to fixed size. **Demo** idea: approximate **field / disc region** before standardizing size. |

---

## Viva one-liner: “hyperparameter” vs “setting”

- **Neural-net hyperparameters** = learning rate, batch size, epochs, model width — tuned for training.  
- **Here:** CleanVision uses **fixed internal rules** (unless you change their API). OpenCV/NumPy values are **engineering choices** for **measurement and plots**; name them if asked, but they are **not** the same as training hyperparameters unless you explicitly optimize them for a classifier later.

---

*End of document.*
