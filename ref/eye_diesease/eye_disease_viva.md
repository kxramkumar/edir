# Eye disease pipeline — viva notes (technical / libraries only)

Companion to `eye_disease.py`. This document lists **what each section computes**, **which algorithm or rule** is used, and **how the main libraries define or implement** the key operations. It intentionally **does not** reproduce charts, graphs, image grids, or tabular outputs—only concepts and APIs you can explain orally.

---

## 1. Introduction — problem and pipeline

- **Purpose (conceptual):** Build a reproducible path from raw class-folder images → quality audit (EDA) → targeted preprocessing → augmentation for classifier training.
- **No single library “algorithm” here:** This section frames the *workflow* (collect manifest, explore, preprocess, augment). The implementation is in later sections.

---

## 2. Setup — imports and constants

| Area | Library / module | Role in this project |
|------|-------------------|----------------------|
| Arrays & numerics | **NumPy** (`numpy`) | Histograms, masks, blending, percentiles, array typing for images. |
| Images (I/O & ops) | **OpenCV** (`cv2`) | `imread`, color conversion, Laplacian, CLAHE, Gaussian blur, Canny, morphology, contours, resize, threshold, `addWeighted`. |
| Tables | **pandas** (`pd`) | Manifest as `DataFrame`; filtering, merges, groupby, categorical bins. |
| Optional audit | **CleanVision** (`cleanvision.Imagelab`) | Automated issue detection; merged into manifest when installed. |
| Hashing (if used elsewhere) | **imagehash** | Perceptual hashing utilities (imported in setup). |
| Deep learning I/O | **TensorFlow Keras** | `ImageDataGenerator`, `load_img`, `img_to_array` for augmentation. |
| PIL | **Pillow** (`PIL.Image`) | Saving augmented arrays as PNG; bridge to CleanVision blurriness property when used. |

**Blur bins (`get_blur_bin`, `BLUR_BINS`):** Custom **threshold bins** on **Laplacian variance** (see §5 / §6.11)—not a library-defined enum; the code maps a scalar variance to human-readable labels.

---

## 3. Paths — resource root and run tag

- **Technique:** Central **RES_ROOT** and **RUN_TAG** so all artifacts are under `res/<RUN_TAG>/meta/...` and `res/<RUN_TAG>/data/...`.
- **Libraries:**
  - **`os.path.join`**, **`os.makedirs(..., exist_ok=True)`**, **`os.path.isdir`**, **`os.listdir`** — Python **standard library** path and filesystem checks.
  - **`pathlib.Path`** is imported but the script mostly uses `os.path` for compatibility.
- **Validation logic:** Fails fast if expected raw train/validate roots are missing or empty (data integrity before any processing).

---

## 4. Common methods

### 4.1 `list_image_files(root_dir)`

- **Algorithm:** Depth-1 walk: for each **immediate subdirectory** (treated as **class name**), collect files whose extension is in `IMAGE_EXTENSIONS`; skip `SKIP_NAMES`.
- **Libraries:** `os.listdir`, `os.path.isdir`, `os.path.join`, string `endswith` on filename.

### 4.2 `path_from_root(full_path)`

- **Algorithm:** Normalize absolute paths and strip the prefix of **RES_ROOT** to store **relative paths** in CSVs (portable manifests).
- **Libraries:** `os.path.abspath`, `os.path.normpath`.

### 4.3 Display helpers (`display_wide`, `show_attr_charts`, `show_class_distribution`, `show_attr_samples`)

- **Viva angle:** These are **presentation layers** (IPython `display(HTML)`, **Plotly**, **Matplotlib**, optional **SciPy** `gaussian_kde` for smooth density). For oral exam, state that the **underlying metrics** come from the manifest columns computed in §5–6; the helpers only visualize them. **No chart details** in this document per your brief.

---

## 5. Collection — raw manifest

### 5.1 Per-image row (core metrics)

- **`os.path.splitext` / manual mapping:** File **format** label (e.g. `jpeg` normalized from `jpg`).
- **`os.path.getsize(path)`** (Python docs): Returns the **size of the file in bytes**. Used as **`file_size`**.
  - **Empty file (for EDA §6.3):** Operationally defined as **`file_size == 0`** (zero-byte file). The OS may still report a path that exists but has no payload.

### 5.2 Corrupt / unreadable flag

- **`cv2.imread(path)`** (OpenCV): Reads an image from a file. If the image **cannot be read** (missing file, unsupported format, corrupted data), the function returns **`None`**.
- **Project rule:** **`is_corrupt = True`** when `img is None`; otherwise **`is_corrupt = False`** and geometric/statistical fields are filled.

### 5.3 Geometry and simple global stats (readable images only)

- **`img.shape`:** Height, width, channel count.
- **`aspect_ratio`:** \(h/w\); **`resolution`:** \(h \times w\).
- **`brightness`:** Mean of all pixel values in the BGR array (`img.mean()`), a coarse global luminance proxy.

### 5.4 Blur metric — Laplacian variance

- **`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`:** BGR → single-channel gray.
- **`cv2.Laplacian(gray, cv2.CV_64F)`** (OpenCV): Computes the **Laplacian** of the image (2nd derivative operator, edge-sensitive). Sharp images produce stronger responses.
- **`.var()`** on the Laplacian output: **Variance** of the Laplacian response—widely used as a **no-reference sharpness** score; **higher variance → typically sharper**, **lower → blurrier**.

### 5.5 CleanVision merge (optional)

- **`Imagelab(data_path=class_path)`** then **`find_issues()`** (CleanVision): Scans images under that folder and populates an **issues** table (boolean flags and scores per image type of issue).
- **`pandas.merge`:** Left-joins CleanVision columns onto the manifest on **`path`** so each row carries both OpenCV metrics and CleanVision flags.

---

## 6. EDA — exploratory data analysis (raw manifest)

**General rule:** Most analyses **exclude corrupt rows** (`is_corrupt == False`) when computing image-quality views. **Empty** and **corrupt** checks use **`file_size`** and **`is_corrupt`** from §5.

### 6.1 Class labels and counts

- **Technique:** **`value_counts()`** on manifest `class` column per split—pure **pandas** aggregation on already-collected rows.

### 6.2 File format and counts

- **Technique:** Groupby **`format` × `class`**; same as counting rows in the manifest (format was set at collection time).

### 6.3 Empty files and counts

- **Definition in this pipeline:** **`file_size == 0`** (see **`os.path.getsize`** in §5.1).
- **Meaning:** The file exists on disk but contains **no bytes**; it is not a usable image for training.

### 6.4 Corrupt / unreadable files and counts

- **Definition in this pipeline:** **`is_corrupt == True`**, i.e. **`cv2.imread` returned `None`** (§5.2).
- **Library reference:** OpenCV **`imread`** failure modes include corrupt encoding, wrong format, or read errors.

### 6.5 Image size (odd size)

- **Source:** CleanVision columns **`is_odd_size_issue`**, **`odd_size_score`** (when present).
- **Technique:** Map boolean flag to bins **`odd_size` / `normal`** for counting; optional **`describe()`** on scores (**pandas** summary statistics: count, mean, std, min, quartiles, max).

### 6.6 Aspect ratio (odd aspect ratio)

- **Source:** **`is_odd_aspect_ratio_issue`**, **`odd_aspect_ratio_score`** (CleanVision).
- **Same pattern as §6.5:** flag → categorical bin; distributional stats via pandas.

### 6.7 Dark (underexposure)

- **Source:** **`is_dark_issue`**, **`dark_score`** (CleanVision). Comment in code: score axis described as **0 = darkest, 1 = brightest** for interpretation in plots (oral: “library-provided relative darkness score”).

### 6.8 Light (overexposure)

- **Source:** **`is_light_issue`**, **`light_score`** (CleanVision). Comment: **0 = brightest, 1 = darkest** for that score in this notebook’s captioning.

### 6.9 Near duplicates

- **Source:** **`is_near_duplicates_issue`**, **`near_duplicates_score`** (CleanVision). Near-duplicate detection is implemented **inside CleanVision** (perceptual similarity / hashing-style logic—cite CleanVision docs if examiner asks for internals).

### 6.10 Exact duplicates

- **Source:** **`is_exact_duplicates_issue`**, **`exact_duplicates_score`** (CleanVision). Flags images that are **identical** (or treated as such by the library’s exact-dup detector).

### 6.11 Blur (manifest column `blur`)

- **Source:** **Laplacian variance** computed at collection (§5.4), **not** the CleanVision blurry flag in this subsection.
- **Technique:** **`get_blur_bin`** maps continuous variance into ordered bins (`BLUR_BINS`).

### 6.12 Low information

- **Source:** **`is_low_information_issue`**, **`low_information_score`** (CleanVision). **Low-information images** are those the library flags as carrying **little usable visual signal** (e.g. nearly uniform, extremely dark/light, or otherwise uninformative)—exact criterion is defined by CleanVision’s implementation.

### 6.13 CleanVision issues summary

- **Technique:** For each issue column in **`CV_ISSUE_COLS`**, count rows where the **`is_*_issue`** boolean is **True**, per class. **`any_issue`:** row-wise **logical OR** across available issue columns (**pandas** boolean reduction).

### 6.14 Class-wise RGB histogram

- **Technique:** For each class, accumulate **per-channel 256-bin histograms** over train images.
- **`cv2.cvtColor(..., COLOR_BGR2RGB)`** then **`cv2.split`:** Separate R, G, B planes.
- **`numpy.histogram`** (NumPy docs): Computes the **histogram** of data over given **bins**; returns counts per bin. Here bins are edges `0…256` → **256 bins** for pixel values 0–255.
- **Persistence:** **`pickle`** (`pickle.dump` / `load`) — Python object serialization to **`rgb_hist_cache.pkl`** to avoid recomputation.
- **Downstream normalization:** Counts summed to proportions per channel (divide by sum); low bins optionally excluded to de-emphasize black borders (**project choice**, not a library requirement).

### 6.15 Class-wise greyscale histogram

- **`cv2.cvtColor(..., COLOR_BGR2GRAY)`:** Standard BGR→gray conversion.
- **`numpy.histogram`** on gray pixels; same caching idea with **`gray_hist_cache.pkl`**.

### 6.16 Edge detection

- **`cv2.createCLAHE(clipLimit, tileGridSize)`** (OpenCV): **CLAHE** = **Contrast Limited Adaptive Histogram Equalization**—adaptive local contrast boost with a **clip limit** to limit noise amplification.
- **`cv2.GaussianBlur(ksize, sigma)`:** **Gaussian smoothing**—reduces noise before edge detection.
- **`cv2.Canny(threshold1, threshold2)`** (OpenCV): **Canny edge detector**—multi-stage algorithm (gradient magnitude + hysteresis thresholding) producing a binary edge map. Parameters **30, 100** are **low/high thresholds** in the Canny procedure.
- **Per-channel application:** Same pipeline on B, G, R and on gray; fundus work often highlights **green channel** vessels (domain note for viva).

### 6.17 Retina preprocessing (demo function)

- **`preprocess_retina_image`:** Per channel: CLAHE → Gaussian blur → **`cv2.threshold(..., THRESH_BINARY)`** → **`cv2.findContours`** → **`cv2.boundingRect`** on **largest contour** → **`cv2.resize`** to target (e.g. 224×224).
- **Contour / ROI idea:** Largest bright region after thresholding approximates the **field / disc region**; bounding box crops content before resize.

---

## 7. Preprocess (prep)

### 7.1 Preprocess manifest

- **Technique:** Copy raw manifest to **`df_prep`**; derive boolean flags **`has_odd_size`**, **`has_dark`**, **`has_exact_dup`** from CleanVision columns when present; **`has_blur`** from **Laplacian bins** (`BLUR_CLEAN_BINS`: “very blurry” and “blurry” ranges only).
- **Constraint:** Only **train** and **non-corrupt** rows may carry these issues; validate/corrupt forced **False**.
- **`issues`:** Comma-separated string built from flags (**application logic**).
- **`prep_path` / `prep_status`:** Bookkeeping columns for pipeline state (`""`, later paths; **`removed`** after duplicate step).

### 7.2 Step 1 — Exact duplicate removal

- **Technique:** Among rows with **`has_exact_dup`**, **group by** `(split, class, blur)`; **keep one row per group** (first after sort), mark others **`prep_status = "removed"`**.
- **Library:** **pandas** `groupby`, `head(1)`, index differencing—**deterministic thinning** of duplicates flagged by CleanVision; **`blur`** used as part of the grouping key in this script.

### 7.3 Step 2 — Odd size correction

- **Target size:** **Mode** of `"width x height"` among normal train images from raw manifest (fallback **512×512**).
- **`_crop_black_then_resize`:**
  - **`cv2.cvtColor` → gray** (or use gray if already 1-channel).
  - **`cv2.threshold(gray, black_thresh, 255, THRESH_BINARY)`:** Pixels above threshold treated as **content**; below as **background/border** (fundus black surround).
  - **`cv2.findNonZero(mask)`** + **`cv2.boundingRect`:** Tight axis-aligned crop around content.
  - **`cv2.resize(..., interpolation=cv2.INTER_LANCZOS4)`:** **Lanczos** high-quality resampling to target width/height.

### 7.4 Step 3 — Blur enhancement

- **Unsharp mask (`_unsharp_mask`):** **`GaussianBlur`** to get a smoothed copy; **`cv2.addWeighted`** combines original and blurred: classic **unsharp masking** (sharpening) in OpenCV.
- **Masked application (`_unsharp_masked`):** **`_eyeball_mask`:** threshold gray → **morphological erode** (`cv2.erode` with elliptical kernel) to shrink mask inward → **Gaussian blur on mask** for **feathering**; blend enhanced and original with mask as alpha (**NumPy** float blend).
- **Optional score (`_blurry_score_for_image`):** Uses CleanVision **`BlurrinessProperty.calculate`** when available; else maps **Laplacian variance** through **`1 - exp(-lap_var/scale)`** clipped to **[0,1]** as a **monotonic sharpness proxy**.

### 7.5 Step 4 — Dark enhancement

- **`_enhance_dark`:** For each BGR channel: **`clahe.apply(channel)`** then **`GaussianBlur(3,3)`** — local contrast lift followed by **noise/grain smoothing** (project comment: align with earlier “v3” style cleaning).
- **`_dark_score_for_image`:** **5th percentile** of gray intensities / 255 (**`numpy.percentile`**) as a **robust brightness** statistic (insensitive to specular highlights), analogous spirit to dark-score interpretation.

---

## 8. Augmentation

### 8.1 Copy process files to aug tree

- **Technique:** **`shutil.copy2`** (Python): Copies file **data and metadata** (e.g. timestamps on supported platforms). Destination: `res/<RUN_TAG>/data/aug/train|validate/<class>/...`.
- **Source selection:** Prefer **`prep_path`** if set, else original **`path`**.
- **Lock / manifest:** **`aug_manifest_path`** tracks **`aug_path`**; **lock file** prevents accidental re-runs.

### 8.2 Augment images (train only, target count per class)

- **Target:** Fill each train class folder up to **`TARGET_COUNT`** (e.g. 2000) with **synthetic** images.
- **`tensorflow.keras.preprocessing.image.ImageDataGenerator`** (Keras / TensorFlow): Declarative **affine / photometric** random transforms. This script uses **three** generators:
  - **Rotation:** `rotation_range=15`, `fill_mode="constant"`, `cval=0` (black padding where needed).
  - **Zoom:** `zoom_range=0.1`.
  - **Flip:** `horizontal_flip=True`.
- **`load_img(..., color_mode="rgb", target_size=IMG_SIZE)`** + **`img_to_array`:** Loads image as RGB array suitable for **`random_transform`**.
- **`Image.fromarray(x).save(..., format="PNG")` (Pillow):** Writes augmented tensors to disk.

---

## Quick viva phrases (empty vs corrupt)

| Term in this project | How it is decided | Library / API |
|---------------------|-------------------|---------------|
| **Empty file** | `file_size == 0` | **`os.path.getsize`** returns **0** bytes. |
| **Corrupt / unreadable** | `cv2.imread` fails | OpenCV **`imread`** returns **`None`** → cannot decode into an image matrix. |

---

## Document metadata

- **Source script:** `nbs/eye_disease.py`
- **Scope:** Algorithms, definitions, and library APIs only—no figure or table reproduction.
