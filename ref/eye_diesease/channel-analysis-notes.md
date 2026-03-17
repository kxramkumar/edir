# Retinal Channel Analysis — Visual Observations

---

## Image 1: Class-wise RGB Histogram

### What the plot shows
Four subplots (one per class) — each overlaying R, G, B channel proportion distributions across all train images, bins 10–255.

### Observations per channel

**Blue channel (purple bars)**
- Peaks sharply at low values (~30–80) across all four classes
- Drops off quickly — very little blue signal above 150
- Reason: fundus cameras illuminate the retina with focused white/green light; the blue wavelength scatters heavily in the lens and vitreous, so most blue energy is absorbed before reaching the sensor cleanly → image is blue-dark overall

**Green channel (teal/green bars)**
- Sits in the mid-range (~50–150), smooth bell-shaped spread
- Consistent shape across all classes — this is the "stable" channel
- Reason: green (~550 nm) is where the retina reflects cleanly; neither too absorbed nor too saturated; this is where tissue texture, vessel branching, and optic disc structure live

**Red channel (red/pink bars)**
- Spreads far rightward (~100–250), wide and flat
- Cataract pushes red even further right — lens scatters and warms the image
- Diabetic retinopathy shows a longer red tail compared to normal — likely microaneurysm and haemorrhage reflectance
- Reason: the choroid (deep retinal layer) is highly vascularised and reflects red; fundus images always skew red

### Cross-class signal
- All four classes show the same channel ordering (B peak low → G mid → R high) but differ in shape and spread
- This means the three channels together carry class-discriminative information — a model using all three sees more than any single channel alone

---

## Image 2: Retina Preprocessing — Very Blurry Bin (0–50 Laplacian)

### Four classes shown: cataract, diabetic_retinopathy, glaucoma, normal

### Panel-by-panel read

**Original RGB**
- All four images have the characteristic circular retinal fundus crop
- Even in the very blurry bin (Laplacian variance 25–40), the disc and vessel structure is still faintly visible
- Cataract original is noticeably more orange/warm — consistent with the rightward red histogram shift

**Green channel (CLAHE + ROI crop)**
- Clearly the strongest channel: vessel branching, optic disc boundary, and lesion spots are all visible
- Cataract: white/bright dots (lens opacities reflecting back) show up distinctly against the dark background
- Diabetic retinopathy: vessel network is denser and brighter than normal — microvasculature pattern visible
- Glaucoma: disc appears slightly enlarged/brighter — cupping artefact on the bright spot
- Normal: clean, dark background with clearly defined disc and minimal vessel density

**Blue channel (CLAHE + ROI crop)**
- Very noisy, texture-heavy
- Glaucoma blue panel shows a "grainy" texture pattern — possibly corneal/lens scatter
- Diabetic retinopathy blue shows some vessel outlines but buried in noise
- Low signal-to-noise ratio — not useful as a primary feature channel

**Red channel (CLAHE + ROI crop)**
- Bright and uniform — most of the image is high-intensity red
- Very low contrast between vessels and background
- Cataract red is almost entirely saturated (all bright) — lens scatter floods the channel
- Red is useful for detecting bright lesions (exudates) but poor for vessel structure

**Grayscale (CLAHE + ROI crop)**
- A weighted average of all three channels — sits between green quality and red saturation
- Retains disc visibility and some vessel structure
- Less contrast than green alone — the red channel's brightness dilutes the vessel-background separation

---

## Summary: Why Green is the Best Single Channel

| Channel | Contrast | Noise | Vessel visibility | Lesion visibility |
|---|---|---|---|---|
| Green | High | Low | Best | Good |
| Grayscale | Medium | Medium | Good | Moderate |
| Red | Low | Low | Poor | Good (exudates) |
| Blue | Low | High | Poor | Poor |

- Green maximises the signal from haemoglobin absorption (~550 nm) — blood vessels absorb green light → appear dark → strong contrast against the brighter retinal background
- CLAHE on green further stretches this contrast without blowing out highlights (unlike red which saturates)
- For a CNN: green alone is the strongest single-channel input; all three channels together give complementary discriminative signal that no single channel provides alone
