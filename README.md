# User Authentication using Biometric Features - Assignment 1

**Course:** Advanced Biometric Systems and Security (AI461), DoAI, SVNIT
**Dataset:** `biomet_data.csv` — 144-dim feature vectors, 100 users × 10 images

---

## 1. Data Format (important — non-obvious)

The raw CSV is **not** samples-as-rows. It is tab-separated with:
- **144 rows** = features
- **1000 columns** = samples (one trailing empty column from a trailing tab, dropped on load)

The 1000 sample columns are grouped **consecutively in blocks of 10 per user**
(columns 0–9 = User 1's 10 images, 10–19 = User 2, etc.). This was verified
empirically: mean Euclidean distance *within* a 10-column block is far lower
than *across* blocks or between random pairs, confirming the block = user
structure.

The script transposes to `(1000 samples, 144 features)` and reshapes to
`(100 users, 10 samples, 144 features)`.

## 2. Pipeline

1. **Split:** per user, first 5 images = enrollment (train), last 5 = test.
2. **Template:** one 144-dim template per user = **mean of the 5 enrollment
   vectors** (standard biometric template generation, matches the pipeline
   given in the assignment's own diagram).
3. **Matching scores**, computed for every test sample against every
   template, using two measures:
   - **Euclidean distance** — lower value = more similar
   - **Cosine similarity** — higher value = more similar
4. **Genuine scores:** each of the 500 test samples (100 users × 5) matched
   against its **own** template → 500 genuine scores per metric.
5. **Impostor scores:** each of the 500 test samples matched against the
   **99 other users'** templates → 49,500 impostor scores per metric.
6. **Performance analysis** (for each metric):
   - Genuine / impostor score histograms (A, B)
   - FAR and FRR vs threshold, swept over the full score range (C, D)
   - ROC curve: TAR (=1−FRR) vs FAR on a log-x axis (E)
   - **EER** — the threshold at which FAR = FRR, found by linear
     interpolation at the sign-change of `FAR − FRR` (F)
   - **Decidability index** `d' = |μ_genuine − μ_impostor| / sqrt((σ²_genuine + σ²_impostor)/2)` (F)

## 3. How to run

```bash
pip install numpy pandas matplotlib
python biometric_authentication.py
```

Data is expected at `data/biomet_data.csv`. Plots land in `plots/`:
- `AB_genuine_impostor_euclidean.png`, `AB_genuine_impostor_cosine.png`
- `CD_FAR_FRR_euclidean.png`, `CD_FAR_FRR_cosine.png`
- `E_ROC_comparison.png`

`results_summary.txt` (all numeric results) is written to the repo root.

## Repo structure

```
.
├── biometric_authentication.py   # main script — run this
├── README.md
├── results_summary.txt           # generated: EER, d', means/stds
├── data/
│   └── biomet_data.csv
└── plots/                        # generated: all PNGs (A,B,C,D,E)
```

## 4. Results

| Metric | Genuine mean ± std | Impostor mean ± std | EER | Decidability (d') |
|---|---|---|---|---|
| Euclidean distance | 291.62 ± 135.46 | 703.82 ± 301.79 | **11.52%** | 1.76 |
| Cosine similarity  | 0.9895 ± 0.0110 | 0.9604 ± 0.0184 | **8.37%**  | 1.93 |

(Full comparison counts: 500 genuine, 49,500 impostor per metric.)

## 5. Which matching criterion performs best?

**Cosine similarity outperforms Euclidean distance** on this dataset:
- Lower EER (8.37% vs 11.52%)
- Higher decidability index (1.93 vs 1.76), i.e. better separation between
  genuine and impostor distributions relative to their spread
- Its ROC curve dominates the Euclidean ROC curve at almost every FAR
  operating point (see `E_ROC_comparison.png`)

**Why:** cosine similarity is scale/magnitude-invariant — it compares the
*direction* of the feature vectors rather than raw magnitude. Since these
144-dim biometric features likely vary in overall magnitude across
acquisitions (e.g. lighting/contrast differences between enrollment and the
"6 months later" test session), Euclidean distance conflates that magnitude
drift with genuine dissimilarity, inflating impostor/genuine overlap.
Cosine similarity is more robust to this kind of magnitude variation, giving
cleaner separation.

## 6. Notes / assumptions for the grader

- Template = mean of 5 enrollment vectors (not a per-sample gallery match);
  this matches the pipeline diagram provided with the assignment.
- Threshold sweep uses 1000 linearly-spaced points spanning the full
  genuine+impostor score range for each metric.
- EER is computed via linear interpolation at the FAR/FRR crossover, not by
  picking the nearest grid point, for a more precise estimate.
- Random seed fixed (42) — results are fully reproducible.
