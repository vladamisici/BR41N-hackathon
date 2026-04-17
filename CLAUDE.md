# BR41N.IO Spring School 2026 — Stroke Rehab MI Classification

## What This Is

Solo entry for the BR41N.IO BCI Data Analysis hackathon (April 25–26, 2026).
Track: **Stroke Rehab Data Analysis** — classify left vs. right hand motor imagery
from a chronic stroke patient's 16-channel EEG. Beat CSP+LDA baseline. Win $1000.

## Dataset Format (g.tec recoveriX .mat files)

- **3 patients**, each with pre-intervention and post-intervention sessions
- Fields: `y` (n_samples × 16) EEG in µV, `trig` (n_samples × 1) event markers, `fs` = 500 Hz
- Load with `scipy.io.loadmat(path, squeeze_me=True)`
- If loadmat fails, try h5py (MATLAB v7.3+)
- Transpose `y` to (16, n_samples) for MNE
- Scale µV → V: multiply by 1e-6

### 16-Channel Montage (10/20)

```
FC5, FC1, FCz, FC2, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP1, CP2, CP6
```

All sensorimotor. Reference: right earlobe. Ground: FPz.

### Hemisphere Indices

- LEFT:    [0,1,5,6,7,12,13]   → FC5, FC1, C5, C3, C1, CP5, CP1
- RIGHT:   [3,4,9,10,11,14,15] → FC2, FC6, C2, C4, C6, CP2, CP6
- MIDLINE: [2,8]               → FCz, Cz

### Trial Paradigm

- T=0s beep → T=2s cue (left/right) → T=3.5s feedback → T=8s relax
- 240 trials per session (3 runs × 80), balanced left/right
- Epoch window: **0.5–4.5s post-cue** (captures MI, avoids evoked response)

## Strategy (Priority Order)

1. **Riemannian geometry classifiers FIRST** — TS+LR, MDM, FgMDM, TS+SVM (no GPU, robust to noise, works with 160 trials)
1. **ACM (Augmented Covariance Method)** — Takens delay embedding → covariance → tangent space → SVM
1. **Deep learning ensemble** — EEGNet + ShallowConvNet via braindecode, with data augmentation
1. **Soft voting ensemble** — 0.4×Riemannian + 0.3×EEGNet + 0.3×ShallowConvNet
1. **Transfer learning** — Riemannian Procrustes Analysis if time allows

## Technical Rules

- **Covariance estimation**: Always use `'oas'` or `'lwf'`, NEVER `'scm'` (ill-conditioned with 16 channels)
- **Bandpass**: 4–40 Hz for stroke (wider than healthy 8–30 Hz — stroke shifts ERD to theta)
- **Artifact rejection**: 150 µV threshold on epochs
- **Cross-validation**: Stratified 5-fold, report accuracy + Cohen's kappa + permutation p-value
- **All classifiers MUST be sklearn Pipeline-compatible** for clean cross-validation
- **Report per-patient results individually**, not just grand average

## Stack

- Python 3.11
- mne >= 1.6
- pyriemann >= 0.5
- braindecode >= 0.8
- moabb >= 1.0
- scikit-learn >= 1.4
- torch >= 2.0
- scipy, numpy, matplotlib, seaborn, pandas, deap, jupyter

## Project Structure

```
stroke-mi-classifier/
├── CLAUDE.md                   ← you are here
├── requirements.txt
├── data/                       ← .mat files go here during hackathon
├── src/
│   ├── __init__.py
│   ├── loading.py              ← load_gtec_stroke_data(), extract_epochs()
│   ├── preprocessing.py        ← filtering, artifact rejection, augmentation
│   ├── classifiers.py          ← build_all_pipelines(), ACM, ensemble
│   ├── transfer.py             ← RPA transfer learning
│   ├── lateralization.py       ← compute_laterality_index()
│   ├── evaluation.py           ← full_evaluation(), permutation testing
│   ├── channel_selection.py    ← csp_rank_channels(), ga_channel_selection()
│   └── visualization.py        ← topomaps, confusion matrices, comparison plots
├── notebooks/
│   ├── 01_explore.ipynb
│   ├── 02_classify.ipynb
│   ├── 03_optimize.ipynb
│   ├── 04_ensemble.ipynb
│   └── 05_presentation.ipynb
└── results/                    ← saved figures and CSV results
```

## Key Domain Knowledge

- Stroke patients have WEAKER motor imagery signals than healthy subjects
- Only ~46% show clear contralateral dominance during MI
- Compensatory activity in unaffected hemisphere is common — include bilateral channels
- **Lateralization Index** (LI) is the key clinical biomarker: LI = (ERD_contra - ERD_ipsi) / (ERD_contra + ERD_ipsi)
- LI correlates with Fugl-Meyer motor scores (r=0.57–0.61)
- Run 1 of each session has FES confound (calibration run) — note this but still use all trials

## What Judges Value

- Thorough comparison of multiple methods with proper statistics
- Clinical relevance (lateralization analysis, per-patient insights)
- Clean visualizations (topomaps, bar charts with error bars, confusion matrices)
- Understanding of WHY Riemannian methods outperform CSP on stroke data
