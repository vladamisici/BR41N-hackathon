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

### 16-Channel Montage (10/20, confirmed from hackathon slides)

```
     FC3  FCz  FC4
C5  C3  C1  Cz  C2  C4  C6
     CP3  CP1  CPz  CP2  CP4
               Pz
```

Channel order in .mat files: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, Pz

### Hemisphere Indices

- LEFT:    [0,3,4,5,10,11]     → FC3, C5, C3, C1, CP3, CP1
- RIGHT:   [2,7,8,9,13,14]     → FC4, C2, C4, C6, CP2, CP4
- MIDLINE: [1,6,12,15]         → FCz, Cz, CPz, Pz

### Variable Description (from hackathon slides)

- `fs` — Sampling rate in Hz
- `y` — EEG data (samples × channels)
- `trig` — Trigger data: **+1 = left movement, -1 = right movement**

### File Structure

12 files: 3 patients × 2 stages (pre/post) × 2 splits (training/test)
```
P1_pre_training.mat   P1_pre_test.mat
P1_post_training.mat  P1_post_test.mat
P2_pre_training.mat   P2_pre_test.mat
P2_post_training.mat  P2_post_test.mat
P3_pre_training.mat   P3_pre_test.mat
P3_post_training.mat  P3_post_test.mat
```

### Trial Paradigm

- T=0s beep → T=2s cue (left/right) → T=3.5s feedback → T=8s relax
- 240 trials per session (3 runs × 80), balanced left/right
- Epoch window: **0.5–4.5s post-cue** (captures MI, avoids evoked response)

## Strategy (Priority Order — validated on BNCI2014_001)

1. **FBCSP+LDA (primary)** — Filter Bank CSP with 6 sub-bands (4–8, 8–12, 12–16, 16–20, 20–24, 24–30 Hz). Best overall on healthy data (+26pp over CSP on hard subjects). Critical for stroke where ERD shifts to theta.
2. **ACM(3,7) (secondary)** — Augmented Covariance Method with Takens delay embedding (order=3, lag=7). Beats TS+LR on all tested subjects by capturing temporal dynamics in the covariance matrix.
3. **TS+LR (reliable backup)** — Riemannian tangent space + logistic regression. Robust to noise, no hyperparameters to tune, works with 160 trials.
4. **CSP+LDA (baseline to beat)** — Standard CSP baseline. Report this to show improvement.
5. **Additional Riemannian (comparison)** — FgMDM, TS+SVM, MDM, TS+LDA for thorough method comparison.
6. **Transfer learning** — Riemannian Procrustes Analysis if time allows.

## Technical Rules

- **Covariance estimation**: Always use `'oas'` or `'lwf'`, NEVER `'scm'` (ill-conditioned with 16 channels)
- **Bandpass**: 0.5–30 Hz (matches recoveriX paper; wider than standard 8–30 Hz)
- **Epoch window**: 3–7s post-trigger (feedback phase — where discriminative MI activity peaks)
- **Artifact rejection**: DISABLED for stroke data (150µV threshold discards too many valid trials)
- **Cross-validation**: Stratified 5-fold, report accuracy + Cohen's kappa + permutation p-value
- **All classifiers MUST be sklearn Pipeline-compatible** for clean cross-validation
- **Report per-patient results individually**, not just grand average

## Stack

- Python 3.11
- mne >= 1.6
- pyriemann >= 0.5
- moabb >= 1.0
- scikit-learn >= 1.4
- scipy, numpy, matplotlib, seaborn, pandas, jupyter

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
│   ├── classifiers.py          ← build_all_pipelines(), FBCSP+LDA, ACM
│   ├── transfer.py             ← RPA transfer learning
│   ├── lateralization.py       ← compute_laterality_index()
│   ├── evaluation.py           ← full_evaluation(), permutation testing
│   ├── channel_selection.py    ← csp_rank_channels()
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
