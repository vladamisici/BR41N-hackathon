# Stroke Rehab Motor Imagery Classification

**BR41N.IO Spring School 2026 — Data Analysis Track**

Patient-stratified left/right hand motor imagery classification from 16-channel EEG recorded with the g.tec recoveriX system. Three chronic stroke patients, pre and post rehabilitation intervention.

## Results

| Condition | Pipeline | Accuracy | vs CSP+LDA | vs PCA+TVLDA |
|-----------|----------|----------|------------|--------------|
| P1 pre | ACM(3,7) | 91.2% | +14.1 pp | −1.7 pp |
| P1 post | FBCSP+LDA | 93.8% | −0.1 pp | −3.2 pp |
| P2 pre | FBCSP+LDA | **83.8%** | +15.3 pp | **+11.4 pp** |
| P2 post | FBCSP+LDA | **100.0%** | +3.9 pp | **+2.6 pp** |
| P3 pre | ACM(3,7) | **97.5%** | +23.1 pp | **+3.9 pp** |
| P3 post | FBCSP+LDA | 93.8% | +14.0 pp | −6.2 pp |

Beat both organizer baselines on 3/6 conditions. Beat CSP+LDA on 5/6. Average improvement: +11.7 pp.

All results statistically validated: permutation test p < 0.005, Cohen's κ ≥ 0.675, binomial significance confirmed on all conditions.

## Key findings

**Epoch window matters more than the classifier.** Switching from the preparation phase (0.5–4.5 s) to the feedback phase (3–7 s) improved P2_pre from 52% to 84%. This is domain knowledge, not hyperparameter tuning — the recoveriX paradigm delivers FES feedback starting at ~3 s, and that's where discriminative MI activity peaks.

**No single pipeline dominates.** FBCSP+LDA wins 4/6 conditions, ACM(3,7) wins 2/6. Patient-stratified selection is necessary.

**Rehabilitation effect is measurable.** Patient 1's lateralization index shifted from −0.297 (right-hemisphere dominant, compensatory) to +0.075 (bilateral, healthier) after intervention. Classification accuracy tracked this improvement (91.2% → 93.8%).

## Pipelines

Eight pipelines compared per patient, in priority order:

1. **FBCSP+LDA** — Filter Bank CSP with 6 sub-bands (4–8, 8–12, 12–16, 16–20, 20–24, 24–30 Hz) + LDA. Captures frequency-shifted ERD that single-band CSP misses.
2. **ACM(3,7)** — Augmented Covariance Method. Takens delay embedding (order=3, lag=7) → OAS covariance → Riemannian tangent space → SVM. Captures temporal dynamics.
3. **TS+LR** — Riemannian tangent space + logistic regression. Robust baseline.
4. **CSP+LDA** — Standard CSP. The baseline to beat.
5. FgMDM, TS+SVM, MDM, TS+LDA — additional Riemannian variants for comparison.

## Preprocessing

- Bandpass: 0.5–30 Hz (Butterworth IIR, order 5)
- Epoch: 3–7 s post-trigger (feedback phase)
- No artifact rejection (80 trials per condition — can't afford to lose any)
- Covariance: OAS estimator (never SCM — ill-conditioned with 16 channels)

## Data

g.tec recoveriX .mat files. 3 patients × 2 stages (pre/post) × 2 splits (train/test) = 12 files. 80 trials per split, balanced left/right. 16 sensorimotor channels at 256 Hz. Trigger codes: +1 left, −1 right.

Montage: FC3, FCz, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, Pz.

## Project structure

```
src/
  loading.py          Data loading, epoch extraction, train/test split
  classifiers.py      All 8 pipelines (FBCSP, ACM, Riemannian, CSP)
  lateralization.py   Lateralization index computation
  evaluation.py       Permutation testing, statistical comparison
  channel_selection.py  CSP-based channel ranking
  transfer.py         Riemannian Procrustes Analysis (cross-patient)
  preprocessing.py    Augmentation strategies
  visualization.py    Publication-quality plots

dashboard/
  app.py              Streamlit interactive dashboard
  dashboard_data.json Pre-computed results

run_hackathon.py      Main analysis script (train/test evaluation)
run_validation_fast.py  Statistical validation suite
run_tuning.py         Parameter sweep (epoch window, bandpass, rejection)
generate_figures.py   Publication-quality figure generation
```

## Infrastructure

- Analysis: AWS EC2 g5.2xlarge (NVIDIA A10G, 8 vCPUs, 32 GB RAM)
- Dashboard: AWS EC2 t3.large (2 vCPUs, 8 GB RAM) running Streamlit
- Stack: Python 3.10, MNE 1.12, pyRiemann 0.11, scikit-learn 1.7, MOABB 1.5

## How to run

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place .mat files in dataset/stroke-rehab/

# Run analysis
python run_hackathon.py

# Run validation
python run_validation_fast.py

# Generate figures
python generate_figures.py

# Start dashboard
streamlit run dashboard/app.py
```

## References

- Irimia et al. (2018). High Classification Accuracy of a Motor Imagery Based BCI for Stroke Rehabilitation Training. *Frontiers in Robotics and AI*.
- Billinger et al. (2013). Is It Significant? Guidelines for Reporting BCI Performance.
- Barachant et al. (2012). Multiclass Brain-Computer Interface Classification by Riemannian Geometry. *IEEE Trans. Biomed. Eng.*
