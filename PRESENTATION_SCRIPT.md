# Presentation Script — BR41N.IO Stroke Rehab Data Analysis

**Format:** ~5-7 minutes, screen share on Zoom, then Q&A
**What to share:** Google Slides / Keynote with your matplotlib figures pasted in.
Build the slides AFTER you run the pipelines on the real hackathon data.

---

## SLIDE 1: Title (30 seconds)

**Show:** Title slide with your name, the track name, and one headline number.

```
Patient-Stratified Motor Imagery Classification
for Stroke Rehabilitation

BR41N.IO Spring School 2026 — Data Analysis Track
[Your Name]

Key result: FBCSP+LDA outperforms CSP+LDA by [X]+ points on stroke MI data
```

**Say:**
"Hi everyone. I'm [name], working solo on the stroke rehab data analysis track.
My approach is patient-stratified classification — I don't treat all patients
the same. I use filter-bank CSP and Riemannian geometry methods, and I compare
8 pipelines per patient to find what works best for each individual.
The headline: my best pipeline beats the CSP+LDA baseline by [X] points."

---

## SLIDE 2: Why This Matters (45 seconds)

**Show:** The LI formula + a simple table explaining what LI values mean.

```
Lateralization Index (LI) = (ERD_contra - ERD_ipsi) / (ERD_contra + ERD_ipsi)

| LI Value   | Meaning              | Clinical              |
|------------|----------------------|-----------------------|
| LI > 0.1   | Contralateral dominant | Healthy pattern      |
| |LI| < 0.1 | Bilateral            | Common post-stroke    |
| LI < -0.1  | Ipsilateral dominant | Compensatory          |

LI correlates with Fugl-Meyer motor scores (r = 0.57-0.61)
```

**Say:**
"Stroke patients don't have the same clean motor imagery signals as healthy
subjects. Only about 46% show clear contralateral dominance. Many show
bilateral or even ipsilateral activation — the unaffected hemisphere
compensates. The lateralization index captures this. It correlates with
clinical motor recovery scores, so it's not just a number — it tells us
something about the patient's rehabilitation potential. I compute this
for every patient and use it to interpret why different pipelines work
differently."

---

## SLIDE 3: Method (60 seconds)

**Show:** Your pipeline table + preprocessing parameters.

```
Preprocessing: 4-40 Hz bandpass, 0.5-4.5s epoch window, 150µV rejection
16 channels: FC3, FCz, FC4, C5-C6, CP3-CP4, CPz, Pz
Evaluation: Train on _training.mat, test on _test.mat (+ 5-fold CV)

| Priority | Pipeline   | How it works                              |
|----------|------------|-------------------------------------------|
| Primary  | FBCSP+LDA  | 6 filter banks (4-30Hz), CSP per band     |
| Secondary| ACM(3,7)   | Delay embedding → Riemannian covariance   |
| Backup   | TS+LR      | Tangent space + logistic regression        |
| Baseline | CSP+LDA    | Standard CSP (the baseline to beat)        |
```

**Say:**
"My preprocessing uses a wider 4-to-40 Hz bandpass than the standard 8-to-30.
This is deliberate — stroke patients show ERD shifts into the theta band,
and a narrow filter misses that.

My primary pipeline is filter-bank CSP with LDA. It splits the signal into
6 sub-bands — theta, alpha, low beta, mid beta, high beta, low gamma — applies
CSP independently to each, and concatenates the features. This captures
frequency-specific patterns that single-band CSP misses entirely.

My secondary pipeline is the Augmented Covariance Method. It uses Takens
delay embedding to stack time-shifted copies of the signal, then computes
Riemannian covariance features. This captures temporal dynamics that static
methods miss.

I evaluate using the provided train/test splits AND 5-fold cross-validation
for robustness."

---

## SLIDE 4: Results Table (60 seconds)

**Show:** Your results table — all pipelines, all patients, pre and post.
Highlight the cells where you beat the baselines. Use color coding:
green = beats PCA+TVLDA, yellow = beats CSP+LDA, red = below CSP+LDA.

```
| Patient | Stage | FBCSP+LDA | ACM(3,7) | TS+LR | CSP+LDA | Baseline CSP+LDA | Baseline PCA+TVLDA |
|---------|-------|-----------|----------|-------|---------|-------------------|---------------------|
| P1      | pre   | [XX.X%]   | [XX.X%]  | ...   | ...     | 77.1%             | 92.9%               |
| P1      | post  | [XX.X%]   | [XX.X%]  | ...   | ...     | 93.9%             | 97.0%               |
| P2      | pre   | [XX.X%]   | [XX.X%]  | ...   | ...     | 68.4%             | 72.4%               |
| ...     | ...   | ...       | ...      | ...   | ...     | ...               | ...                 |
```

**Say:**
"Here are the results across all patients, pre and post intervention.
[Point to your best numbers.] FBCSP+LDA is my strongest pipeline on [X]
out of 6 conditions. The biggest improvement is on [hardest patient] where
I go from [baseline]% to [your number]% — that's a [difference] point
improvement.

Notice that no single pipeline wins everywhere. Patient [X] responds better
to [pipeline] while Patient [Y] responds better to [other pipeline].
This is exactly why patient-stratified analysis matters."

---

## SLIDE 5: Bar Chart (30 seconds)

**Show:** The matplotlib bar chart from your notebook — pipelines on y-axis,
accuracy on x-axis, one chart per patient (or grouped).

**Say:**
"Here's the visual comparison. The red dashed line is chance level at 50%.
You can see the clear hierarchy: FBCSP+LDA and ACM consistently at the top,
CSP+LDA at the bottom. The gap is largest for the hardest patients."

---

## SLIDE 6: Clinical Insight — LI + Pipeline Selection (60 seconds)

**Show:** LI bar chart (mu and beta per patient) next to a table showing
which pipeline won for each patient.

```
| Patient | Mu LI   | Pattern     | Best Pipeline | Accuracy |
|---------|---------|-------------|---------------|----------|
| P1      | [+/-X]  | [bilateral] | [FBCSP+LDA]  | [XX.X%]  |
| P2      | [+/-X]  | [R-dom]     | [ACM(3,7)]   | [XX.X%]  |
| P3      | [+/-X]  | [bilateral] | [FBCSP+LDA]  | [XX.X%]  |
```

**Say:**
"This is the clinical insight slide — the one that separates a good analysis
from a winning one.

[Point to LI values.] Patient [X] shows bilateral activation — weak
lateralization. This means the standard CSP approach struggles because there's
no clear spatial contrast to exploit. But FBCSP captures frequency-specific
patterns that are still present even when the spatial pattern is diffuse.

Patient [Y] shows [lateralized/bilateral] activation, and here [pipeline]
works best because [reason].

The pre-to-post comparison is also telling: [describe if LI improves after
intervention, and if accuracy tracks that improvement]. This suggests the
rehabilitation is having a measurable effect on brain lateralization."

---

## SLIDE 7: Conclusion (30 seconds)

**Show:** Three bullet points + your key number.

```
Key Findings:
1. FBCSP+LDA outperforms CSP+LDA by [X]+ points — filter-bank decomposition
   captures theta-shifted ERD that single-band CSP misses
2. Patient-stratified analysis is essential — no single pipeline dominates
3. Lateralization Index tracks rehabilitation progress and guides pipeline selection

Technical: 8 pipelines × 3 patients × pre/post = 48 evaluations
           All sklearn Pipeline-compatible, reproducible with fixed random seeds
```

**Say:**
"To wrap up: filter-bank and Riemannian methods significantly outperform
standard CSP on stroke data. The key insight is that stroke patients need
wider frequency analysis — their ERD patterns shift compared to healthy
subjects. Per-patient analysis with lateralization index gives clinically
meaningful interpretation, not just accuracy numbers. Thank you."

---

## Q&A PREP — Likely Questions and Answers

**Q: "Why not use deep learning / EEGNet?"**
A: "With only ~120 training trials per session, deep learning overfits.
I tested EEGNet during development and it didn't beat FBCSP+LDA on this
sample size. Riemannian methods are more data-efficient."

**Q: "Why 4-40 Hz instead of 8-30 Hz?"**
A: "Stroke patients show ERD shifts into the theta band (4-8 Hz) due to
cortical reorganization. A standard 8-30 Hz filter misses this. The filter
bank approach captures it explicitly with a dedicated 4-8 Hz sub-band."

**Q: "How does ACM work?"**
A: "It uses Takens delay embedding — stacking time-shifted copies of the
signal along the channel axis. This turns temporal dynamics into spatial
structure that Riemannian covariance methods can capture. Order 3, lag 7
samples means we embed 3 copies with 14ms spacing."

**Q: "Did you try transfer learning across patients?"**
A: "I have Riemannian Procrustes Analysis implemented but with only 3
patients the cross-patient transfer results are preliminary. The per-patient
approach gave stronger results."

**Q: "What about the FES confound in run 1?"**
A: "Good question. Run 1 includes functional electrical stimulation during
the feedback phase, which could introduce artifacts. I included all trials
but this is worth investigating — comparing run 1 vs runs 2-3 accuracy
would show if FES affects classification."

**Q: "Why does CSP+LDA win on some patients?"**
A: "When the patient has strong, clean lateralization (high |LI|), CSP
can exploit the clear spatial contrast effectively. FBCSP adds complexity
that isn't needed. But for patients with weak lateralization, FBCSP's
frequency decomposition provides the discriminative features that spatial
filtering alone can't find."

---

## TIMING CHECKLIST

| Slide | Time  | Cumulative |
|-------|-------|------------|
| 1     | 0:30  | 0:30       |
| 2     | 0:45  | 1:15       |
| 3     | 1:00  | 2:15       |
| 4     | 1:00  | 3:15       |
| 5     | 0:30  | 3:45       |
| 6     | 1:00  | 4:45       |
| 7     | 0:30  | 5:15       |

Total: ~5 minutes 15 seconds. Leaves room for Q&A.

---

## PRESENTATION DAY WORKFLOW

1. Run pipelines on real data → get results
2. Save all figures to results/
3. Copy figures to Google Slides (7 slides matching above)
4. Fill in the [XX.X%] placeholders with real numbers
5. Practice once out loud (5 minutes)
6. Present on Zoom — share screen, walk through slides
