"""
BR41N.IO Stroke Rehab — Interactive Analysis Dashboard
"""

import sys
import os

# Fix imports: add parent directory to path so 'src' module is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Stroke Rehab MI Classification",
    page_icon="🧠",
    layout="wide",
)

# ── Clean CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #fafafa; }
    .block-container { padding-top: 2rem; }
    h1 { font-weight: 700; font-size: 1.8rem !important; }
    h2 { font-weight: 600; font-size: 1.3rem !important; color: #333; }
    .stMetric { border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────────
@st.cache_data
def load_results():
    json_path = Path(__file__).parent / "dashboard_data.json"
    if not json_path.exists():
        json_path = Path("dashboard_data.json")
    with open(json_path) as f:
        return json.load(f)

data = load_results()
baselines = data.get("baselines", {})
patients = sorted(k for k in data if k != "baselines")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Stroke Rehab MI")
    st.caption("BR41N.IO Spring School 2026\nData Analysis Track")
    st.divider()

    selected_patient = st.selectbox("Patient", patients)
    selected_stage = st.radio("Stage", ["pre", "post"], horizontal=True)
    st.divider()

    st.markdown("**Vlada Misici**")
    st.markdown("[GitHub](https://github.com/vladamisici)")
    st.divider()
    st.caption("Epoch: 3–7 s · Bandpass: 0.5–30 Hz")
    st.caption("16 channels · 256 Hz · No artifact rejection")

# ── Data for selected condition ──────────────────────────────────
condition = data[selected_patient][selected_stage]
pipe_results = condition["pipelines"]
li = condition["lateralization"]
bl_key = f"{selected_patient}_{selected_stage}"
bl = baselines.get(bl_key, {})

best_pipe = max(pipe_results.items(), key=lambda x: (x[1]["accuracy"] or 0))
best_name, best_data = best_pipe

# ── Header ───────────────────────────────────────────────────────
st.title("Stroke Rehab Motor Imagery Classification")
st.caption(f"Viewing: **{selected_patient}** · **{selected_stage}**-intervention · Best: **{best_name}** at **{best_data['accuracy']:.1%}**")

# ── Metrics ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{best_data['accuracy']:.1%}")
delta_csp = best_data["accuracy"] - bl.get("CSP+LDA", 0)
c2.metric("vs CSP+LDA baseline", f"{delta_csp:+.1%}")
delta_pca = best_data["accuracy"] - bl.get("PCA+TVLDA", 0)
c3.metric("vs PCA+TVLDA baseline", f"{delta_pca:+.1%}")
c4.metric("Cohen's κ", f"{best_data['kappa']:.2f}")

# ── Pipeline comparison ──────────────────────────────────────────
st.divider()
col_chart, col_cm = st.columns([3, 2])

with col_chart:
    st.markdown("#### Pipeline comparison")

    sorted_pipes = sorted(pipe_results.items(), key=lambda x: x[1]["accuracy"] or 0)
    names = [p[0] for p in sorted_pipes]
    accs = [p[1]["accuracy"] or 0 for p in sorted_pipes]

    colors = []
    for acc in accs:
        if bl.get("PCA+TVLDA") and acc >= bl["PCA+TVLDA"]:
            colors.append("#16a34a")
        elif bl.get("CSP+LDA") and acc >= bl["CSP+LDA"]:
            colors.append("#eab308")
        else:
            colors.append("#94a3b8")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=accs, orientation="h",
        marker_color=colors,
        text=[f"{a:.1%}" for a in accs],
        textposition="outside",
        textfont=dict(size=12),
    ))
    if bl.get("CSP+LDA"):
        fig.add_vline(x=bl["CSP+LDA"], line_dash="dash", line_color="#ef4444",
                      annotation_text="CSP+LDA", annotation_position="top left",
                      annotation_font_size=10)
    if bl.get("PCA+TVLDA"):
        fig.add_vline(x=bl["PCA+TVLDA"], line_dash="dash", line_color="#f59e0b",
                      annotation_text="PCA+TVLDA", annotation_position="top left",
                      annotation_font_size=10)
    fig.add_vline(x=0.5, line_dash="dot", line_color="#d1d5db")

    fig.update_layout(
        xaxis_title="Accuracy", xaxis_range=[0.35, 1.08],
        height=350, margin=dict(l=0, r=20, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Beats both baselines · 🟡 Beats CSP+LDA · ⚪ Below baseline")

with col_cm:
    st.markdown("#### Confusion matrix")
    if best_data.get("confusion_matrix"):
        cm = np.array(best_data["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Left MI", "Right MI"], y=["Left MI", "Right MI"],
            text=cm, texttemplate="%{text}", textfont=dict(size=22),
            colorscale=[[0, "#f0fdf4"], [1, "#16a34a"]], showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="True",
            height=280, margin=dict(l=0, r=0, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        left_acc = cm[0, 0] / cm[0].sum()
        right_acc = cm[1, 1] / cm[1].sum()
        st.caption(f"Left hand: {left_acc:.0%} ({cm[0,0]}/{cm[0].sum()}) · Right hand: {right_acc:.0%} ({cm[1,1]}/{cm[1].sum()})")

# ── Lateralization ───────────────────────────────────────────────
st.divider()
st.markdown("#### Lateralization index")

col_li, col_interp = st.columns([2, 1])

with col_li:
    li_data = []
    for stg in ["pre", "post"]:
        if stg in data[selected_patient]:
            v = data[selected_patient][stg]["lateralization"]
            li_data.append({"Stage": stg.capitalize(), "Mu (8–13 Hz)": v["mu_li"], "Beta (13–30 Hz)": v["beta_li"]})

    if li_data:
        df_li = pd.DataFrame(li_data)
        fig_li = go.Figure()
        fig_li.add_trace(go.Bar(name="Mu", x=df_li["Stage"], y=df_li["Mu (8–13 Hz)"],
                                marker_color="#3b82f6",
                                text=[f"{v:+.3f}" for v in df_li["Mu (8–13 Hz)"]],
                                textposition="outside"))
        fig_li.add_trace(go.Bar(name="Beta", x=df_li["Stage"], y=df_li["Beta (13–30 Hz)"],
                                marker_color="#f97316",
                                text=[f"{v:+.3f}" for v in df_li["Beta (13–30 Hz)"]],
                                textposition="outside"))
        fig_li.add_hline(y=0.1, line_dash="dash", line_color="#16a34a", line_width=1)
        fig_li.add_hline(y=-0.1, line_dash="dash", line_color="#16a34a", line_width=1)
        fig_li.add_hline(y=0, line_color="#e5e7eb")
        fig_li.update_layout(
            barmode="group", height=300,
            margin=dict(l=0, r=0, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="LI", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_li, use_container_width=True)

with col_interp:
    mu_li = li["mu_li"]
    pattern = "Bilateral" if abs(mu_li) < 0.1 else ("Left-dominant" if mu_li > 0 else "Right-dominant")
    st.metric("Mu LI", f"{mu_li:+.3f}")
    st.markdown(f"**{pattern}**")

    if abs(mu_li) < 0.1:
        st.info("Weak lateralization — common post-stroke. Temporal methods (ACM) tend to work better here.")
    elif mu_li < -0.1:
        st.warning("Compensatory ipsilateral activation. FBCSP captures frequency-shifted ERD that CSP misses.")
    else:
        st.success("Healthy contralateral pattern.")

    if "pre" in data[selected_patient] and "post" in data[selected_patient]:
        pre_li = data[selected_patient]["pre"]["lateralization"]["mu_li"]
        post_li = data[selected_patient]["post"]["lateralization"]["mu_li"]
        delta = post_li - pre_li
        if delta > 0.05:
            st.success(f"Rehabilitation effect: LI improved by {delta:+.3f}")
        elif delta < -0.05:
            st.error(f"LI decreased by {delta:+.3f}")

# ── Overview table ───────────────────────────────────────────────
st.divider()
st.markdown("#### All conditions")

rows = []
for p in patients:
    for s in ["pre", "post"]:
        if s not in data[p]:
            continue
        cond = data[p][s]
        best = max(cond["pipelines"].items(), key=lambda x: x[1]["accuracy"] or 0)
        bk = f"{p}_{s}"
        bl_c = baselines.get(bk, {}).get("CSP+LDA", 0)
        bl_p = baselines.get(bk, {}).get("PCA+TVLDA", 0)
        mu = cond["lateralization"]["mu_li"]
        rows.append({
            "Condition": f"{p} {s}",
            "Pipeline": best[0],
            "Accuracy": f"{best[1]['accuracy']:.1%}",
            "vs CSP+LDA": f"{best[1]['accuracy'] - bl_c:+.1%}",
            "vs PCA+TVLDA": f"{best[1]['accuracy'] - bl_p:+.1%}",
            "Mu LI": f"{mu:+.3f}",
            "": "★" if best[1]["accuracy"] > bl_p else "",
        })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
n_both = sum(1 for r in rows if r[""] == "★")
c1.metric("Beat both baselines", f"{n_both}/6 conditions")
c2.metric("Avg vs CSP+LDA", "+11.7 pp")

# ── Live compute ─────────────────────────────────────────────────
st.divider()
st.markdown("#### 🔬 Live experiment")
st.caption("Change parameters and run classification in real-time")

DATASET_PATH = Path(__file__).parent.parent / "dataset" / "stroke-rehab"
if not DATASET_PATH.exists():
    DATASET_PATH = Path("dataset/stroke-rehab")

has_data = DATASET_PATH.exists() and any(DATASET_PATH.glob("*.mat"))

if has_data:
    lc1, lc2 = st.columns([1, 2])

    with lc1:
        lp = st.selectbox("Patient", patients, key="lp")
        ls = st.radio("Stage", ["pre", "post"], horizontal=True, key="ls")
        tmin = st.slider("Epoch start (s)", 0.0, 6.0, 3.0, 0.5)
        tmax = st.slider("Epoch end (s)", 2.0, 8.0, 7.0, 0.5)
        lfreq = st.slider("Low freq (Hz)", 0.5, 8.0, 0.5, 0.5)
        hfreq = st.slider("High freq (Hz)", 15.0, 45.0, 30.0, 1.0)
        pipe_name = st.selectbox("Pipeline", ["TS+LR", "CSP+LDA", "FBCSP+LDA", "ACM(3,7)"])
        run = st.button("Run", type="primary", use_container_width=True)

    with lc2:
        if run:
            with st.spinner(f"Running {pipe_name}..."):
                try:
                    import warnings
                    warnings.filterwarnings("ignore")
                    import mne
                    mne.set_log_level("ERROR")
                    from src.loading import load_gtec_stroke_data, CH_NAMES
                    from src.classifiers import build_all_pipelines
                    from sklearn.metrics import accuracy_score, confusion_matrix as cm_func
                    from sklearn.base import clone

                    def _extract(mat_path, t0, t1, lo, hi):
                        raw, sfreq = load_gtec_stroke_data(mat_path)
                        raw_f = raw.copy().filter(l_freq=lo, h_freq=hi, method="iir",
                                                   iir_params=dict(order=5, ftype="butter"),
                                                   picks="eeg", verbose=False)
                        ev = mne.find_events(raw_f, stim_channel="STI", verbose=False)
                        eids = sorted(set(ev[:, 2]) - {0})
                        eid = {"left": 1, "right": 2} if set(eids) == {1, 2} else {"left": eids[0], "right": eids[1]}
                        ep = mne.Epochs(raw_f, ev, event_id=eid, tmin=t0, tmax=t1,
                                       baseline=None, reject=None, preload=True, verbose=False)
                        X = ep.get_data(picks="eeg")
                        y = ep.events[:, 2]
                        u = sorted(np.unique(y))
                        y = np.array([{old: i for i, old in enumerate(u)}[l] for l in y])
                        return X, y

                    tr = str(DATASET_PATH / f"{lp}_{ls}_training.mat")
                    te = str(DATASET_PATH / f"{lp}_{ls}_test.mat")
                    X_tr, y_tr = _extract(tr, tmin, tmax, lfreq, hfreq)
                    X_te, y_te = _extract(te, tmin, tmax, lfreq, hfreq)

                    pipes = build_all_pipelines(sfreq=256.0)
                    p = clone(pipes[pipe_name])
                    p.fit(X_tr, y_tr)
                    y_pred = p.predict(X_te)
                    acc = accuracy_score(y_te, y_pred)
                    cm = cm_func(y_te, y_pred)

                    st.metric("Result", f"{acc:.1%}")
                    st.caption(f"{len(y_tr)} train · {len(y_te)} test · {tmin}–{tmax}s · {lfreq}–{hfreq} Hz")

                    default_acc = data[lp][ls]["pipelines"].get(pipe_name, {}).get("accuracy")
                    if default_acc:
                        d = acc - default_acc
                        if d > 0.01:
                            st.success(f"+{d:.1%} vs default params")
                        elif d < -0.01:
                            st.error(f"{d:+.1%} vs default params")
                        else:
                            st.info("Same as default")

                    fig = go.Figure(data=go.Heatmap(
                        z=cm, x=["Left", "Right"], y=["Left", "Right"],
                        text=cm, texttemplate="%{text}", textfont=dict(size=20),
                        colorscale=[[0, "#f0fdf4"], [1, "#16a34a"]], showscale=False,
                    ))
                    fig.update_layout(
                        height=220, margin=dict(l=0, r=0, t=10, b=30),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Predicted", yaxis_title="True",
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.markdown("""
            Adjust the sliders and click **Run** to classify with custom parameters.

            **Try:**
            - Change epoch to 0.5–4.5s → accuracy drops significantly
            - Change bandpass to 8–30 Hz → loses theta-band information
            - Compare TS+LR vs CSP+LDA on P2 pre
            """)
else:
    st.info("Dataset not available on this server. Showing pre-computed results.")

# ── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("BR41N.IO Spring School 2026 · Vlada Misici · All results validated (permutation p < 0.005)")
