"""
sepsis-gram-classification
"""
import streamlit as st, traceback, sys
from pathlib import Path

st.set_page_config(page_title="sepsis-gram-classification", page_icon="🧫", layout="wide",
                   initial_sidebar_state="collapsed")

MODEL_OK, error_msg = False, ""
try:
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    from model_predictor import SepsisPredictor
    @st.cache_resource
    def load_predictor():
        return SepsisPredictor(str(Path(__file__).parent / "best_models_fixed_leakage"))
    predictor = load_predictor(); MODEL_OK = True
except Exception as e:
    error_msg = traceback.format_exc()

st.markdown("""
<style>
.stApp { background: #f8fafc; }

/* ── Base text ── */
html, body, p, span, div, label, caption {
    font-size: 2rem !important; line-height: 1.6 !important;
}
h1 { font-size: 3rem !important; font-weight: 900 !important; }
h2 { font-size: 2.6rem !important; font-weight: 800 !important; }

/* ── Number input ── */
div[data-testid="stNumberInput"] { margin: 3rem 0 !important; }
div[data-testid="stNumberInput"] button { display: none !important; }
div[data-testid="stNumberInput"] label {
    font-size: 2.6rem !important; font-weight: 800 !important;
    color: #0f1825 !important; margin-bottom: 0.8rem !important;
}
div[data-testid="stNumberInput"] input {
    font-size: 3rem !important; font-weight: 800 !important;
    padding: 1.5rem !important; border-radius: 14px !important;
    border: 2.5px solid #cbd5e1 !important; text-align: center !important;
    background: #fff !important; color: #0f1825 !important;
    width: 100% !important; box-sizing: border-box !important;
    box-shadow: none !important; height: auto !important;
}
div[data-testid="stNumberInput"] input:focus { border-color: #1a3a5c !important; }

/* ── Button ── */
.stButton > button {
    background: #1a3a5c; color: #fff !important; border: none !important;
    padding: 2rem !important; font-size: 3rem !important;
    font-weight: 800 !important; border-radius: 14px !important;
    width: 100% !important; box-shadow: none !important; margin: 3rem 0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
    font-size: 2rem !important; font-weight: 800 !important; padding: 2rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1a3a5c !important; border-bottom: 4px solid #1a3a5c !important;
}
.stTabs { margin: 3rem 0 !important; }

/* ── No shadows ── */
* { box-shadow: none !important; }
input { box-shadow: none !important; }

/* ── Full-width tall layout ── */
.block-container { padding: 3rem 2rem 6rem 2rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 2.5rem !important; }

/* ── No sidebar ── */
section[data-testid="stSidebar"] { display: none !important; }

/* ── Divider spacing ── */
hr { margin: 3rem 0 !important; }
</style>
""", unsafe_allow_html=True)

if not MODEL_OK:
    st.error("Model loading — please wait for rebuild.")
    with st.expander("Details"): st.code(error_msg[:3000])
    st.stop()

import pandas as pd, numpy as np, plotly.graph_objects as go

FEATURES = [
    ("pt",          "Prothrombin Time",           "seconds",       8.0, 80.0,   14.5, 0.1),
    ("platelet",    "Platelet Count",             "10⁹/L",         5.0, 1500.0, 185.0, 1.0),
    ("hemoglobin",  "Hemoglobin",                 "g/dL",          3.0, 22.0,   10.0, 0.1),
    ("bicarbonate", "Bicarbonate (HCO₃⁻)",         "mmol/L",        5.0, 45.0,   23.0, 1.0),
    ("resp_rate",   "Respiratory Rate",           "breaths / min", 5.0, 60.0,   19.0, 1.0),
]

WINDOWS = [
    ("p3", "Period 3"),
    ("p2", "Period 2"),
    ("p1", "Period 1"),
]

# ═══ Hero ═══
st.markdown("""
<div style="background:linear-gradient(160deg,#0c1929,#162d47,#1a3a5c);border-radius:20px;padding:5rem 3rem;margin-bottom:4rem;color:#fff;">
    <h1 style="text-align:center;font-size:4rem;font-weight:900;margin:0;color:#fff;">🧫 sepsis-gram-classification</h1>
    <p style="text-align:center;font-size:2.2rem;opacity:0.75;margin:0.5rem 0 0 0;color:#fff;">Early prediction of Gram‑stain classification in sepsis with bloodstream infection</p>
</div>
""", unsafe_allow_html=True)

# ═══ Quick toggle ═══
quick = st.checkbox("Auto‑fill all time windows (use Period 3 values for all)", value=True)
st.divider()

# ═══ Input ═══
st.markdown("## Clinical Parameters")

tabs = st.tabs([w[1] for w in WINDOWS])
all_inputs = {}

for tab, (pkey, plabel) in zip(tabs, WINDOWS):
    with tab:
        for key, name, unit, vmin, vmax, default, step in FEATURES:
            val = st.number_input(
                f"{name}  ({unit})",
                min_value=vmin, max_value=vmax, value=default, step=step,
                key=f"{pkey}_{key}"
            )
            all_inputs[f"{pkey}_{key}"] = val

# ═══ Predict ═══
btn = st.button("🔬 Predict Gram Classification", type="primary", use_container_width=True)

if btn:
    with st.spinner("Computing..."):
        feat_keys = [f[0] for f in FEATURES]
        vals = []
        for pk in ['p3','p2','p1']:
            if quick and pk != 'p3':
                vals.append(np.array([all_inputs[f"p3_{k}"] for k in feat_keys]))
            else:
                vals.append(np.array([all_inputs[f"{pk}_{k}"] for k in feat_keys]))
        proba = predictor.predict(vals)
        is_pos = proba > predictor.threshold

    st.divider()
    label = "Gram‑positive" if is_pos else "Gram‑negative"
    color = "#dc2626" if is_pos else "#16a34a"

    st.markdown(f"""
    <div style="background:#fff;border-radius:24px;padding:3rem;text-align:center;border:2px solid #e2e8f0;margin:2rem 0;">
        <p style="font-size:2.2rem;color:#64748b;margin:0;">Predicted Classification</p>
        <h1 style="font-size:5rem;font-weight:900;color:{color};margin:0.8rem 0;">{label}</h1>
        <p style="font-size:2.2rem;font-weight:700;color:#1e293b;">Gram‑positive {proba:.1%}  ·  Gram‑negative {1-proba:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+delta", value=proba*100,
        title={"text":"Gram‑positive Probability (%)","font":{"size":20}},
        delta={"reference":predictor.threshold*100,"increasing":{"color":"#dc2626"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":14}},
            "bar":{"color":color,"thickness":0.2},
            "threshold":{"line":{"color":"#0f1825","width":3},"value":predictor.threshold*100}
        }
    ))
    fig.update_layout(height=380, margin=dict(t=60,b=10,l=30,r=30),
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("📁 Batch Prediction (CSV)"):
    feat_keys = [f[0] for f in FEATURES]
    tcols = [f"{k}_{s}" for s in ['p3','p2','p1'] for k in feat_keys]
    st.download_button("📥 Template CSV", pd.DataFrame(columns=tcols).to_csv(index=False), "template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type=['csv'], key="b")
    if up and st.button("Run Batch", type="primary"):
        data = pd.read_csv(up)
        X = np.zeros((len(data),3,len(feat_keys)))
        for i,k in enumerate(feat_keys):
            for t,s in enumerate(['p3','p2','p1']):
                c = f"{k}_{s}"
                if c in data.columns: X[:,t,i] = data[c].values
        probs = predictor.predict_temporal(X)
        res = data.copy(); res['Gram_Positive_Prob'] = probs
        res['Prediction'] = ['Gram-positive' if p > predictor.threshold else 'Gram-negative' for p in probs]
        st.dataframe(res[['Gram_Positive_Prob','Prediction']], use_container_width=True)
        st.download_button("📥 Results", res.to_csv(index=False), "results.csv", "text/csv")

st.divider()
st.markdown("<div style='text-align:center;color:#94a3b8;padding:2rem;'>v6.4 · For Research Use Only</div>", unsafe_allow_html=True)
