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

html, body, p, span, div, label, caption {
    font-size: 0.9rem !important; line-height: 1.4 !important;
}
h1 { font-size: 1.5rem !important; font-weight: 800 !important; }
h2 { font-size: 1.2rem !important; font-weight: 700 !important; }

div[data-testid="stNumberInput"] { margin: 0.3rem 0 !important; }
div[data-testid="stNumberInput"] button { display: none !important; }
div[data-testid="stNumberInput"] label {
    font-size: 1rem !important; font-weight: 700 !important;
    color: #0f1825 !important; margin-bottom: 0.2rem !important;
}
div[data-testid="stNumberInput"] input {
    font-size: 1.2rem !important; font-weight: 700 !important;
    padding: 0.5rem !important; border-radius: 8px !important;
    border: 2px solid #cbd5e1 !important; text-align: center !important;
    background: #fff !important; color: #0f1825 !important;
    width: 100% !important; box-sizing: border-box !important;
    box-shadow: none !important; height: auto !important;
}
div[data-testid="stNumberInput"] input:focus { border-color: #1a3a5c !important; }

.stButton > button {
    background: #1a3a5c; color: #fff !important; border: none !important;
    padding: 0.8rem !important; font-size: 1.2rem !important;
    font-weight: 700 !important; border-radius: 10px !important;
    width: 100% !important; box-shadow: none !important; margin: 0.5rem 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem !important; font-weight: 700 !important; padding: 0.5rem 1rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #1a3a5c !important; border-bottom: 3px solid #1a3a5c !important;
}
.stTabs { margin: 0.5rem 0 !important; }

* { box-shadow: none !important; }
input { box-shadow: none !important; }

.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.5rem !important; }

section[data-testid="stSidebar"] { display: none !important; }

hr { margin: 0.5rem 0 !important; }

.stCheckbox { font-size: 0.9rem !important; }
.stCheckbox label { font-size: 0.9rem !important; }
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
<div style="background:linear-gradient(160deg,#0c1929,#162d47,#1a3a5c);border-radius:12px;padding:1.2rem 2rem;margin-bottom:1rem;color:#fff;">
    <h1 style="text-align:center;font-size:1.6rem;font-weight:800;margin:0;color:#fff;">🧫 sepsis-gram-classification</h1>
    <p style="text-align:center;font-size:1rem;opacity:0.75;margin:0.2rem 0 0 0;color:#fff;">Early prediction of Gram‑stain classification in sepsis with bloodstream infection</p>
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
    <div style="background:#fff;border-radius:12px;padding:1rem;text-align:center;border:1px solid #e2e8f0;margin:0.5rem 0;">
        <p style="font-size:1rem;color:#64748b;margin:0;">Predicted Classification</p>
        <h1 style="font-size:2rem;font-weight:900;color:{color};margin:0.3rem 0;">{label}</h1>
        <p style="font-size:1.1rem;font-weight:700;color:#1e293b;">Gram‑positive {proba:.1%}  ·  Gram‑negative {1-proba:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+delta", value=proba*100,
        title={"text":"Gram‑positive Probability (%)","font":{"size":14}},
        delta={"reference":predictor.threshold*100,"increasing":{"color":"#dc2626"}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":10}},
            "bar":{"color":color,"thickness":0.2},
            "threshold":{"line":{"color":"#0f1825","width":2},"value":predictor.threshold*100}
        }
    ))
    fig.update_layout(height=220, margin=dict(t=40,b=5,l=20,r=20),
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
