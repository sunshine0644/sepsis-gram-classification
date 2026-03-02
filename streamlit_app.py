# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import joblib
import json
import os
import sys
from pathlib import Path
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Add backend path to system path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

# Import predictor
try:
    from model_predictor import SepsisPredictor
except ImportError:
    st.error("Failed to import SepsisPredictor. Please check backend/model_predictor.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Gram Classification System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header - 深蓝色底白色字 */
    .main-header {
        background: linear-gradient(135deg, #0a1929 0%, #1a2b3c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
    }
    
    .main-header p {
        color: #e0e0e0 !important;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    .gram-positive {
        color: #dc3545;
        font-size: 3rem;
        font-weight: bold;
    }
    
    .gram-negative {
        color: #28a745;
        font-size: 3rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #2a5298;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card h3 {
        margin: 0;
        color: #495057;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .metric-card .value {
        margin: 0.5rem 0 0 0;
        color: #2a5298;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.4);
    }
    
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .uploadedFile {
        border: 2px dashed #2a5298;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* 时间点标签页样式 */
    .timepoint-tab {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* 时间点摘要卡片 */
    .period-summary {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = [
        "heart_rate", "sbp", "resp_rate", "spo2", "wbc", 
        "hemoglobin", "platelet", "bun", "pt", "glucose", 
        "sodium", "potassium", "chloride", "bicarbonate"
    ]

# Header
st.markdown("""
<div class="main-header">
    <h1>🦠 Gram Classification System</h1>
    <p style="font-size: 1.2rem; opacity: 0.9;">Machine Learning-based Gram-positive vs Gram-negative Classification for Sepsis with Bloodstream Infection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bacteria.png", width=80)
    st.markdown("## 📋 Control Panel")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                st.session_state.predictor = SepsisPredictor()
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Model loading failed: {e}")
    
    # Model info
    if st.session_state.model_loaded:
        with st.expander("ℹ️ Model Information", expanded=True):
            st.markdown(f"""
            - **Features**: {len(st.session_state.feature_cols)}
            - **Model**: LightGBM Classifier
            - **Task**: Gram Classification
            - **Output**: Probability (Gram-positive)
            - **Time Points**: 3 periods (0-8h, 8-16h, 16-24h)
            """)
    
    # Navigation
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Function",
        ["🎯 Single Patient (3 Time Points)", "📊 Batch Prediction", "📋 Sample Cases", "ℹ️ About"],
        label_visibility="collapsed"
    )

# Feature descriptions for tooltips
feature_descriptions = {
    "heart_rate": "Heart rate (beats per minute)",
    "sbp": "Systolic blood pressure (mmHg)",
    "resp_rate": "Respiratory rate (breaths per minute)",
    "spo2": "Oxygen saturation (%)",
    "wbc": "White blood cell count (10⁹/L)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "platelet": "Platelet count (10⁹/L)",
    "bun": "Blood urea nitrogen (mg/dL)",
    "pt": "Prothrombin time (seconds)",
    "glucose": "Glucose (mg/dL)",
    "sodium": "Sodium (mmol/L)",
    "potassium": "Potassium (mmol/L)",
    "chloride": "Chloride (mmol/L)",
    "bicarbonate": "Bicarbonate (mmol/L)"
}

# 修改后的函数：为特定时间点创建输入字段
def create_timepoint_inputs(timepoint_name, timepoint_idx, default_values=None):
    """为特定时间点创建输入字段"""
    if default_values is None:
        default_values = {
            "heart_rate": 80, "sbp": 120, "resp_rate": 16, "spo2": 98,
            "wbc": 8.5, "hemoglobin": 13.5, "platelet": 250,
            "bun": 15.0, "pt": 12.0, "glucose": 100,
            "sodium": 140, "potassium": 4.0, "chloride": 102, "bicarbonate": 24
        }
    
    col1, col2 = st.columns(2)
    values = {}
    
    with col1:
        st.markdown("##### Vital Signs")
        values["heart_rate"] = st.number_input(
            f"Heart Rate (bpm) - {timepoint_name}", 
            min_value=0, max_value=300, value=default_values["heart_rate"], step=1,
            key=f"hr_{timepoint_idx}",
            help=feature_descriptions["heart_rate"]
        )
        values["sbp"] = st.number_input(
            f"SBP (mmHg) - {timepoint_name}", 
            min_value=0, max_value=300, value=default_values["sbp"], step=1,
            key=f"sbp_{timepoint_idx}",
            help=feature_descriptions["sbp"]
        )
        values["resp_rate"] = st.number_input(
            f"Respiratory Rate (breaths/min) - {timepoint_name}", 
            min_value=0, max_value=100, value=default_values["resp_rate"], step=1,
            key=f"rr_{timepoint_idx}",
            help=feature_descriptions["resp_rate"]
        )
        values["spo2"] = st.number_input(
            f"SpO₂ (%) - {timepoint_name}", 
            min_value=0, max_value=100, value=default_values["spo2"], step=1,
            key=f"spo2_{timepoint_idx}",
            help=feature_descriptions["spo2"]
        )
        
        st.markdown("##### Blood Count")
        values["wbc"] = st.number_input(
            f"WBC (10⁹/L) - {timepoint_name}", 
            min_value=0.0, max_value=100.0, value=default_values["wbc"], step=0.1,
            key=f"wbc_{timepoint_idx}",
            help=feature_descriptions["wbc"]
        )
        values["hemoglobin"] = st.number_input(
            f"Hemoglobin (g/dL) - {timepoint_name}", 
            min_value=0.0, max_value=20.0, value=default_values["hemoglobin"], step=0.1,
            key=f"hgb_{timepoint_idx}",
            help=feature_descriptions["hemoglobin"]
        )
        values["platelet"] = st.number_input(
            f"Platelet (10⁹/L) - {timepoint_name}", 
            min_value=0, max_value=1000, value=default_values["platelet"], step=1,
            key=f"plt_{timepoint_idx}",
            help=feature_descriptions["platelet"]
        )
    
    with col2:
        st.markdown("##### Chemistry")
        values["bun"] = st.number_input(
            f"BUN (mg/dL) - {timepoint_name}", 
            min_value=0.0, max_value=100.0, value=default_values["bun"], step=0.1,
            key=f"bun_{timepoint_idx}",
            help=feature_descriptions["bun"]
        )
        values["pt"] = st.number_input(
            f"PT (seconds) - {timepoint_name}", 
            min_value=0.0, max_value=100.0, value=default_values["pt"], step=0.1,
            key=f"pt_{timepoint_idx}",
            help=feature_descriptions["pt"]
        )
        values["glucose"] = st.number_input(
            f"Glucose (mg/dL) - {timepoint_name}", 
            min_value=0, max_value=500, value=default_values["glucose"], step=1,
            key=f"glu_{timepoint_idx}",
            help=feature_descriptions["glucose"]
        )
        
        st.markdown("##### Electrolytes")
        values["sodium"] = st.number_input(
            f"Sodium (mmol/L) - {timepoint_name}", 
            min_value=100, max_value=160, value=default_values["sodium"], step=1,
            key=f"na_{timepoint_idx}",
            help=feature_descriptions["sodium"]
        )
        values["potassium"] = st.number_input(
            f"Potassium (mmol/L) - {timepoint_name}", 
            min_value=2.0, max_value=8.0, value=default_values["potassium"], step=0.1,
            key=f"k_{timepoint_idx}",
            help=feature_descriptions["potassium"]
        )
        values["chloride"] = st.number_input(
            f"Chloride (mmol/L) - {timepoint_name}", 
            min_value=80, max_value=120, value=default_values["chloride"], step=1,
            key=f"cl_{timepoint_idx}",
            help=feature_descriptions["chloride"]
        )
        values["bicarbonate"] = st.number_input(
            f"Bicarbonate (mmol/L) - {timepoint_name}", 
            min_value=10, max_value=40, value=default_values["bicarbonate"], step=1,
            key=f"hco3_{timepoint_idx}",
            help=feature_descriptions["bicarbonate"]
        )
    
    return values

# Helper function to display prediction result
def display_prediction(probability):
    """Display prediction result with styling"""
    col1, col2, col3 = st.columns(3)
    with col2:
        if probability > 0.5:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #dc3545;">Gram-positive</h3>
                <div class="gram-positive">{probability:.1%}</div>
                <p style="color: #6c757d;">Probability of Gram-positive</p>
                <p style="color: #28a745;">Gram-negative: {(1-probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #28a745;">Gram-negative</h3>
                <div class="gram-negative">{(1-probability):.1%}</div>
                <p style="color: #6c757d;">Probability of Gram-negative</p>
                <p style="color: #dc3545;">Gram-positive: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Main content based on selected page
if page == "🎯 Single Patient (3 Time Points)":
    st.header("🎯 Single Patient Prediction with 3 Time Points")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Instructions</strong><br>
        Enter patient clinical parameters for all three time periods to predict Gram classification.
        The model captures temporal dynamics across 0-8h (Period 3), 8-16h (Period 2), and 16-24h (Period 1).
    </div>
    """, unsafe_allow_html=True)
    
    # 创建三个时间点的标签页
    timepoint_names = ["Period 3 (0-8h)", "Period 2 (8-16h)", "Period 1 (16-24h)"]
    tabs = st.tabs([f"🕒 {name}" for name in timepoint_names])
    
    # 存储三个时间点的数据
    timepoint_data = {}
    
    # 为每个时间点创建输入表单
    for idx, (tab, name) in enumerate(zip(tabs, timepoint_names)):
        with tab:
            st.markdown(f"### {name}")
            # 使用不同的默认值来演示时间变化
            if idx == 0:  # Period 3 (0-8h) - 早期
                defaults = {
                    "heart_rate": 85, "sbp": 118, "resp_rate": 18, "spo2": 96,
                    "wbc": 9.5, "hemoglobin": 13.2, "platelet": 240,
                    "bun": 16, "pt": 12.5, "glucose": 110,
                    "sodium": 139, "potassium": 4.1, "chloride": 103, "bicarbonate": 23
                }
            elif idx == 1:  # Period 2 (8-16h) - 中期
                defaults = {
                    "heart_rate": 92, "sbp": 112, "resp_rate": 21, "spo2": 94,
                    "wbc": 12.0, "hemoglobin": 12.8, "platelet": 210,
                    "bun": 20, "pt": 13.5, "glucose": 125,
                    "sodium": 138, "potassium": 4.3, "chloride": 101, "bicarbonate": 22
                }
            else:  # Period 1 (16-24h) - 晚期
                defaults = {
                    "heart_rate": 105, "sbp": 105, "resp_rate": 24, "spo2": 92,
                    "wbc": 15.0, "hemoglobin": 12.0, "platelet": 180,
                    "bun": 28, "pt": 15.0, "glucose": 145,
                    "sodium": 136, "potassium": 4.6, "chloride": 99, "bicarbonate": 20
                }
            
            timepoint_data[idx] = create_timepoint_inputs(name, idx, defaults)
    
    # 显示数据预览
    with st.expander("📊 View Entered Data Summary"):
        for idx, name in enumerate(timepoint_names):
            st.markdown(f"**{name}**")
            df_preview = pd.DataFrame([timepoint_data[idx]]).T
            df_preview.columns = ['Value']
            st.dataframe(df_preview, use_container_width=True)
    
    # Prediction button
    if st.button("🔍 Predict Gram Classification", use_container_width=True):
        if st.session_state.model_loaded:
            try:
                # 构建3D数组: (1, 3, 14)
                X_3d_list = []
                for idx in range(3):
                    df = pd.DataFrame([timepoint_data[idx]])[st.session_state.feature_cols]
                    X_3d_list.append(df.values[0])
                
                X_3d = np.array(X_3d_list).reshape(1, 3, -1)
                
                # 使用 predict_temporal 方法
                probability = st.session_state.predictor.predict_temporal(X_3d)[0]
                
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                # Display result
                display_prediction(probability)
                
                # 显示时间序列趋势
                st.subheader("📈 Temporal Trends")
                
                # 创建趋势图数据
                trend_data = []
                for feature in ['wbc', 'heart_rate', 'resp_rate']:
                    values = [timepoint_data[i][feature] for i in range(3)]
                    trend_data.append({
                        'Feature': feature,
                        'Period 3 (0-8h)': values[0],
                        'Period 2 (8-16h)': values[1],
                        'Period 1 (16-24h)': values[2]
                    })
                
                trend_df = pd.DataFrame(trend_data)
                st.dataframe(trend_df, use_container_width=True)
                
                # Feature importance
                st.subheader("📈 Feature Contributions")
                importance = st.session_state.predictor.get_feature_importance()
                if importance:
                    imp_df = pd.DataFrame({
                        'Feature': list(importance.keys()),
                        'Importance': list(importance.values())
                    }).head(10)
                    
                    fig = px.bar(
                        imp_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 10 Feature Importance',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("Model not loaded. Please refresh the page.")

elif page == "📊 Batch Prediction":
    st.header("📊 Batch Prediction")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Instructions</strong><br>
        Upload a CSV file containing patient records for all three time points.
        The file must contain columns with suffixes: _t1, _t2, _t3 for each feature.
    </div>
    """, unsafe_allow_html=True)
    
    # Template download
    template_cols = []
    for suffix in ['_t1', '_t2', '_t3']:
        for feat in st.session_state.feature_cols:
            template_cols.append(f"{feat}{suffix}")
    
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_template,
        file_name="template_3timepoints.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="File must contain all features with _t1, _t2, _t3 suffixes"
    )
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)
        
        # Check features
        expected_cols = []
        for suffix in ['_t1', '_t2', '_t3']:
            for feat in st.session_state.feature_cols:
                expected_cols.append(f"{feat}{suffix}")
        
        missing_cols = set(expected_cols) - set(data.columns)
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            if st.button("Start Batch Prediction", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    # 构建3D数组
                    n_samples = len(data)
                    X_3d = np.zeros((n_samples, 3, len(st.session_state.feature_cols)))
                    
                    for i, feat in enumerate(st.session_state.feature_cols):
                        for t_idx, suffix in enumerate(['_t1', '_t2', '_t3']):
                            X_3d[:, t_idx, i] = data[f"{feat}{suffix}"].values
                    
                    # Predict
                    probas = st.session_state.predictor.predict_temporal(X_3d)
                    
                    # Create results
                    results = data.copy()
                    results['Gram_Positive_Probability'] = probas
                    results['Gram_Negative_Probability'] = 1 - probas
                    results['Prediction'] = ['Gram-positive' if p > 0.5 else 'Gram-negative' for p in probas]
                    
                    # Statistics
                    st.success("✅ Prediction completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Samples", len(results))
                    with col2:
                        gram_pos = (results['Gram_Positive_Probability'] > 0.5).sum()
                        st.metric("Gram-positive", gram_pos, f"{gram_pos/len(results):.1%}")
                    with col3:
                        gram_neg = (results['Gram_Positive_Probability'] <= 0.5).sum()
                        st.metric("Gram-negative", gram_neg, f"{gram_neg/len(results):.1%}")
                    with col4:
                        st.metric("Avg Probability", f"{probas.mean():.2%}")
                    
                    # Show results
                    st.subheader("Prediction Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Distribution plot
                    fig = px.histogram(
                        results, 
                        x='Gram_Positive_Probability',
                        nbins=20,
                        title='Distribution of Gram-positive Probabilities',
                        labels={'Gram_Positive_Probability': 'Probability of Gram-positive'},
                        color_discrete_sequence=['#2a5298']
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

elif page == "📋 Sample Cases":
    st.header("📋 Sample Cases with Time Series")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Sample Cases</strong><br>
        Explore pre-loaded sample cases with complete 3-timepoint data.
    </div>
    """, unsafe_allow_html=True)
    
    # Sample cases data with 3 time points
    sample_cases = {
        "Gram-positive (Worsening)": {
            "t1": {"heart_rate": 88, "sbp": 115, "resp_rate": 19, "spo2": 95,
                   "wbc": 10.2, "hemoglobin": 13.0, "platelet": 230,
                   "bun": 18, "pt": 13.0, "glucose": 115,
                   "sodium": 138, "potassium": 4.2, "chloride": 102, "bicarbonate": 23},
            "t2": {"heart_rate": 98, "sbp": 108, "resp_rate": 22, "spo2": 93,
                   "wbc": 13.5, "hemoglobin": 12.5, "platelet": 195,
                   "bun": 24, "pt": 14.0, "glucose": 132,
                   "sodium": 137, "potassium": 4.4, "chloride": 100, "bicarbonate": 21},
            "t3": {"heart_rate": 112, "sbp": 98, "resp_rate": 26, "spo2": 90,
                   "wbc": 17.8, "hemoglobin": 11.8, "platelet": 160,
                   "bun": 32, "pt": 15.5, "glucose": 155,
                   "sodium": 135, "potassium": 4.7, "chloride": 98, "bicarbonate": 19}
        },
        "Gram-negative (Improving)": {
            "t1": {"heart_rate": 105, "sbp": 95, "resp_rate": 25, "spo2": 91,
                   "wbc": 16.5, "hemoglobin": 11.5, "platelet": 165,
                   "bun": 30, "pt": 15.0, "glucose": 150,
                   "sodium": 135, "potassium": 4.6, "chloride": 99, "bicarbonate": 20},
            "t2": {"heart_rate": 95, "sbp": 105, "resp_rate": 21, "spo2": 94,
                   "wbc": 12.8, "hemoglobin": 12.2, "platelet": 210,
                   "bun": 24, "pt": 14.0, "glucose": 130,
                   "sodium": 137, "potassium": 4.3, "chloride": 101, "bicarbonate": 22},
            "t3": {"heart_rate": 82, "sbp": 118, "resp_rate": 18, "spo2": 97,
                   "wbc": 9.2, "hemoglobin": 13.0, "platelet": 265,
                   "bun": 18, "pt": 12.5, "glucose": 108,
                   "sodium": 139, "potassium": 4.0, "chloride": 103, "bicarbonate": 24}
        }
    }
    
    # Case selection
    selected_case = st.selectbox("Select Sample Case", list(sample_cases.keys()))
    
    if selected_case:
        case_data = sample_cases[selected_case]
        
        # Display time series data
        timepoint_names = ["Period 3 (0-8h)", "Period 2 (8-16h)", "Period 1 (16-24h)"]
        tabs = st.tabs([f"📊 {name}" for name in timepoint_names])
        
        for idx, (tab, name) in enumerate(zip(tabs, timepoint_names)):
            with tab:
                st.json({k: v for k, v in case_data[f"t{idx+1}"].items()})
        
        # Predict button
        if st.button("Run Prediction with Time Series", use_container_width=True):
            # Build 3D array
            X_3d_list = []
            for t_idx in range(3):
                df = pd.DataFrame([case_data[f"t{t_idx+1}"]])[st.session_state.feature_cols]
                X_3d_list.append(df.values[0])
            
            X_3d = np.array(X_3d_list).reshape(1, 3, -1)
            probability = st.session_state.predictor.predict_temporal(X_3d)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            display_prediction(probability)

else:  # About page
    st.header("ℹ️ About the System")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.icons8.com/color/240/000000/bacteria.png", width=200)
    
    with col2:
        st.markdown("""
        ### 🦠 Gram Classification System
        
        **Version**: 2.0.0 (with Time Series Support)
        
        **Clinical Application**:
        - Predicts Gram-positive vs Gram-negative classification in sepsis patients with bloodstream infection
        - Uses **3 time points** (0-8h, 8-16h, 16-24h) to capture temporal dynamics
        - Assists in early antibiotic therapy decision-making
        - Provides interpretable predictions using SHAP values
        
        **Model Features**:
        - **14 Clinical Parameters**: Vital signs, blood count, chemistry, electrolytes
        - **Time Series Processing**: Extracts mean, std, max, min, median, and trends
        - **Algorithm**: LightGBM with optimized hyperparameters
        - **Performance**: Validated on internal and external datasets
        
        **Key Benefits**:
        - Captures disease progression through temporal trends
        - Rapid Gram classification within 24 hours
        - No additional cost beyond routine labs
        - Transparent AI with explainable predictions
        
        **References**:
        1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
        2. SHAP: https://github.com/slundberg/shap
        """)
    
    st.markdown("---")
    
    # Features list
    st.subheader("📊 Clinical Features (3 Time Points)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Vital Signs**")
        for feat in ["heart_rate", "sbp", "resp_rate", "spo2"]:
            st.markdown(f"- {feat.replace('_', ' ').title()}")
    
    with col2:
        st.markdown("**Laboratory**")
        for feat in ["wbc", "hemoglobin", "platelet", "bun", "pt", "glucose"]:
            st.markdown(f"- {feat.replace('_', ' ').title()}")
    
    with col3:
        st.markdown("**Electrolytes**")
        for feat in ["sodium", "potassium", "chloride", "bicarbonate"]:
            st.markdown(f"- {feat.replace('_', ' ').title()}")
    
    st.markdown("---")
    
    # Citation
    st.markdown("""
    ### 📝 Citation
    
    If you use this tool in your research, please cite:
    @software{gram_classification_2024,
title = {Gram Classification System for Sepsis with Bloodstream Infection},
author = {Li Zeqi},
year = {2024},
url = {https://sepsis-gram-classification.streamlit.app}
}

### 📧 Contact

For questions or collaborations, please contact: lizeqi0726@163.com
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
<p>🦠 Gram Classification System | For Research Use Only | Version 2.0.0 (Time Series Support)</p>
<p style="font-size: 0.8rem;">© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)