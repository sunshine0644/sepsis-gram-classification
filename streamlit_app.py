# streamlit_app.py - 极简重构版

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add backend path
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
    layout="wide"
)

# Custom CSS - 标题为白色
st.markdown("""
<style>
    /* Main header - 深蓝色背景，白色文字 */
    .main-header {
        background: linear-gradient(135deg, #0a1929 0%, #1a2b3c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white !important;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.2rem;
    }
    
    /* 预测结果卡片 */
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
    
    /* 按钮样式 */
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
    
    /* 信息框 */
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    /* 标签页样式 */
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
    
    /* 文件上传器 */
    .uploadedFile {
        border: 2px dashed #2a5298;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* 趋势表格样式 */
    .trend-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Header - 深蓝色背景白色文字
st.markdown("""
<div class="main-header">
    <h1>🦠 Gram Classification System</h1>
    <p>Machine Learning-based Gram Classification for Sepsis with Bloodstream Infection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bacteria.png", width=80)
    st.markdown("## Control Panel")
    
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            try:
                st.session_state.predictor = SepsisPredictor()
                st.session_state.model_loaded = True
                st.success("✅ Model loaded!")
            except Exception as e:
                st.error(f"❌ Model loading failed: {e}")
    
    page = st.radio(
        "Navigation",
        ["Single Patient (3 Time Points)", "Batch Prediction", "Sample Cases", "About"]
    )

# Feature list
feature_cols = [
    "heart_rate", "sbp", "resp_rate", "spo2", "wbc", 
    "hemoglobin", "platelet", "bun", "pt", "glucose", 
    "sodium", "potassium", "chloride", "bicarbonate"
]

feature_labels = {
    "heart_rate": "Heart Rate (bpm)",
    "sbp": "SBP (mmHg)",
    "resp_rate": "Respiratory Rate",
    "spo2": "SpO₂ (%)",
    "wbc": "WBC (10⁹/L)",
    "hemoglobin": "Hemoglobin (g/dL)",
    "platelet": "Platelet (10⁹/L)",
    "bun": "BUN (mg/dL)",
    "pt": "PT (seconds)",
    "glucose": "Glucose (mg/dL)",
    "sodium": "Sodium (mmol/L)",
    "potassium": "Potassium (mmol/L)",
    "chloride": "Chloride (mmol/L)",
    "bicarbonate": "Bicarbonate (mmol/L)"
}

# Helper function to display prediction
def display_prediction(probability):
    col1, col2, col3 = st.columns(3)
    with col2:
        if probability > 0.5:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #dc3545;">Gram-positive</h3>
                <div class="gram-positive">{probability:.1%}</div>
                <p>Gram-negative: {(1-probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #28a745;">Gram-negative</h3>
                <div class="gram-negative">{(1-probability):.1%}</div>
                <p>Gram-positive: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Main content
if page == "Single Patient (3 Time Points)":
    st.header("Single Patient with 3 Time Points")
    
    st.markdown("""
    <div class="info-box">
        Enter patient data for all three time periods (0-8h, 8-16h, 16-24h).
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for three time points
    tab1, tab2, tab3 = st.tabs(["🕒 Period 3 (0-8h)", "🕒 Period 2 (8-16h)", "🕒 Period 1 (16-24h)"])
    
    # Store data for each time point
    data_t1 = {}
    data_t2 = {}
    data_t3 = {}
    
    # Period 3 (0-8h)
    with tab1:
        st.markdown("### Period 3 (0-8h) - Early")
        col1, col2 = st.columns(2)
        with col1:
            data_t1["heart_rate"] = st.number_input("Heart Rate", min_value=0, max_value=300, value=85, step=1, key="t1_hr")
            data_t1["sbp"] = st.number_input("SBP", min_value=0, max_value=300, value=118, step=1, key="t1_sbp")
            data_t1["resp_rate"] = st.number_input("Respiratory Rate", min_value=0, max_value=100, value=18, step=1, key="t1_rr")
            data_t1["spo2"] = st.number_input("SpO₂", min_value=0, max_value=100, value=96, step=1, key="t1_spo2")
            data_t1["wbc"] = st.number_input("WBC", min_value=0.0, max_value=100.0, value=9.5, step=0.1, key="t1_wbc")
            data_t1["hemoglobin"] = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=13.2, step=0.1, key="t1_hgb")
            data_t1["platelet"] = st.number_input("Platelet", min_value=0, max_value=1000, value=240, step=1, key="t1_plt")
        with col2:
            data_t1["bun"] = st.number_input("BUN", min_value=0.0, max_value=100.0, value=16.0, step=0.1, key="t1_bun")
            data_t1["pt"] = st.number_input("PT", min_value=0.0, max_value=100.0, value=12.5, step=0.1, key="t1_pt")
            data_t1["glucose"] = st.number_input("Glucose", min_value=0, max_value=500, value=110, step=1, key="t1_glu")
            data_t1["sodium"] = st.number_input("Sodium", min_value=100, max_value=160, value=139, step=1, key="t1_na")
            data_t1["potassium"] = st.number_input("Potassium", min_value=2.0, max_value=8.0, value=4.1, step=0.1, key="t1_k")
            data_t1["chloride"] = st.number_input("Chloride", min_value=80, max_value=120, value=103, step=1, key="t1_cl")
            data_t1["bicarbonate"] = st.number_input("Bicarbonate", min_value=10, max_value=40, value=23, step=1, key="t1_hco3")
    
    # Period 2 (8-16h)
    with tab2:
        st.markdown("### Period 2 (8-16h) - Middle")
        col1, col2 = st.columns(2)
        with col1:
            data_t2["heart_rate"] = st.number_input("Heart Rate", min_value=0, max_value=300, value=92, step=1, key="t2_hr")
            data_t2["sbp"] = st.number_input("SBP", min_value=0, max_value=300, value=112, step=1, key="t2_sbp")
            data_t2["resp_rate"] = st.number_input("Respiratory Rate", min_value=0, max_value=100, value=21, step=1, key="t2_rr")
            data_t2["spo2"] = st.number_input("SpO₂", min_value=0, max_value=100, value=94, step=1, key="t2_spo2")
            data_t2["wbc"] = st.number_input("WBC", min_value=0.0, max_value=100.0, value=12.0, step=0.1, key="t2_wbc")
            data_t2["hemoglobin"] = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=12.8, step=0.1, key="t2_hgb")
            data_t2["platelet"] = st.number_input("Platelet", min_value=0, max_value=1000, value=210, step=1, key="t2_plt")
        with col2:
            data_t2["bun"] = st.number_input("BUN", min_value=0.0, max_value=100.0, value=20.0, step=0.1, key="t2_bun")
            data_t2["pt"] = st.number_input("PT", min_value=0.0, max_value=100.0, value=13.5, step=0.1, key="t2_pt")
            data_t2["glucose"] = st.number_input("Glucose", min_value=0, max_value=500, value=125, step=1, key="t2_glu")
            data_t2["sodium"] = st.number_input("Sodium", min_value=100, max_value=160, value=138, step=1, key="t2_na")
            data_t2["potassium"] = st.number_input("Potassium", min_value=2.0, max_value=8.0, value=4.3, step=0.1, key="t2_k")
            data_t2["chloride"] = st.number_input("Chloride", min_value=80, max_value=120, value=101, step=1, key="t2_cl")
            data_t2["bicarbonate"] = st.number_input("Bicarbonate", min_value=10, max_value=40, value=22, step=1, key="t2_hco3")
    
    # Period 1 (16-24h)
    with tab3:
        st.markdown("### Period 1 (16-24h) - Late")
        col1, col2 = st.columns(2)
        with col1:
            data_t3["heart_rate"] = st.number_input("Heart Rate", min_value=0, max_value=300, value=105, step=1, key="t3_hr")
            data_t3["sbp"] = st.number_input("SBP", min_value=0, max_value=300, value=105, step=1, key="t3_sbp")
            data_t3["resp_rate"] = st.number_input("Respiratory Rate", min_value=0, max_value=100, value=24, step=1, key="t3_rr")
            data_t3["spo2"] = st.number_input("SpO₂", min_value=0, max_value=100, value=92, step=1, key="t3_spo2")
            data_t3["wbc"] = st.number_input("WBC", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="t3_wbc")
            data_t3["hemoglobin"] = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=12.0, step=0.1, key="t3_hgb")
            data_t3["platelet"] = st.number_input("Platelet", min_value=0, max_value=1000, value=180, step=1, key="t3_plt")
        with col2:
            data_t3["bun"] = st.number_input("BUN", min_value=0.0, max_value=100.0, value=28.0, step=0.1, key="t3_bun")
            data_t3["pt"] = st.number_input("PT", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="t3_pt")
            data_t3["glucose"] = st.number_input("Glucose", min_value=0, max_value=500, value=145, step=1, key="t3_glu")
            data_t3["sodium"] = st.number_input("Sodium", min_value=100, max_value=160, value=136, step=1, key="t3_na")
            data_t3["potassium"] = st.number_input("Potassium", min_value=2.0, max_value=8.0, value=4.6, step=0.1, key="t3_k")
            data_t3["chloride"] = st.number_input("Chloride", min_value=80, max_value=120, value=99, step=1, key="t3_cl")
            data_t3["bicarbonate"] = st.number_input("Bicarbonate", min_value=10, max_value=40, value=20, step=1, key="t3_hco3")
    
    # Predict button
    if st.button("Predict Gram Classification", use_container_width=True):
        if st.session_state.model_loaded:
            try:
                # Build 3D array
                X_3d_list = []
                for t_data in [data_t1, data_t2, data_t3]:
                    df = pd.DataFrame([t_data])[feature_cols]
                    X_3d_list.append(df.values[0])
                
                X_3d = np.array(X_3d_list).reshape(1, 3, -1)
                probability = st.session_state.predictor.predict_temporal(X_3d)[0]
                
                st.markdown("---")
                st.subheader("Prediction Result")
                display_prediction(probability)
                
                # 显示所有特征的完整趋势
                st.subheader("Temporal Trends - All Features")
                
                # 创建包含所有特征的趋势数据框
                trend_data = {'Period': ['0-8h', '8-16h', '16-24h']}
                for feature in feature_cols:
                    display_name = feature_labels[feature]
                    trend_data[display_name] = [
                        data_t1[feature], 
                        data_t2[feature], 
                        data_t3[feature]
                    ]
                
                trend_df = pd.DataFrame(trend_data)
                # 设置Period为索引，方便显示
                trend_display = trend_df.set_index('Period')
                
                # 显示完整表格
                st.dataframe(trend_display, use_container_width=True)
                
                # 显示关键指标的折线图
                st.subheader("Key Trends Chart")
                key_features = ['Heart Rate (bpm)', 'WBC (10⁹/L)', 'SBP (mmHg)', 'Respiratory Rate']
                # 确保这些特征在数据中
                available_keys = [k for k in key_features if k in trend_display.columns]
                if available_keys:
                    chart_df = trend_display[available_keys]
                    st.line_chart(chart_df)
                
                # 可选：显示所有特征的折线图（可能会有点挤）
                with st.expander("View All Features Chart"):
                    st.line_chart(trend_display)
                
                # Feature importance
                st.subheader("Feature Contributions")
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
                st.exception(e)  # 显示详细错误信息
        else:
            st.error("Model not loaded")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    
    st.markdown("""
    <div class="info-box">
        Upload a CSV file with columns: heart_rate_t1, heart_rate_t2, heart_rate_t3, etc.
    </div>
    """, unsafe_allow_html=True)
    
    # Template download
    template_cols = []
    for suffix in ['_t1', '_t2', '_t3']:
        for feat in feature_cols:
            template_cols.append(f"{feat}{suffix}")
    
    template_df = pd.DataFrame(columns=template_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download Template CSV",
        data=csv_template,
        file_name="template_3timepoints.csv",
        mime="text/csv"
    )
    
    # File upload
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        
        if st.button("Start Batch Prediction"):
            with st.spinner("Processing..."):
                # Build 3D array
                n_samples = len(data)
                X_3d = np.zeros((n_samples, 3, len(feature_cols)))
                
                for i, feat in enumerate(feature_cols):
                    for t_idx, suffix in enumerate(['_t1', '_t2', '_t3']):
                        X_3d[:, t_idx, i] = data[f"{feat}{suffix}"].values
                
                probas = st.session_state.predictor.predict_temporal(X_3d)
                
                results = data.copy()
                results['Gram_Positive_Probability'] = probas
                results['Prediction'] = ['Gram-positive' if p > 0.5 else 'Gram-negative' for p in probas]
                
                st.success("Prediction completed!")
                st.dataframe(results)
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

elif page == "Sample Cases":
    st.header("Sample Cases")
    
    st.markdown("""
    <div class="info-box">
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
                df = pd.DataFrame([case_data[f"t{t_idx+1}"]])[feature_cols]
                X_3d_list.append(df.values[0])
            
            X_3d = np.array(X_3d_list).reshape(1, 3, -1)
            probability = st.session_state.predictor.predict_temporal(X_3d)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            display_prediction(probability)

else:  # About
    st.header("About")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.icons8.com/color/240/000000/bacteria.png", width=150)
    with col2:
        st.markdown("""
        ### 🦠 Gram Classification System
        
        **Version**: 2.0.0
        
        **Clinical Application**:
        - Predicts Gram-positive vs Gram-negative classification in sepsis patients with bloodstream infection
        - Uses **3 time points** (0-8h, 8-16h, 16-24h) to capture temporal dynamics
        - Assists in early antibiotic therapy decision-making
        
        **Model Features**:
        - **14 Clinical Parameters**: Vital signs, blood count, chemistry, electrolytes
        - **Time Series Processing**: Extracts mean, std, max, min, median, and trends
        - **Algorithm**: LightGBM with SHAP interpretability
        - **Performance**: Validated on internal and external datasets
        
        **Key Benefits**:
        - Captures disease progression through temporal trends
        - Rapid Gram classification within 24 hours
        - No additional cost beyond routine labs
        - Transparent AI with explainable predictions
        
        **Reference**:
        Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Citation
@software{gram_classification_2026,
title = {Gram Classification System for Sepsis with Bloodstream Infection},
author = {Li Zeqi},
year = {2026},
url = {https://www.sepsis-bsi-gram.cn}
}

### Contact
For questions or collaborations: lizeqi0726@163.com
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
<p>🦠 Gram Classification System | For Research Use Only | Version 2.0.0</p>
</div>
""", unsafe_allow_html=True)
