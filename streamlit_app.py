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

# Custom CSS for dark blue theme with white text
st.markdown("""
<style>
    /* 全局深蓝色背景 */
    .stApp {
        background-color: #0a1929;
    }
    
    /* 主标题 - 深蓝色渐变 */
    .main-header {
        background: linear-gradient(135deg, #0a1929 0%, #1a2b3c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2a3b4c;
    }
    
    .main-header h1 {
        color: white !important;
    }
    
    .main-header p {
        color: #e0e0e0 !important;
    }
    
    /* 所有文本颜色 */
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stText {
        color: white !important;
    }
    
    /* 卡片样式 - 深色背景 */
    .prediction-card {
        background: #1e2a3a;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #2a3b4c;
    }
    
    .prediction-card h3 {
        color: white !important;
    }
    
    .gram-positive {
        color: #ff6b6b;
        font-size: 3rem;
        font-weight: bold;
    }
    
    .gram-negative {
        color: #51cf66;
        font-size: 3rem;
        font-weight: bold;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1e2a3a;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #4dabf7;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        margin: 0;
        color: #e0e0e0 !important;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .metric-card .value {
        margin: 0.5rem 0 0 0;
        color: #4dabf7;
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1e2a3a;
        color: white;
        border: 1px solid #4dabf7;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #2a3b4c;
        border-color: #74c0fc;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(77, 171, 247, 0.3);
    }
    
    /* Info box */
    .info-box {
        background-color: #1e2a3a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4dabf7;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    .info-box strong {
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1e2a3a;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #2a3b4c;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #e0e0e0 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2a3b4c !important;
        color: white !important;
        border: 1px solid #4dabf7;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4dabf7;
        border-radius: 10px;
        padding: 1rem;
        background-color: #1e2a3a;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #4dabf7 !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        gap: 0.5rem;
    }
    
    .stRadio [role="radiogroup"] > div {
        background-color: #1e2a3a;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #2a3b4c;
    }
    
    .stRadio [data-testid="stMarkdownContainer"] {
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e2a3a;
        border-radius: 8px;
        border-left: 3px solid #4dabf7;
        color: white !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1e2a3a;
        border: 1px solid #2a3b4c;
        border-radius: 0 0 8px 8px;
    }
    
    /* Success/Error messages */
    .stAlert {
        background-color: #1e2a3a !important;
        border-left: 4px solid #4dabf7;
        color: #e0e0e0 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #1e2a3a;
        color: white;
        border: 1px solid #4dabf7;
    }
    
    .stDownloadButton > button:hover {
        background: #2a3b4c;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #0a1929 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #0a1929;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #1e2a3a;
        color: white;
        border: 1px solid #2a3b4c;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #4dabf7;
    }
    
    /* Labels */
    label {
        color: #e0e0e0 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: #1e2a3a;
        color: white;
    }
    
    .stDataFrame th {
        background-color: #2a3b4c;
        color: white;
    }
    
    .stDataFrame td {
        background-color: #1e2a3a;
        color: #e0e0e0;
    }
    
    /* Metric labels */
    .metric-label {
        color: #4dabf7;
        font-weight: 600;
    }
    
    /* Plot backgrounds */
    .js-plotly-plot .plotly {
        background-color: #1e2a3a !important;
    }
    
    /* Footer */
    .footer {
        color: #6c757d;
        text-align: center;
        padding: 1rem;
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

# Header - 深蓝色渐变
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
            """)
    
    # Navigation
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Function",
        ["🎯 Single Prediction", "📊 Batch Prediction", "📋 Sample Cases", "ℹ️ About"],
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

# Helper function to create feature input
def create_feature_inputs():
    """Create input fields for all features"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Vital Signs")
        heart_rate = st.number_input(
            "Heart Rate (bpm)", 
            min_value=0, max_value=300, value=80, step=1,
            help=feature_descriptions["heart_rate"]
        )
        sbp = st.number_input(
            "SBP (mmHg)", 
            min_value=0, max_value=300, value=120, step=1,
            help=feature_descriptions["sbp"]
        )
        resp_rate = st.number_input(
            "Respiratory Rate (breaths/min)", 
            min_value=0, max_value=100, value=16, step=1,
            help=feature_descriptions["resp_rate"]
        )
        spo2 = st.number_input(
            "SpO₂ (%)", 
            min_value=0, max_value=100, value=98, step=1,
            help=feature_descriptions["spo2"]
        )
        
        st.markdown("#### Laboratory - Blood Count")
        wbc = st.number_input(
            "WBC (10⁹/L)", 
            min_value=0.0, max_value=100.0, value=8.5, step=0.1,
            help=feature_descriptions["wbc"]
        )
        hemoglobin = st.number_input(
            "Hemoglobin (g/dL)", 
            min_value=0.0, max_value=20.0, value=13.5, step=0.1,
            help=feature_descriptions["hemoglobin"]
        )
        platelet = st.number_input(
            "Platelet (10⁹/L)", 
            min_value=0, max_value=1000, value=250, step=1,
            help=feature_descriptions["platelet"]
        )
    
    with col2:
        st.markdown("#### Laboratory - Chemistry")
        bun = st.number_input(
            "BUN (mg/dL)", 
            min_value=0.0, max_value=100.0, value=15.0, step=0.1,
            help=feature_descriptions["bun"]
        )
        pt = st.number_input(
            "PT (seconds)", 
            min_value=0.0, max_value=100.0, value=12.0, step=0.1,
            help=feature_descriptions["pt"]
        )
        glucose = st.number_input(
            "Glucose (mg/dL)", 
            min_value=0, max_value=500, value=100, step=1,
            help=feature_descriptions["glucose"]
        )
        
        st.markdown("#### Electrolytes")
        sodium = st.number_input(
            "Sodium (mmol/L)", 
            min_value=100, max_value=160, value=140, step=1,
            help=feature_descriptions["sodium"]
        )
        potassium = st.number_input(
            "Potassium (mmol/L)", 
            min_value=2.0, max_value=8.0, value=4.0, step=0.1,
            help=feature_descriptions["potassium"]
        )
        chloride = st.number_input(
            "Chloride (mmol/L)", 
            min_value=80, max_value=120, value=102, step=1,
            help=feature_descriptions["chloride"]
        )
        bicarbonate = st.number_input(
            "Bicarbonate (mmol/L)", 
            min_value=10, max_value=40, value=24, step=1,
            help=feature_descriptions["bicarbonate"]
        )
    
    return {
        "heart_rate": heart_rate,
        "sbp": sbp,
        "resp_rate": resp_rate,
        "spo2": spo2,
        "wbc": wbc,
        "hemoglobin": hemoglobin,
        "platelet": platelet,
        "bun": bun,
        "pt": pt,
        "glucose": glucose,
        "sodium": sodium,
        "potassium": potassium,
        "chloride": chloride,
        "bicarbonate": bicarbonate
    }

# Helper function to display prediction result
def display_prediction(probability):
    """Display prediction result with styling"""
    col1, col2, col3 = st.columns(3)
    with col2:
        if probability > 0.5:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #ff6b6b;">Gram-positive</h3>
                <div class="gram-positive">{probability:.1%}</div>
                <p style="color: #adb5bd;">Probability of Gram-positive</p>
                <p style="color: #51cf66;">Gram-negative: {(1-probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card">
                <h3 style="color: #51cf66;">Gram-negative</h3>
                <div class="gram-negative">{(1-probability):.1%}</div>
                <p style="color: #adb5bd;">Probability of Gram-negative</p>
                <p style="color: #ff6b6b;">Gram-positive: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

# Main content based on selected page
if page == "🎯 Single Prediction":
    st.header("🎯 Single Patient Prediction")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Instructions</strong><br>
        Enter patient clinical parameters below to predict Gram classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Create input fields
    input_values = create_feature_inputs()
    
    # Prediction button
    if st.button("🔍 Predict Gram Classification", use_container_width=True):
        if st.session_state.model_loaded:
            # Prepare input data
            input_df = pd.DataFrame([input_values])[st.session_state.feature_cols]
            
            try:
                # Get prediction
                probability = st.session_state.predictor.predict_single(input_df)[0]
                
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                # Display result
                display_prediction(probability)
                
                # Feature importance for this prediction
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
                    fig.update_layout(
                        height=400,
                        paper_bgcolor='#1e2a3a',
                        plot_bgcolor='#1e2a3a',
                        font_color='white'
                    )
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
        Upload a CSV file containing multiple patient records for batch prediction.
        The file must contain all required features.
    </div>
    """, unsafe_allow_html=True)
    
    # Template download
    template_df = pd.DataFrame(columns=st.session_state.feature_cols)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Template CSV",
        data=csv_template,
        file_name="template.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="File must contain all feature columns"
    )
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(), use_container_width=True)
        
        # Check features
        missing_cols = set(st.session_state.feature_cols) - set(data.columns)
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            if st.button("Start Batch Prediction", use_container_width=True):
                with st.spinner("Processing predictions..."):
                    # Prepare data
                    X = data[st.session_state.feature_cols]
                    
                    # Predict
                    probas = st.session_state.predictor.predict_single(X)
                    
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
                        color_discrete_sequence=['#4dabf7']
                    )
                    fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                    fig.update_layout(
                        paper_bgcolor='#1e2a3a',
                        plot_bgcolor='#1e2a3a',
                        font_color='white'
                    )
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
    st.header("📋 Sample Cases")
    
    st.markdown("""
    <div class="info-box">
        <strong>📌 Sample Cases</strong><br>
        Explore pre-loaded sample cases to see how the model performs.
    </div>
    """, unsafe_allow_html=True)
    
    # Sample cases data
    sample_cases = {
        "Typical Gram-positive": {
            "heart_rate": 110, "sbp": 90, "resp_rate": 24, "spo2": 92,
            "wbc": 15.5, "hemoglobin": 10.2, "platelet": 150,
            "bun": 35, "pt": 18, "glucose": 180,
            "sodium": 135, "potassium": 4.5, "chloride": 100, "bicarbonate": 18
        },
        "Typical Gram-negative": {
            "heart_rate": 95, "sbp": 110, "resp_rate": 20, "spo2": 95,
            "wbc": 8.5, "hemoglobin": 12.5, "platelet": 280,
            "bun": 22, "pt": 14, "glucose": 140,
            "sodium": 138, "potassium": 4.0, "chloride": 105, "bicarbonate": 22
        },
        "Borderline Case": {
            "heart_rate": 100, "sbp": 100, "resp_rate": 22, "spo2": 94,
            "wbc": 11.0, "hemoglobin": 11.5, "platelet": 200,
            "bun": 28, "pt": 16, "glucose": 160,
            "sodium": 136, "potassium": 4.2, "chloride": 102, "bicarbonate": 20
        }
    }
    
    # Case selection
    selected_case = st.selectbox("Select Sample Case", list(sample_cases.keys()))
    
    if selected_case:
        case_data = sample_cases[selected_case]
        
        # Display case data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Vital Signs")
            st.json({k: case_data[k] for k in ["heart_rate", "sbp", "resp_rate", "spo2"]})
            
            st.markdown("#### Blood Count")
            st.json({k: case_data[k] for k in ["wbc", "hemoglobin", "platelet"]})
        
        with col2:
            st.markdown("#### Chemistry")
            st.json({k: case_data[k] for k in ["bun", "pt", "glucose"]})
            
            st.markdown("#### Electrolytes")
            st.json({k: case_data[k] for k in ["sodium", "potassium", "chloride", "bicarbonate"]})
        
        # Predict button
        if st.button("Run Prediction", use_container_width=True):
            input_df = pd.DataFrame([case_data])[st.session_state.feature_cols]
            probability = st.session_state.predictor.predict_single(input_df)[0]
            
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
        
        **Version**: 1.0.0
        
        **Clinical Application**:
        - Predicts Gram-positive vs Gram-negative classification in sepsis patients with bloodstream infection
        - Assists in early antibiotic therapy decision-making
        - Provides interpretable predictions using SHAP values
        
        **Model Features**:
        - **14 Clinical Parameters**: Vital signs, blood count, chemistry, electrolytes
        - **Algorithm**: LightGBM with optimized hyperparameters
        - **Performance**: Validated on internal and external datasets
        
        **Key Benefits**:
        - Rapid Gram classification within 24 hours
        - No additional cost beyond routine labs
        - Transparent AI with explainable predictions
        
        **References**:
        1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
        2. SHAP: https://github.com/slundberg/shap
        """)
    
    st.markdown("---")
    
    # Features list
    st.subheader("📊 Clinical Features")
    
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
<p>🦠 Gram Classification System | For Research Use Only | Version 1.0.0</p>
<p style="font-size: 0.8rem;">© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)