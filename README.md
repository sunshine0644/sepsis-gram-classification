# sepsis-gram-classification

Machine learning–based early prediction of Gram‑stain classification in sepsis with bloodstream infection.

## Live Demo

https://www.sepsis-bsi-gram.cn

## Model

- **Algorithm**: LightGBM
- **Features**: 5 key parameters (PT, platelet, hemoglobin, bicarbonate, respiratory rate)
- **Time windows**: 3 × 8-hour windows before blood culture collection
- **Training**: MIMIC-IV (n = 1,403) | External validation: 101 patients

### Performance

| Metric | Internal Test | External Validation |
|--------|:------------:|:-------------------:|
| AUROC | 0.970 | 0.962 |
| AUPRC | 0.989 | 0.968 |

## Project Structure

```
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── models/                 # Trained model artifacts
│   ├── LightGBM_model.pkl
│   ├── LightGBM_preprocessor.pkl
│   ├── LightGBM_threshold.pkl
│   ├── scaler.pkl
│   ├── preprocessor.json
│   ├── threshold.json
│   ├── config.json
│   ├── model_info.json
│   └── performance_metrics.json
├── src/
│   └── model_predictor.py  # Model loading and prediction
└── training/               # Training and analysis code
    ├── data_preprocessing.R
    ├── train_model.py
    ├── shap_analysis.py
    └── plot_curves.py
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

The model was trained on MIMIC-IV (available at https://physionet.org/).

## Citation

Li Z, Guo M, Fu C, Sun X, Lin S, Yang X. Machine Learning for Early Prediction of
Gram-Stain Classification in Sepsis with Bloodstream Infection. 2026.

## License

MIT License. For research use only.
