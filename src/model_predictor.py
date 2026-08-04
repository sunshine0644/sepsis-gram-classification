"""
model_predictor.py — 5-feature LightGBM predictor
"""
import numpy as np
import json
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


class SepsisPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "best_models_fixed_leakage"
        self.model_dir = Path(model_dir)

        # Load model
        self.model = joblib.load(str(self.model_dir / "LightGBM_model.pkl"))

        # Load preprocessor
        self.preprocessor = joblib.load(str(self.model_dir / "LightGBM_preprocessor.pkl"))

        # Load threshold
        self.threshold = float(joblib.load(str(self.model_dir / "LightGBM_threshold.pkl")))

        # Load scaler
        self.scaler = joblib.load(str(self.model_dir / "scaler.pkl"))

        self.feature_cols = self.preprocessor['feature_cols']
        self.n_features = self.preprocessor['n_features']

        # Load metrics
        with open(self.model_dir / "performance_metrics.json", 'r') as f:
            self.performance_metrics = json.load(f)

    def _stats(self, X):
        return np.concatenate([
            np.mean(X, axis=0),
            np.std(X, axis=0) if X.shape[0] > 1 else np.zeros(X.shape[1]),
            np.max(X, axis=0),
            np.min(X, axis=0),
            np.median(X, axis=0)
        ])

    def predict(self, values_list):
        if len(values_list) < 3:
            pad = np.tile(values_list[0], (3 - len(values_list), 1))
            values_list = list(values_list) + [pad[i] for i in range(len(pad))]
        X = np.array(values_list)
        feats = self._stats(X).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        return float(self.model.predict_proba(feats_scaled)[0, 1])

    def predict_temporal(self, X_3d):
        n = X_3d.shape[0]
        probs = np.zeros(n)
        for i in range(n):
            probs[i] = self.predict([X_3d[i, t, :] for t in range(X_3d.shape[1])])
        return probs

    def get_feature_importance(self):
        try:
            imp = self.model.feature_importances_
            names = [f"{f}_{s}" for s in ['mean','std','max','min','median'] for f in self.feature_cols]
            idx = np.argsort(imp)[::-1][:10]
            return {names[i]: float(imp[i]) for i in idx}
        except:
            return {}
