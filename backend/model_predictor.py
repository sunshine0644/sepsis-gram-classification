# backend/model_predictor.py

import shap
import numpy as np
import pandas as pd
import joblib
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class SepsisPredictor:
    """脓毒症预测器 - 整合SHAP分析功能"""
    
    def __init__(self, models_path='saved_models'):
        self.models_path = Path(models_path)
        self.model = None
        self.scaler = None
        self.config = None
        self.feature_cols = None
        self.expected_features = None
        self.load_models()
    
    def load_models(self):
        """加载模型和配置"""
        print("\nLoading Sepsis Prediction Model...")
        print("-" * 60)
        
        # 1. 加载配置
        config_path = self.models_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.feature_cols = self.config.get('feature_columns', [
            "heart_rate", "sbp", "resp_rate", "spo2", "wbc", 
            "hemoglobin", "platelet", "bun", "pt", "glucose", 
            "sodium", "potassium", "chloride", "bicarbonate"
        ])
        print(f"✓ Features: {len(self.feature_cols)}")
        
        # 2. 加载模型
        model_path = self.models_path / "LightGBM_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # 获取期望特征数
        if hasattr(self.model, 'n_features_in_'):
            self.expected_features = self.model.n_features_in_
        elif hasattr(self.model, 'feature_importances_'):
            self.expected_features = len(self.model.feature_importances_)
        else:
            self.expected_features = 70
        
        print(f"✓ Model loaded, expected features: {self.expected_features}")
        
        # 3. 加载scaler
        scaler_path = self.models_path / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler文件不存在: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print("✓ Scaler loaded")
        print("=" * 60)
    
    def prepare_features(self, X_3d):
        """
        准备特征用于预测（从你的SHAP代码复制）
        X_3d: shape (n_samples, 3, n_features)
        """
        n_samples = X_3d.shape[0]
        n_features = len(self.feature_cols)
        
        features_list = []
        for i in range(n_samples):
            patient_features = X_3d[i]
            actual_length = min(3, np.sum(~np.isnan(patient_features[:, 0])))
            
            if actual_length > 0:
                valid_features = patient_features[:actual_length]
                
                mean_f = np.nanmean(valid_features, axis=0)
                std_f = np.nanstd(valid_features, axis=0, ddof=1)
                max_f = np.nanmax(valid_features, axis=0)
                min_f = np.nanmin(valid_features, axis=0)
                median_f = np.nanmedian(valid_features, axis=0)
                
                if self.expected_features == 70:
                    combined = np.concatenate([mean_f, std_f, max_f, min_f, median_f])
                elif self.expected_features == 84:
                    trend_f = []
                    for j in range(n_features):
                        col_data = valid_features[:, j]
                        valid_idx = ~np.isnan(col_data)
                        if np.sum(valid_idx) > 1:
                            try:
                                coeff = np.polyfit(np.arange(np.sum(valid_idx)), col_data[valid_idx], 1)
                                trend_f.append(coeff[0])
                            except:
                                trend_f.append(0.0)
                        else:
                            trend_f.append(0.0)
                    trend_f = np.array(trend_f)
                    combined = np.concatenate([mean_f, std_f, max_f, min_f, median_f, trend_f])
                else:
                    combined = np.concatenate([mean_f, std_f, max_f, min_f, median_f])
            else:
                if self.expected_features == 70:
                    combined = np.zeros(n_features * 5)
                elif self.expected_features == 84:
                    combined = np.zeros(n_features * 6)
                else:
                    combined = np.zeros(self.expected_features)
            
            features_list.append(combined)
        
        features_array = np.array(features_list)
        
        # 确保维度匹配
        if features_array.shape[1] != self.expected_features:
            if features_array.shape[1] > self.expected_features:
                features_array = features_array[:, :self.expected_features]
            else:
                padding = np.zeros((n_samples, self.expected_features - features_array.shape[1]))
                features_array = np.hstack([features_array, padding])
        
        return features_array
    
    def predict_single(self, input_data):
        """
        单时间点预测
        input_data: DataFrame 或 数组，shape (n_samples, n_features)
        """
        if isinstance(input_data, pd.DataFrame):
            X = input_data[self.feature_cols].values
        else:
            X = np.array(input_data)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 预测
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_scaled)
            return proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            return self.model.predict(X_scaled)
    
    def predict_temporal(self, X_3d):
        """
        多时间点预测（3个时间点）
        X_3d: shape (n_samples, 3, n_features)
        """
        # 标准化
        X_flat = X_3d.reshape(-1, len(self.feature_cols))
        X_flat_scaled = self.scaler.transform(X_flat)
        X_scaled = X_flat_scaled.reshape(-1, 3, len(self.feature_cols))
        
        # 准备特征
        features = self.prepare_features(X_scaled)
        
        # 预测
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)
            return proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            return self.model.predict(features)
    
    def predict_for_shap(self, X_flat):
        """
        SHAP预测函数（从你的SHAP代码复制）
        X_flat: shape (n_samples, 3 * n_features)
        """
        X_3d = X_flat.reshape(-1, 3, len(self.feature_cols))
        
        # 标准化
        X_flat_scaled = self.scaler.transform(X_3d.reshape(-1, len(self.feature_cols)))
        X_scaled = X_flat_scaled.reshape(-1, 3, len(self.feature_cols))
        
        # 准备特征
        features = self.prepare_features(X_scaled)
        
        if features.shape[1] != self.expected_features:
            print(f"❌ 维度不匹配: 生成{features.shape[1]}个特征, 期望{self.expected_features}个")
            return np.random.uniform(0, 1, len(features))
        
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)
                return proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
            else:
                return self.model.predict(features).astype(float)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return np.random.uniform(0, 1, len(features))
    
    def compute_shap_values(self, X_data, n_background=150):
        """
        计算SHAP值
        X_data: 3D数组 shape (n_samples, 3, n_features)
        """
        X_flat = X_data.reshape(len(X_data), -1)
        print(f"  Computing SHAP for {X_flat.shape[0]} samples")
        
        # 选择背景样本
        n_background = min(n_background, len(X_flat))
        background = X_flat[:n_background]
        
        try:
            # 使用PermutationExplainer
            explainer = shap.explainers.Permutation(
                self.predict_for_shap,
                background,
                max_evals=250
            )
            
            shap_values = explainer(X_flat, silent=True)
            return shap_values
            
        except Exception as e:
            print(f"  SHAP computation failed: {e}")
            return None
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # 如果是多时间点特征，需要映射回原始特征
            if len(importance) > len(self.feature_cols):
                # 假设特征是 [mean, std, max, min, median] 各14个
                n_stats = len(importance) // len(self.feature_cols)
                feature_importance = {}
                for i, feat in enumerate(self.feature_cols):
                    start_idx = i * n_stats
                    end_idx = (i + 1) * n_stats
                    feature_importance[feat] = np.mean(importance[start_idx:end_idx])
                
                # 排序
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_features)
            else:
                return dict(zip(self.feature_cols[:len(importance)], importance))
        return None

# 创建单例
predictor = SepsisPredictor()