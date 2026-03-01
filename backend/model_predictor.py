# backend/model_predictor.py

import shap
import numpy as np
import pandas as pd
import joblib
import json
import warnings
import matplotlib
matplotlib.use('Agg')  # 解决GUI问题
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path
import sys
warnings.filterwarnings('ignore')

class SepsisPredictor:
    """脓毒症预测器 - 整合SHAP分析功能"""
    
    def __init__(self, models_path=None):
        # 获取当前文件所在目录
        self.current_dir = Path(__file__).parent
        
        # 设置模型路径
        if models_path is None:
            # 默认使用当前目录下的 saved_models
            self.models_path = self.current_dir / "saved_models"
        else:
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
        print(f"Current directory: {self.current_dir}")
        print(f"Models path: {self.models_path}")
        
        # 1. 加载配置
        config_path = self.models_path / "config.json"
        if not config_path.exists():
            # 尝试上一级目录
            alt_path = self.current_dir.parent / "saved_models" / "config.json"
            if alt_path.exists():
                config_path = alt_path
                self.models_path = self.current_dir.parent / "saved_models"
            else:
                # 列出所有可能的位置帮助调试
                print("Looking for config.json in:")
                print(f"  - {config_path}")
                print(f"  - {alt_path}")
                print(f"  - {self.current_dir / 'config.json'}")
                print(f"  - {self.current_dir.parent / 'config.json'}")
                raise FileNotFoundError(f"配置文件不存在，已搜索多个位置")
        
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
            # 尝试上一级目录
            alt_path = self.current_dir.parent / "saved_models" / "LightGBM_model.pkl"
            if alt_path.exists():
                model_path = alt_path
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # 获取期望特征数
        if hasattr(self.model, 'n_features_in_'):
            self.expected_features = self.model.n_features_in_
        elif hasattr(self.model, 'feature_importances_'):
            self.expected_features = len(self.model.feature_importances_)
        else:
            self.expected_features = 70
        
        print(f"✓ Expected features: {self.expected_features}")
        
        # 3. 加载scaler
        scaler_path = self.models_path / "scaler.pkl"
        if not scaler_path.exists():
            # 尝试上一级目录
            alt_path = self.current_dir.parent / "saved_models" / "scaler.pkl"
            if alt_path.exists():
                scaler_path = alt_path
            else:
                raise FileNotFoundError(f"Scaler文件不存在: {scaler_path}")
        
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from: {scaler_path}")
        print("=" * 60)
    
    def prepare_features(self, X_3d):
        """
        准备特征用于预测
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
        注意：为了匹配模型，会将单个时间点复制成3个时间点
        """
        if isinstance(input_data, pd.DataFrame):
            # 确保只使用需要的特征列
            X = input_data[self.feature_cols].values
        else:
            X = np.array(input_data)
        
        # 确保输入是2D数组
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        print(f"predict_single - Input shape: {X.shape}")
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 将单个时间点复制成3个时间点，形成3D数组
        n_samples = X_scaled.shape[0]
        # 创建3D数组: (n_samples, 3, n_features)
        X_3d = np.array([X_scaled] * 3).transpose(1, 0, 2)
        print(f"predict_single - 3D shape: {X_3d.shape}")
        
        # 准备特征（从3D到70/84个统计特征）
        features = self.prepare_features(X_3d)
        print(f"predict_single - Features shape: {features.shape}, Expected: {self.expected_features}")
        
        # 预测
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features)
            return proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
        else:
            return self.model.predict(features)
    
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
        SHAP预测函数
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
            
            shap_values = explainer(X_flat, silent=False)
            return shap_values
            
        except Exception as e:
            print(f"  PermutationExplainer failed: {e}")
            print("  Falling back to KernelExplainer...")
            
            try:
                explainer = shap.KernelExplainer(
                    self.predict_for_shap,
                    background,
                    link="identity"
                )
                
                shap_values_raw = explainer.shap_values(
                    X_flat,
                    nsamples=150,
                    silent=True
                )
                
                if isinstance(shap_values_raw, list):
                    shap_values_raw = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
                
                shap_values = shap.Explanation(
                    values=shap_values_raw,
                    base_values=explainer.expected_value,
                    data=X_flat
                )
                
                return shap_values
                
            except Exception as e2:
                print(f"  All explainers failed: {e2}")
                return None
    
    def generate_beeswarm_plot(self, shap_values, max_display=15):
        """
        生成蜂群图的base64编码
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            shap.summary_plot(
                shap_values.values,
                shap_values.data,
                feature_names=self.feature_cols,
                show=False,
                max_display=max_display,
                plot_type="dot",
                alpha=0.6
            )
            
            plt.tight_layout()
            
            # 转换为base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating beeswarm plot: {e}")
            return None
    
    def generate_bar_plot(self, shap_values, max_display=15):
        """
        生成条形图的base64编码
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            shap.summary_plot(
                shap_values.values,
                shap_values.data,
                feature_names=self.feature_cols,
                show=False,
                plot_type="bar",
                max_display=max_display
            )
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating bar plot: {e}")
            return None
    
    def generate_waterfall_plot(self, shap_values, case_idx=0, period_idx=0):
        """
        为特定病例和时间段生成瀑布图
        """
        try:
            # 获取该病例的SHAP值
            if len(shap_values.values.shape) == 2:
                # 已经是2D
                shap_values_case = shap_values.values[case_idx]
            else:
                # 需要重塑
                n_features = len(self.feature_cols)
                shap_3d = shap_values.values.reshape(-1, 3, n_features)
                shap_values_case = shap_3d[case_idx, period_idx, :]
            
            # 获取特征值
            if hasattr(shap_values, 'data'):
                if len(shap_values.data.shape) == 2:
                    feature_values = shap_values.data[case_idx]
                else:
                    feature_3d = shap_values.data.reshape(-1, 3, n_features)
                    feature_values = feature_3d[case_idx, period_idx, :]
            else:
                feature_values = np.zeros(len(self.feature_cols))
            
            # 获取基准值
            if hasattr(shap_values, 'base_values'):
                base_value = shap_values.base_values[0] if isinstance(shap_values.base_values, (list, np.ndarray)) else shap_values.base_values
            else:
                base_value = 0.5
            
            # 创建特征名称列表（包含特征值）
            feature_names_with_values = []
            for j in range(len(self.feature_cols)):
                feat_name = self.feature_cols[j]
                feat_value = feature_values[j]
                feature_names_with_values.append(f"{feat_name} = {feat_value:.2f}")
            
            # 创建Explanation对象
            explanation = shap.Explanation(
                values=shap_values_case,
                base_values=base_value,
                data=feature_values,
                feature_names=feature_names_with_values
            )
            
            # 绘制瀑布图
            fig, ax = plt.subplots(figsize=(12, 6))
            
            shap.waterfall_plot(
                explanation,
                max_display=10,
                show=False
            )
            
            plt.tight_layout()
            
            # 转换为base64
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            print(f"Error generating waterfall plot: {e}")
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
    
    def get_global_shap_analysis(self, sample_data, n_background=100):
        """
        获取全局SHAP分析（蜂群图和条形图）
        sample_data: 3D数组 shape (n_samples, 3, n_features)
        """
        # 计算SHAP值
        shap_values = self.compute_shap_values(sample_data, n_background)
        
        if shap_values is None:
            return None
        
        # 生成图表
        beeswarm = self.generate_beeswarm_plot(shap_values)
        bar = self.generate_bar_plot(shap_values)
        
        # 计算每个样本的预测概率
        probas = self.predict_temporal(sample_data)
        
        return {
            'beeswarm': beeswarm,
            'bar': bar,
            'shap_summary': {
                'mean_abs_shap': np.mean(np.abs(shap_values.values), axis=0).tolist(),
                'feature_names': self.feature_cols
            },
            'predictions': probas.tolist()
        }
    
    def get_case_waterfall(self, sample_data, case_idx=0, period_idx=0):
        """
        获取单个病例的瀑布图
        sample_data: 3D数组 shape (n_samples, 3, n_features)
        """
        # 计算SHAP值
        shap_values = self.compute_shap_values(sample_data, n_background=50)
        
        if shap_values is None:
            return None
        
        # 生成瀑布图
        waterfall = self.generate_waterfall_plot(shap_values, case_idx, period_idx)
        
        # 预测概率
        proba = self.predict_temporal(sample_data)[case_idx]
        
        return {
            'waterfall': waterfall,
            'probability': float(proba),
            'base_value': float(shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.5),
            'feature_names': self.feature_cols
        }


# 创建单例 - 使用绝对路径
try:
    # 获取当前文件所在目录
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    models_dir = current_dir / "saved_models"
    
    print(f"Initializing predictor with models from: {models_dir}")
    predictor = SepsisPredictor(models_path=str(models_dir))
except Exception as e:
    print(f"Error initializing predictor: {e}")
    # 如果失败，尝试相对路径
    print("Trying relative path...")
    predictor = SepsisPredictor()