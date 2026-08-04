import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                            accuracy_score, f1_score, confusion_matrix,
                            recall_score, precision_score, roc_curve,
                            classification_report, average_precision_score,
                            brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import json
from tqdm import tqdm
import random
import os
from datetime import datetime
from scipy import stats
import copy
from sklearn.base import clone as sklearn_clone
import joblib

# 设置随机种子保证可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 增强的配置参数 ====================
class EnhancedConfig:
    # 数据路径
    train_path = "/Users/lizeqi/Desktop/MIMIC/MIMIC数据R语言代码-新版/MIMIC新/data analysis/train_data.csv"
    test_path = "/Users/lizeqi/Desktop/MIMIC/MIMIC数据R语言代码-新版/MIMIC新/data analysis/test_data.csv"
    external_path = "/Users/lizeqi/Desktop/MIMIC/MIMIC数据R语言代码-新版/MIMIC新/data analysis/external validation.csv"
    
    # 特征和标签
    feature_cols = ["heart_rate", "sbp", "resp_rate", "spo2", "wbc", 
                    "hemoglobin", "platelet", "bun", "pt", "glucose", 
                    "sodium", "potassium", "chloride", "bicarbonate"]
    id_cols = ["subject_id", "vital_period"]
    label_col = "gram_type"
    
    # 模型参数
    sequence_length = 3
    input_dim = len(feature_cols)
    lstm_hidden_dim = 64
    lstm_num_layers = 2
    dropout_rate = 0.3
    lstm_bidirectional = True
    
    # GRU模型参数
    gru_hidden_dim = 64
    gru_num_layers = 2
    gru_bidirectional = True
    
    # 随机森林参数
    rf_n_estimators = 200
    rf_max_depth = 15
    rf_min_samples_split = 5
    
    # 训练参数
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    num_epochs = 200
    early_stopping_patience = 20
    
    # 验证集比例
    val_ratio = 0.1
    
    # Bootstrap参数
    n_bootstrap = 1000
    confidence_level = 0.95
    
    # 交叉验证
    n_folds = 10
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存路径
    save_dir = "./sepsis_classification_fixed_leakage/"
    
    # 最佳模型保存相关配置
    best_model_save_dir = "./best_models_fixed_leakage/"
    save_onnx_for_web = True
    save_pytorch_model = True
    save_scikit_model = True
    
config = EnhancedConfig()

# ==================== 评估函数 ====================
def bootstrap_ci_fixed(y_true, y_pred_prob, metric_func, n_bootstrap=1000, confidence=0.95):
    """计算指标的Bootstrap置信区间"""
    scores = []
    n = len(y_true)
    
    # 检查数据有效性
    if len(np.unique(y_true)) < 2:
        try:
            point_estimate = metric_func(y_true, y_pred_prob)
            return point_estimate, point_estimate, point_estimate
        except:
            return 0.5, 0.5, 0.5
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        y_true_sample = y_true[indices]
        y_pred_prob_sample = y_pred_prob[indices]
        
        # 检查样本中是否有两种标签
        if len(np.unique(y_true_sample)) < 2:
            continue
        
        try:
            score = metric_func(y_true_sample, y_pred_prob_sample)
            if not np.isnan(score):
                scores.append(score)
        except:
            continue
    
    if len(scores) > 10:
        scores = np.array(scores)
        lower = np.percentile(scores, (1-confidence)/2 * 100)
        upper = np.percentile(scores, (1+confidence)/2 * 100)
        mean_score = np.mean(scores)
        return mean_score, lower, upper
    else:
        if len(scores) == 0:
            print(f"  [警告] Bootstrap: 所有{n_bootstrap}次重采样均无法计算指标，使用点估计代替")
        else:
            print(f"  [警告] Bootstrap: 仅{len(scores)}/{n_bootstrap}次有效重采样(<10)，置信区间不可靠，使用点估计代替")
        try:
            point_estimate = metric_func(y_true, y_pred_prob)
            return point_estimate, point_estimate, point_estimate
        except:
            return 0.5, 0.5, 0.5

def calculate_metrics_with_ci_fixed(y_true, y_pred, y_pred_prob, model_name="Model", optimal_threshold=0.5):
    """计算所有评估指标及置信区间

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测类别
    y_pred_prob : array-like
        预测概率
    model_name : str
        模型名称
    optimal_threshold : float
        最优分类阈值
    """

    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_prob = np.array(y_pred_prob)

    # 计算基本指标
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # 计算AUROC及其置信区间
    if len(np.unique(y_true)) > 1:
        try:
            auroc = roc_auc_score(y_true, y_pred_prob)
            auroc_mean, auroc_lower, auroc_upper = bootstrap_ci_fixed(y_true, y_pred_prob, roc_auc_score)
        except:
            auroc = 0.5
            auroc_mean, auroc_lower, auroc_upper = 0.5, 0.5, 0.5
    else:
        auroc = 0.5
        auroc_mean, auroc_lower, auroc_upper = 0.5, 0.5, 0.5

    # 计算AUPRC
    if len(np.unique(y_true)) > 1:
        try:
            auprc = average_precision_score(y_true, y_pred_prob)
            auprc_mean, auprc_lower, auprc_upper = bootstrap_ci_fixed(y_true, y_pred_prob, average_precision_score)
        except Exception as e:
            print(f"  AUPRC计算错误: {str(e)}")
            auprc = 0.5
            auprc_mean, auprc_lower, auprc_upper = 0.5, 0.5, 0.5
    else:
        auprc = 0.5
        auprc_mean, auprc_lower, auprc_upper = 0.5, 0.5, 0.5

    # 计算Brier Score
    brier = brier_score_loss(y_true, y_pred_prob)

    # 计算其他指标
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        'Model': model_name,
        'AUROC': f"{auroc_mean:.3f} ({auroc_lower:.3f}-{auroc_upper:.3f})",
        'AUROC_value': auroc_mean,
        'AUPRC': f"{auprc_mean:.3f} ({auprc_lower:.3f}-{auprc_upper:.3f})",
        'AUPRC_value': auprc_mean,
        'Sensitivity': f"{sensitivity:.3f}",
        'Specificity': f"{specificity:.3f}",
        'Accuracy': f"{accuracy:.3f}",
        'PPV': f"{ppv:.3f}",
        'NPV': f"{npv:.3f}",
        'F1_Score': f"{f1:.3f}",
        'Brier_Score': f"{brier:.3f}",
        'Optimal_Threshold': f"{optimal_threshold:.3f}",
        'Positives': int(y_true.sum()),
        'Total': len(y_true),
        'Pos_Ratio': f"{y_true.sum()/len(y_true):.3f}"
    }

    return metrics

# ==================== 阈值优化函数 ====================
def find_optimal_threshold(y_true, y_pred_prob, metric='f1'):
    """
    寻找最佳分类阈值
    
    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_prob : array-like
        预测概率
    metric : str
        优化指标 ('f1', 'youden', 'precision', 'recall')
    
    Returns:
    --------
    optimal_threshold : float
        最佳阈值
    best_score : float
        最佳分数
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # 检查数据有效性
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    
    # 生成候选阈值
    thresholds = np.linspace(0.01, 0.99, 99)
    
    best_threshold = 0.5
    best_score = 0.0
    
    if metric == 'f1':
        for thresh in thresholds:
            y_pred = (y_pred_prob >= thresh).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = thresh
    elif metric == 'youden':
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_prob)
        # 直接在所有ROC阈值点上计算Youden指数，取最大值
        # 兼容旧版sklearn：确保thresholds与fpr/tpr长度一致
        if len(roc_thresholds) > len(fpr):
            roc_thresholds = roc_thresholds[:len(fpr)]
        youden_values = tpr - fpr
        best_idx = np.argmax(youden_values)
        best_threshold = float(roc_thresholds[best_idx])
        best_score = float(youden_values[best_idx])
    elif metric in ['precision', 'recall']:
        precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
        # precision_recall_curve 返回的 thresholds 比 precision/recall 少一个元素
        # 最后一个 precision/recall 对对应 threshold=0（全部预测为阳性），无对应阈值
        if len(thresholds_pr) > 0:
            # 截取与 thresholds 等长的 precision/recall
            valid_precisions = precisions[:-1]
            valid_recalls = recalls[:-1]
            if metric == 'precision':
                best_idx = np.argmax(valid_precisions)
            else:
                best_idx = np.argmax(valid_recalls)
            best_threshold = float(thresholds_pr[best_idx])
            best_score = float(precisions[best_idx] if metric == 'precision' else recalls[best_idx])
    else:
        # 默认使用F1
        for thresh in thresholds:
            y_pred = (y_pred_prob >= thresh).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = thresh
    
    return best_threshold, best_score

# ==================== 安全的数据预处理器（修复数据泄露） ====================
class SecureDataPreprocessor:
    """安全的数据预处理器，彻底杜绝数据泄露"""
    def __init__(self, feature_cols, sequence_length=3):
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.scaler = None
        self.feature_stats = {}  # 存储训练集的统计信息用于特征工程
        self.imputation_values = None  # 存储训练集的填充值
        self.is_fitted = False
        
    def fit(self, train_df):
        """仅在训练集上拟合预处理参数（严格隔离）"""
        print("安全预处理：在训练集上拟合预处理参数...")
        
        # 1. 计算用于缺失值填充的统计量（使用训练集全局统计）
        all_features = []
        patient_ids = train_df['subject_id'].unique()
        
        for patient_id in patient_ids:
            patient_df = train_df[train_df['subject_id'] == patient_id]
            features = patient_df[self.feature_cols].values
            all_features.append(features)
        
        if len(all_features) > 0:
            all_features = np.vstack(all_features)
            # 计算每个特征的中位数（用于填充）
            self.imputation_values = np.nanmedian(all_features, axis=0)
            # 将NaN替换为0
            self.imputation_values = np.where(
                np.isnan(self.imputation_values), 0, self.imputation_values
            )
        else:
            self.imputation_values = np.zeros(len(self.feature_cols))
        
        # 2. 拟合标准化器
        if len(all_features) > 0:
            self.scaler = StandardScaler()
            # 先用训练集全局统计量填充缺失值，再拟合标准化器
            all_features_filled = all_features.copy()
            for i in range(all_features_filled.shape[1]):
                col_values = all_features_filled[:, i]
                if np.isnan(col_values).any():
                    all_features_filled[:, i] = np.where(
                        np.isnan(col_values), 
                        self.imputation_values[i], 
                        col_values
                    )
            self.scaler.fit(all_features_filled)
        
        # 3. 计算用于特征工程的统计信息
        self._compute_feature_statistics(train_df)
        
        self.is_fitted = True
        print(f"  填充值已计算: {self.imputation_values[:5]}...")
        print(f"  特征统计信息已计算: {len(self.feature_stats)} 个统计量")
        
        return self
    
    def _compute_feature_statistics(self, train_df):
        """计算训练集的特征统计信息，用于安全特征工程"""
        patient_ids = train_df['subject_id'].unique()
        
        # 收集所有患者的特征统计量
        all_mean_features = []
        all_std_features = []
        all_max_features = []
        all_min_features = []
        all_median_features = []
        
        for patient_id in patient_ids:
            patient_df = train_df[train_df['subject_id'] == patient_id].sort_values('vital_period')
            features = patient_df[self.feature_cols].values
            
            if len(features) > 0:
                # 处理缺失值（使用训练集全局统计量）
                features_filled = self._impute_missing_values(features)
                
                # 计算统计特征
                if len(features_filled) > 0:
                    all_mean_features.append(np.mean(features_filled, axis=0))
                    all_std_features.append(np.std(features_filled, axis=0))
                    all_max_features.append(np.max(features_filled, axis=0))
                    all_min_features.append(np.min(features_filled, axis=0))
                    all_median_features.append(np.median(features_filled, axis=0))
        
        # 计算训练集上的全局统计量
        if len(all_mean_features) > 0:
            self.feature_stats = {
                'mean_of_means': np.mean(all_mean_features, axis=0),
                'std_of_means': np.std(all_mean_features, axis=0),
                'mean_of_stds': np.mean(all_std_features, axis=0),
                'std_of_stds': np.std(all_std_features, axis=0),
                'mean_of_maxs': np.mean(all_max_features, axis=0),
                'mean_of_mins': np.mean(all_min_features, axis=0),
                'mean_of_medians': np.mean(all_median_features, axis=0)
            }
        else:
            # 如果没有数据，创建空的统计信息
            n_features = len(self.feature_cols)
            self.feature_stats = {
                'mean_of_means': np.zeros(n_features),
                'std_of_means': np.ones(n_features),
                'mean_of_stds': np.zeros(n_features),
                'std_of_stds': np.ones(n_features),
                'mean_of_maxs': np.zeros(n_features),
                'mean_of_mins': np.zeros(n_features),
                'mean_of_medians': np.zeros(n_features)
            }
    
    def _impute_missing_values(self, features):
        """使用训练集统计量填充缺失值"""
        if self.imputation_values is None:
            return features
        
        features_imputed = features.copy()
        for i in range(features.shape[1]):
            col_values = features[:, i]
            if np.isnan(col_values).any():
                features_imputed[:, i] = np.where(
                    np.isnan(col_values), 
                    self.imputation_values[i], 
                    col_values
                )
        
        return features_imputed
    
    def transform_sequence(self, patient_df):
        """转换单个患者的数据（安全无泄露）"""
        if not self.is_fitted:
            raise ValueError("必须先调用fit方法")
        
        patient_df = patient_df.sort_values('vital_period')
        features = patient_df[self.feature_cols].values
        actual_length = len(features)
        
        # 处理缺失的时间段
        if actual_length < self.sequence_length:
            if actual_length > 0:
                # 用训练集统计量填充缺失的时间段
                padding = np.tile(self.imputation_values, (self.sequence_length - actual_length, 1))
                features = np.vstack([features, padding])
            else:
                features = np.tile(self.imputation_values, (self.sequence_length, 1))
        elif actual_length > self.sequence_length:
            features = features[:self.sequence_length]
        
        # 处理特征中的NaN值（使用训练集统计量）
        features = self._impute_missing_values(features)
        
        # 标准化
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features, min(actual_length, self.sequence_length)
    
    def extract_features_for_ml(self, patient_df):
        """为传统ML模型提取特征（使用训练集统计信息，防止泄露）"""
        features, actual_length = self.transform_sequence(patient_df)
        
        if actual_length > 0:
            valid_features = features[:actual_length]
            
            # 基于训练集统计信息计算标准化后的统计特征
            mean_features = np.mean(valid_features, axis=0)
            std_features = np.std(valid_features, axis=0)
            max_features = np.max(valid_features, axis=0)
            min_features = np.min(valid_features, axis=0)
            median_features = np.median(valid_features, axis=0)
        else:
            # 如果没有有效数据，使用训练集的平均统计量
            mean_features = self.feature_stats['mean_of_means']
            std_features = self.feature_stats['mean_of_stds']
            max_features = self.feature_stats['mean_of_maxs']
            min_features = self.feature_stats['mean_of_mins']
            median_features = self.feature_stats['mean_of_medians']
        
        # 组合特征（移除趋势特征，避免泄露）
        combined_features = np.concatenate([
            mean_features, std_features, max_features, 
            min_features, median_features
        ])
        
        return combined_features
    
    def get_scaler(self):
        """获取标准化器"""
        return self.scaler

# ==================== 安全的数据集创建函数 ====================
def create_dataset_dict(df, feature_cols, label_col, preprocessor=None):
    """创建数据集字典（使用预处理器防止泄露）"""
    patients_data = []
    patient_ids = df['subject_id'].unique()
    
    for patient_id in patient_ids:
        patient_df = df[df['subject_id'] == patient_id]
        
        if preprocessor is not None:
            features, actual_length = preprocessor.transform_sequence(patient_df)
            label = patient_df[label_col].iloc[0]
        else:
            # 回退到原始方法（不推荐）
            print("警告：使用不安全的预处理方法，可能存在数据泄露风险")
            patient_df = patient_df.sort_values('vital_period')
            features = patient_df[feature_cols].values
            label = patient_df[label_col].iloc[0]
            actual_length = len(features)
            
            # 处理缺失的时间段
            if actual_length < config.sequence_length:
                if actual_length > 0:
                    col_medians = np.nanmedian(features, axis=0)
                    padding = np.tile(col_medians, (config.sequence_length - actual_length, 1))
                    features = np.vstack([features, padding])
                else:
                    features = np.zeros((config.sequence_length, len(feature_cols)))
            elif actual_length > config.sequence_length:
                features = features[:config.sequence_length]
            
            # 处理特征中的NaN值
            for i in range(features.shape[1]):
                col_values = features[:, i]
                if np.isnan(col_values).any():
                    col_median = np.nanmedian(col_values)
                    if np.isnan(col_median):
                        col_median = 0
                    features[:, i] = np.where(np.isnan(col_values), col_median, col_values)
            
            actual_length = min(actual_length, config.sequence_length)
        
        patients_data.append({
            'features': features.astype(np.float32),
            'label': label,
            'actual_length': actual_length,
            'patient_id': patient_id
        })
    
    return patients_data

# ==================== 数据集类 ====================
class SepsisTimeSeriesDataset(Dataset):
    def __init__(self, patients_data):
        self.patients_data = patients_data
    
    def __len__(self):
        return len(self.patients_data)
    
    def __getitem__(self, idx):
        data = self.patients_data[idx]
        features = data['features'].astype(np.float32)
        label = data['label']
        actual_length = data['actual_length']
        
        # 创建掩码（1表示真实数据，0表示填充）
        mask = np.zeros(config.sequence_length)
        mask[:actual_length] = 1
        
        return {
            'features': torch.FloatTensor(features),
            'label': torch.FloatTensor([label]),
            'mask': torch.FloatTensor(mask),
            'actual_length': actual_length
        }
    
    def get_labels(self):
        """获取所有标签"""
        return np.array([data['label'] for data in self.patients_data])
    
    def get_patient_ids(self):
        """获取所有患者ID"""
        return [data.get('patient_id', i) for i, data in enumerate(self.patients_data)]
    
    def split_by_patient(self, val_ratio=0.1, random_state=42):
        """按患者ID划分训练集和验证集"""
        patient_ids = self.get_patient_ids()
        labels = self.get_labels()
        
        # 使用分层划分确保验证集中的类别比例与训练集相似
        train_idx, val_idx = train_test_split(
            range(len(self)), 
            test_size=val_ratio, 
            random_state=random_state,
            stratify=labels
        )
        
        train_data = [self.patients_data[i] for i in train_idx]
        val_data = [self.patients_data[i] for i in val_idx]
        
        return SepsisTimeSeriesDataset(train_data), SepsisTimeSeriesDataset(val_data)

# ==================== 改进的LSTM模型 ====================
class ImprovedLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, bidirectional=True):
        super(ImprovedLSTMClassifier, self).__init__()
        
        # 增加dropout率
        lstm_dropout = dropout_rate if num_layers > 1 else 0
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 增加批量归一化层
        self.batch_norm = nn.BatchNorm1d(lstm_output_dim)
        
        # 增加更多dropout层和减小网络宽度
        self.dropout1 = nn.Dropout(dropout_rate * 1.3)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """更好的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'lstm' in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and 'fc' in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name and 'fc' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and 'batch_norm' in name:
                nn.init.ones_(param)
            elif 'bias' in name and 'batch_norm' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, actual_lengths):
        batch_size = x.size(0)
        
        # 处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, actual_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 获取每个序列最后一个时间步的输出
        last_outputs = []
        for i in range(batch_size):
            last_idx = actual_lengths[i] - 1
            last_outputs.append(output[i, last_idx, :])
        last_output = torch.stack(last_outputs)
        
        # 添加批量归一化
        last_output = self.batch_norm(last_output)
        
        # 添加额外的dropout以减少过拟合
        last_output = self.dropout1(last_output)
        
        # 全连接层
        out = self.fc(last_output)
        
        return out

# ==================== 改进的GRU模型 ====================
class ImprovedGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, bidirectional=True):
        super(ImprovedGRUClassifier, self).__init__()
        
        # 增加dropout率
        gru_dropout = dropout_rate if num_layers > 1 else 0
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 增加批量归一化层
        self.batch_norm = nn.BatchNorm1d(gru_output_dim)
        
        # 增加更多dropout层和减小网络宽度
        self.dropout1 = nn.Dropout(dropout_rate * 1.3)
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.2),
            
            nn.Linear(16, 1)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """更好的权重初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'gru' in name and param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'gru' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and 'fc' in name and param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'bias' in name and 'fc' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name and 'batch_norm' in name:
                nn.init.ones_(param)
            elif 'bias' in name and 'batch_norm' in name:
                nn.init.zeros_(param)
        
    def forward(self, x, actual_lengths):
        batch_size = x.size(0)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, actual_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        last_outputs = []
        for i in range(batch_size):
            last_idx = actual_lengths[i] - 1
            last_outputs.append(output[i, last_idx, :])
        last_output = torch.stack(last_outputs)
        
        # 添加批量归一化
        last_output = self.batch_norm(last_output)
        
        # 添加额外的dropout
        last_output = self.dropout1(last_output)
        
        out = self.fc(last_output)
        
        return out

# ==================== 单独的LSTM模型训练和评估类 ====================
class StandaloneLSTMModel:
    """单独的LSTM模型"""
    def __init__(self, lstm_hidden_dim, lstm_num_layers, dropout_rate, bidirectional=True):
        self.model = None
        self.optimal_threshold = 0.5
        self.config = {
            'lstm_hidden_dim': lstm_hidden_dim,
            'lstm_num_layers': lstm_num_layers,
            'dropout_rate': dropout_rate,
            'bidirectional': bidirectional
        }
        
    def train(self, train_dataset, val_dataset=None):
        """训练模型"""
        print("训练单独LSTM模型...")
        
        if val_dataset is None:
            train_subset, val_subset = train_dataset.split_by_patient(
                val_ratio=config.val_ratio, random_state=42
            )
        else:
            train_subset = train_dataset
            val_subset = val_dataset
        
        print(f"  训练集: {len(train_subset)} 例患者")
        print(f"  验证集: {len(val_subset)} 例患者")
        
        # 计算类别信息
        train_labels = train_subset.get_labels()
        pos_ratio = sum(train_labels) / len(train_labels)
        neg_ratio = 1 - pos_ratio
        print(f"  训练集阳性比例: {pos_ratio:.3f} ({sum(train_labels)}/{len(train_labels)})")
        print(f"  类别不平衡比例: {max(pos_ratio, neg_ratio):.1f}:{min(pos_ratio, neg_ratio):.1f}")
        
        self.model = ImprovedLSTMClassifier(
            input_dim=config.input_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout_rate=config.dropout_rate,
            bidirectional=config.lstm_bidirectional
        )
        
        self.model = self._train_model_fixed(self.model, train_subset, val_subset)
        
        # 在验证集上计算最优阈值
        print("  在验证集上计算最优阈值...")
        y_val_true, _, y_val_prob = self.predict(val_subset)
        optimal_threshold, best_f1 = find_optimal_threshold(y_val_true, y_val_prob, metric='f1')
        self.optimal_threshold = optimal_threshold
        print(f"  最优阈值: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
        
        print("  模型训练完成!")
        
    def _train_model_fixed(self, model, train_dataset, internal_val_dataset):
        """训练模型部分

        注：drop_last=True 防止最后一个 batch 只有 1 个样本时 BatchNorm 崩溃。
        每轮最多丢弃 batch_size-1 个样本，shuffle 后不同 epoch 丢弃的样本不同，
        对整体训练影响可忽略。
        """
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        internal_val_loader = DataLoader(internal_val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # 计算类别权重
        train_labels = train_dataset.get_labels()
        pos_ratio = sum(train_labels) / len(train_labels)
        
        # pos_weight = n_negative / n_positive
        if pos_ratio > 0 and pos_ratio < 1:
            pos_weight_tensor = torch.tensor([(1 - pos_ratio) / pos_ratio]).to(config.device)
        else:
            pos_weight_tensor = torch.tensor([1.0]).to(config.device)
        
        print(f"  pos_weight: {pos_weight_tensor.item():.3f} (用于BCEWithLogitsLoss)")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                             weight_decay=config.weight_decay * 2)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6
        )
        
        best_val_auroc = 0.0
        patience_counter = 0
        model.to(config.device)
        
        for epoch in range(config.num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                features = batch['features'].to(config.device)
                labels = batch['label'].to(config.device)
                actual_lengths = batch['actual_length']
                
                optimizer.zero_grad()
                outputs = model(features, actual_lengths)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # 在内部验证集上评估
            model.eval()
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in internal_val_loader:
                    features = batch['features'].to(config.device)
                    labels = batch['label'].to(config.device)
                    actual_lengths = batch['actual_length']
                    
                    outputs = model(features, actual_lengths)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                    val_probs.extend(probs.flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())
            
            val_labels = np.array(val_labels)
            val_probs = np.array(val_probs)
            
            # 计算验证集AUROC
            if len(np.unique(val_labels)) > 1:
                val_auroc = roc_auc_score(val_labels, val_probs)
            else:
                val_auroc = 0.5
            
            scheduler.step(val_auroc)
            
            # 保存最佳模型
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= config.early_stopping_patience:
                print(f"    早停于第{epoch+1}轮")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    第{epoch+1}轮: 训练损失 = {avg_train_loss:.4f}, 验证集AUROC = {val_auroc:.4f}")
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        return model
    
    def predict(self, dataset, use_optimal_threshold=True):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练!")
        
        self.model.eval()
        y_true_list = []
        y_pred_prob_list = []
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(config.device)
                labels = batch['label'].cpu().numpy().flatten()
                actual_lengths = batch['actual_length']
                
                outputs = self.model(features, actual_lengths)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                y_true_list.extend(labels)
                y_pred_prob_list.extend(probs)
        
        y_true = np.array(y_true_list)
        y_pred_prob = np.array(y_pred_prob_list)
        
        # 使用最优阈值或默认0.5
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        return y_true, y_pred, y_pred_prob

# ==================== 单独的GRU模型训练和评估类 ====================
class StandaloneGRUModel:
    """单独的GRU模型"""
    def __init__(self, gru_hidden_dim, gru_num_layers, dropout_rate, bidirectional=True):
        self.model = None
        self.optimal_threshold = 0.5
        self.config = {
            'gru_hidden_dim': gru_hidden_dim,
            'gru_num_layers': gru_num_layers,
            'dropout_rate': dropout_rate,
            'bidirectional': bidirectional
        }
        
    def train(self, train_dataset, val_dataset=None):
        """训练模型"""
        print("训练单独GRU模型...")
        
        if val_dataset is None:
            train_subset, val_subset = train_dataset.split_by_patient(
                val_ratio=config.val_ratio, random_state=42
            )
        else:
            train_subset = train_dataset
            val_subset = val_dataset
        
        print(f"  训练集: {len(train_subset)} 例患者")
        print(f"  验证集: {len(val_subset)} 例患者")
        
        # 计算类别信息
        train_labels = train_subset.get_labels()
        pos_ratio = sum(train_labels) / len(train_labels)
        neg_ratio = 1 - pos_ratio
        print(f"  训练集阳性比例: {pos_ratio:.3f} ({sum(train_labels)}/{len(train_labels)})")
        print(f"  类别不平衡比例: {max(pos_ratio, neg_ratio):.1f}:{min(pos_ratio, neg_ratio):.1f}")
        
        self.model = ImprovedGRUClassifier(
            input_dim=config.input_dim,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            dropout_rate=config.dropout_rate,
            bidirectional=config.gru_bidirectional
        )
        
        self.model = self._train_model_fixed(self.model, train_subset, val_subset)
        
        # 在验证集上计算最优阈值
        print("  在验证集上计算最优阈值...")
        y_val_true, _, y_val_prob = self.predict(val_subset)
        optimal_threshold, best_f1 = find_optimal_threshold(y_val_true, y_val_prob, metric='f1')
        self.optimal_threshold = optimal_threshold
        print(f"  最优阈值: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
        
        print("  模型训练完成!")
        
    def _train_model_fixed(self, model, train_dataset, internal_val_dataset):
        """训练模型部分

        注：drop_last=True 防止最后一个 batch 只有 1 个样本时 BatchNorm 崩溃。
        每轮最多丢弃 batch_size-1 个样本，shuffle 后不同 epoch 丢弃的样本不同，
        对整体训练影响可忽略。
        """
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        internal_val_loader = DataLoader(internal_val_dataset, batch_size=config.batch_size, shuffle=False)
        
        train_labels = train_dataset.get_labels()
        pos_ratio = sum(train_labels) / len(train_labels)
        
        # pos_weight = n_negative / n_positive
        if pos_ratio > 0 and pos_ratio < 1:
            pos_weight_tensor = torch.tensor([(1 - pos_ratio) / pos_ratio]).to(config.device)
        else:
            pos_weight_tensor = torch.tensor([1.0]).to(config.device)
        
        print(f"  pos_weight: {pos_weight_tensor.item():.3f} (用于BCEWithLogitsLoss)")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                             weight_decay=config.weight_decay * 2)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6
        )
        
        best_val_auroc = 0.0
        patience_counter = 0
        model.to(config.device)
        
        for epoch in range(config.num_epochs):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                features = batch['features'].to(config.device)
                labels = batch['label'].to(config.device)
                actual_lengths = batch['actual_length']
                
                optimizer.zero_grad()
                outputs = model(features, actual_lengths)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in internal_val_loader:
                    features = batch['features'].to(config.device)
                    labels = batch['label'].to(config.device)
                    actual_lengths = batch['actual_length']
                    
                    outputs = model(features, actual_lengths)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                    val_probs.extend(probs.flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())
            
            val_labels = np.array(val_labels)
            val_probs = np.array(val_probs)
            
            if len(np.unique(val_labels)) > 1:
                val_auroc = roc_auc_score(val_labels, val_probs)
            else:
                val_auroc = 0.5
            
            scheduler.step(val_auroc)
            
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= config.early_stopping_patience:
                print(f"    早停于第{epoch+1}轮")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"    第{epoch+1}轮: 训练损失 = {avg_train_loss:.4f}, 验证集AUROC = {val_auroc:.4f}")
        
        model.load_state_dict(best_model_state)
        
        return model
    
    def predict(self, dataset, use_optimal_threshold=True):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练!")
        
        self.model.eval()
        y_true_list = []
        y_pred_prob_list = []
        
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(config.device)
                labels = batch['label'].cpu().numpy().flatten()
                actual_lengths = batch['actual_length']
                
                outputs = self.model(features, actual_lengths)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                y_true_list.extend(labels)
                y_pred_prob_list.extend(probs)
        
        y_true = np.array(y_true_list)
        y_pred_prob = np.array(y_pred_prob_list)
        
        # 使用最优阈值或默认0.5
        threshold = self.optimal_threshold if use_optimal_threshold else 0.5
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        return y_true, y_pred, y_pred_prob

# ==================== 修复的增强传统机器学习模型 ====================
class EnhancedTraditionalMLModels:
    def __init__(self):
        self.models = {}
        self.fitted_models = {}
        self.preprocessors = {}
        self.optimal_thresholds = {}
        self.class_ratios = {}
    
    def _create_models(self, pos_ratio):
        """根据数据分布动态创建模型"""
        # 计算scale_pos_weight
        if pos_ratio > 0 and pos_ratio < 1:
            scale_pos_weight = (1 - pos_ratio) / pos_ratio
        else:
            scale_pos_weight = 1.0
        
        print(f"  XGBoost scale_pos_weight: {scale_pos_weight:.3f}")
        
        return {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                   min_samples_split=5, class_weight='balanced', 
                                                   random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                                    scale_pos_weight=scale_pos_weight,  # 动态计算
                                    random_state=42, eval_metric='logloss',
                                    use_label_encoder=False, verbosity=0),
            'LightGBM': lgb.LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, 
                                          class_weight='balanced', random_state=42, verbose=-1),
            'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, 
                                                           max_depth=5, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1500,
                                early_stopping=True, random_state=42)
        }
    
    def train_and_evaluate_with_cv(self, train_df, test_df, n_folds=10):
        """使用安全预处理器进行交叉验证"""
        print(f"使用安全预处理器进行{n_folds}折交叉验证...")
        
        results = {}
        
        # 获取所有患者ID和标签
        patient_ids = train_df['subject_id'].unique()
        labels = []
        for pid in patient_ids:
            label = train_df[train_df['subject_id'] == pid][config.label_col].iloc[0]
            labels.append(label)
        labels = np.array(labels)
        pos_ratio = labels.mean()
        neg_ratio = 1 - pos_ratio
        
        print(f"\n训练集阳性比例: {pos_ratio:.3f} ({sum(labels)}/{len(labels)})")
        print(f"类别不平衡比例: {max(pos_ratio, neg_ratio):.1f}:{min(pos_ratio, neg_ratio):.1f}")
        
        # 根据实际比例创建模型
        self.models = self._create_models(pos_ratio)
        self.class_ratios = {'positive': pos_ratio, 'negative': neg_ratio}
        
        # 为每个模型创建交叉验证结果
        for name, model in self.models.items():
            print(f"\n训练 {name} (使用安全预处理器)...")
            
            try:
                cv_scores = []
                cv_preprocessors = []
                cv_val_probs = []
                cv_val_labels = []
                
                # 创建交叉验证对象
                kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(patient_ids, labels)):
                    print(f"  第{fold+1}/{n_folds}折...")
                    
                    # 获取当前fold的训练和验证患者ID
                    train_patient_ids = patient_ids[train_idx]
                    val_patient_ids = patient_ids[val_idx]
                    
                    # 提取当前fold的训练和验证数据
                    train_fold_df = train_df[train_df['subject_id'].isin(train_patient_ids)]
                    val_fold_df = train_df[train_df['subject_id'].isin(val_patient_ids)]
                    
                    # 创建并拟合安全预处理器（仅在训练fold上）
                    fold_preprocessor = SecureDataPreprocessor(config.feature_cols, config.sequence_length)
                    fold_preprocessor.fit(train_fold_df)
                    cv_preprocessors.append(fold_preprocessor)
                    
                    # 为传统ML准备特征（使用安全预处理器）
                    X_train = []
                    y_train = []
                    
                    for pid in train_patient_ids:
                        patient_df = train_fold_df[train_fold_df['subject_id'] == pid]
                        features = fold_preprocessor.extract_features_for_ml(patient_df)
                        label = patient_df[config.label_col].iloc[0]
                        X_train.append(features)
                        y_train.append(label)
                    
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    # 准备验证集特征（使用训练fold的预处理器）
                    X_val = []
                    y_val = []
                    
                    for pid in val_patient_ids:
                        patient_df = val_fold_df[val_fold_df['subject_id'] == pid]
                        features = fold_preprocessor.extract_features_for_ml(patient_df)
                        label = patient_df[config.label_col].iloc[0]
                        X_val.append(features)
                        y_val.append(label)
                    
                    X_val = np.array(X_val)
                    y_val = np.array(y_val)
                    
                    # 克隆并训练模型
                    model_clone = self.clone_model(model)
                    model_clone.fit(X_train, y_train)
                    
                    # 预测
                    if hasattr(model_clone, 'predict_proba'):
                        y_val_prob = model_clone.predict_proba(X_val)[:, 1]
                    else:
                        y_val_pred = model_clone.predict(X_val)
                        y_val_prob = y_val_pred.astype(float)
                    
                    # 存储验证集结果用于阈值优化
                    cv_val_labels.extend(y_val)
                    cv_val_probs.extend(y_val_prob)
                    
                    # 计算验证集AUROC
                    if len(np.unique(y_val)) > 1:
                        val_auroc = roc_auc_score(y_val, y_val_prob)
                        cv_scores.append(val_auroc)
                
                # 在所有训练数据上训练最终模型
                print(f"  在所有训练数据上训练最终模型...")
                
                # 创建最终预处理器
                final_preprocessor = SecureDataPreprocessor(config.feature_cols, config.sequence_length)
                final_preprocessor.fit(train_df)
                self.preprocessors[name] = final_preprocessor
                
                # 准备训练特征
                X_train_all = []
                y_train_all = []
                
                for pid in patient_ids:
                    patient_df = train_df[train_df['subject_id'] == pid]
                    features = final_preprocessor.extract_features_for_ml(patient_df)
                    label = patient_df[config.label_col].iloc[0]
                    X_train_all.append(features)
                    y_train_all.append(label)
                
                X_train_all = np.array(X_train_all)
                y_train_all = np.array(y_train_all)
                
                # 训练最终模型
                final_model = self.clone_model(model)
                final_model.fit(X_train_all, y_train_all)
                self.fitted_models[name] = final_model
                
                # 在验证集上计算最优阈值
                cv_val_labels = np.array(cv_val_labels)
                cv_val_probs = np.array(cv_val_probs)
                optimal_threshold, best_f1 = find_optimal_threshold(cv_val_labels, cv_val_probs, metric='f1')
                self.optimal_thresholds[name] = optimal_threshold
                print(f"  最优阈值: {optimal_threshold:.3f} (F1: {best_f1:.3f})")
                
                # 在测试集上评估（使用最优阈值）
                X_test = []
                y_test = []
                
                test_patient_ids = test_df['subject_id'].unique()
                for pid in test_patient_ids:
                    patient_df = test_df[test_df['subject_id'] == pid]
                    features = final_preprocessor.extract_features_for_ml(patient_df)
                    label = patient_df[config.label_col].iloc[0]
                    X_test.append(features)
                    y_test.append(label)
                
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                
                # 预测
                if hasattr(final_model, 'predict_proba'):
                    y_pred_prob = final_model.predict_proba(X_test)[:, 1]
                else:
                    y_pred = final_model.predict(X_test)
                    y_pred_prob = y_pred.astype(float)
                
                # 使用最优阈值
                y_pred = (y_pred_prob >= optimal_threshold).astype(int)
                
                # 计算指标
                metrics = calculate_metrics_with_ci_fixed(y_test, y_pred, y_pred_prob, name, optimal_threshold)
                metrics['CV_AUROC_mean'] = np.mean(cv_scores) if cv_scores else 0
                metrics['CV_AUROC_std'] = np.std(cv_scores) if cv_scores else 0
                metrics['Optimal_Threshold'] = f"{optimal_threshold:.3f}"
                
                results[name] = {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_pred_prob': y_pred_prob,
                    'metrics': metrics,
                    'preprocessor': final_preprocessor,
                    'optimal_threshold': optimal_threshold
                }
                
                print(f"  测试集AUROC: {metrics['AUROC']}")
                print(f"  测试集AUPRC: {metrics['AUPRC']}")
                print(f"  交叉验证平均AUROC: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
                print(f"  最优阈值: {optimal_threshold:.3f}")
                
            except Exception as e:
                print(f"  训练{name}时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def clone_model(self, model):
        """克隆模型以保持独立性（使用 sklearn.base.clone 泛化处理所有模型类型）"""
        try:
            return sklearn_clone(model)
        except Exception:
            # 如果 sklearn clone 失败，回退到 get_params 手动重建
            return model.__class__(**model.get_params())
    
    def predict(self, model_name, X, use_optimal_threshold=True):
        """使用指定模型预测"""
        if model_name not in self.fitted_models:
            raise ValueError(f"模型 {model_name} 未训练")
        
        model = self.fitted_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            y_pred_prob = model.predict_proba(X)[:, 1]
        else:
            y_pred = model.predict(X)
            y_pred_prob = y_pred.astype(float)
        
        threshold = self.optimal_thresholds.get(model_name, 0.5) if use_optimal_threshold else 0.5
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        return y_pred, y_pred_prob

# ==================== 可视化函数 ====================
def plot_roc_curves_simple(models_results, save_path, title="ROC Curves"):
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_pred_prob = results['y_pred_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auroc = roc_auc_score(y_true, y_pred_prob)
        
        # 获取置信区间信息
        metrics = results.get('metrics', {})
        auroc_ci = metrics.get('AUROC', f"{auroc:.3f} ({auroc:.3f}-{auroc:.3f})")
        
        # 提取数值部分用于显示
        try:
            auroc_value = float(auroc_ci.split('(')[0].strip())
            auroc_ci_range = auroc_ci.split('(')[1].replace(')', '').strip()
            display_text = f'{model_name}\nAUROC = {auroc_value:.3f} (95% CI: {auroc_ci_range})'
        except:
            display_text = f'{model_name} (AUROC = {auroc:.3f})'
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=display_text,
                alpha=0.8)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_pr_curves_simple(models_results, save_path, title="Precision-Recall Curves"):
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_pred_prob = results['y_pred_prob']
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        auprc = auc(recall, precision)
        
        # 获取置信区间信息
        metrics = results.get('metrics', {})
        auprc_ci = metrics.get('AUPRC', f"{auprc:.3f} ({auprc:.3f}-{auprc:.3f})")
        
        # 提取数值部分用于显示
        try:
            auprc_value = float(auprc_ci.split('(')[0].strip())
            auprc_ci_range = auprc_ci.split('(')[1].replace(')', '').strip()
            display_text = f'{model_name}\nAUPRC = {auprc_value:.3f} (95% CI: {auprc_ci_range})'
        except:
            display_text = f'{model_name} (AUPRC = {auprc:.3f})'
        
        plt.plot(recall, precision, color=colors[idx], lw=2,
                label=display_text,
                alpha=0.8)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
    plt.ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_model_comparison_bar(metrics_df, save_path, title="Model Performance Comparison"):
    """绘制模型性能比较柱状图"""
    plt.figure(figsize=(14, 8))
    
    models = metrics_df.index.tolist()
    metrics = ['AUROC_value', 'AUPRC_value', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_Score']
    metric_names = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 Score']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        if i < len(axes):
            # 提取数值
            values = []
            ci_strings = []
            for model in models:
                if metric in ['AUROC_value', 'AUPRC_value']:
                    # 从字符串中提取数值和置信区间
                    if metric == 'AUROC_value':
                        ci_str = metrics_df.loc[model, 'AUROC'] if 'AUROC' in metrics_df.columns else 'N/A'
                    else:
                        ci_str = metrics_df.loc[model, 'AUPRC'] if 'AUPRC' in metrics_df.columns else 'N/A'
                    
                    try:
                        val = float(ci_str.split('(')[0].strip())
                        ci_range = ci_str.split('(')[1].replace(')', '').strip()
                        values.append(val)
                        ci_strings.append(f"{val:.3f}\n({ci_range})")
                    except:
                        values.append(0)
                        ci_strings.append('N/A')
                elif metric in metrics_df.columns:
                    try:
                        val = float(metrics_df.loc[model, metric])
                        values.append(val)
                        ci_strings.append(f"{val:.3f}")
                    except:
                        values.append(0)
                        ci_strings.append('N/A')
                else:
                    values.append(0)
                    ci_strings.append('N/A')
            
            # 创建柱状图
            bars = axes[i].bar(models, values, color=plt.cm.Set1(np.linspace(0, 1, len(models))))
            axes[i].set_title(f'{metric_name}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric_name, fontsize=12)
            axes[i].set_ylim([0, 1.0])
            axes[i].tick_params(axis='x', rotation=45)
            
            # 在柱子上添加数值和置信区间
            for bar, value, ci_str in zip(bars, values, ci_strings):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           ci_str, ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_calibration_curves(models_results, save_path, title="Calibration Curves"):
    """绘制校准曲线"""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_results)))
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        y_true = results['y_true']
        y_pred_prob = results['y_pred_prob']
        
        # 计算校准曲线
        fraction_positive, mean_predicted_value = calibration_curve(
            y_true, y_pred_prob, n_bins=10, strategy='quantile'
        )
        
        # 绘制校准曲线
        plt.plot(mean_predicted_value, fraction_positive, 
                marker='o', linewidth=2, color=colors[idx],
                label=model_name, markersize=8)
    
    # 绘制完美校准线
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# ==================== 保存最佳模型的函数 ====================
def save_best_model_for_deployment(best_model_name, all_results, external_results, 
                                 traditional_ml_models, preprocessor, config):
    """
    保存最佳模型用于部署（包含scaler.pkl）
    """
    print(f"\n{'='*60}")
    print(f"保存最佳模型: {best_model_name}")
    print(f"{'='*60}")
    
    # 创建保存目录
    os.makedirs(config.best_model_save_dir, exist_ok=True)
    
    # 根据模型类型保存不同的文件
    if best_model_name in traditional_ml_models.fitted_models:
        # 传统机器学习模型
        print(f"保存传统机器学习模型: {best_model_name}")
        model = traditional_ml_models.fitted_models[best_model_name]
        
        # 保存模型
        model_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)
        print(f"  模型已保存到: {model_path}")
        
        # 保存预处理器
        preprocessor_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"  预处理器已保存到: {preprocessor_path}")
        
        # 保存最优阈值
        optimal_threshold = traditional_ml_models.optimal_thresholds.get(best_model_name, 0.5)
        threshold_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_threshold.pkl")
        joblib.dump(optimal_threshold, threshold_path)
        print(f"  最优阈值已保存到: {threshold_path}")
        
        # 单独保存标准化器
        scaler = preprocessor.scaler
        if scaler is not None:
            scaler_path = os.path.join(config.best_model_save_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"  标准化器已保存到: {scaler_path}")
        else:
            print("  警告: 预处理器中没有标准化器")
        
        # 保存模型信息
        model_info = {
            'model_type': 'traditional_ml',
            'model_name': best_model_name,
            'model_class': model.__class__.__name__,
            'feature_names': config.feature_cols,
            'sequence_length': config.sequence_length,
            'optimal_threshold': float(optimal_threshold),
            'saved_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'preprocessor_path': preprocessor_path,
            'threshold_path': threshold_path,
            'scaler_path': os.path.join(config.best_model_save_dir, "scaler.pkl") if scaler is not None else None
        }
        
    elif best_model_name in all_results:
        # 深度学习模型
        result = all_results[best_model_name]
        model_info = {
            'model_type': 'deep_learning',
            'model_name': best_model_name,
            'feature_names': config.feature_cols,
            'sequence_length': config.sequence_length,
            'input_dim': config.input_dim,
            'saved_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 检查模型类型并保存
        if 'model' in result:
            model = result['model']
            
            # 保存PyTorch模型
            if config.save_pytorch_model:
                if hasattr(model, 'model'):  # 单独的LSTM或GRU模型
                    pytorch_model = model.model
                    model_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_pytorch.pth")
                    torch.save(pytorch_model.state_dict(), model_path)
                    print(f"  PyTorch模型已保存到: {model_path}")
                    model_info['pytorch_model_path'] = model_path
                    model_info['model_architecture'] = pytorch_model.__class__.__name__
                    
                    # 保存最优阈值
                    if hasattr(model, 'optimal_threshold'):
                        threshold_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_threshold.pkl")
                        joblib.dump(model.optimal_threshold, threshold_path)
                        print(f"  最优阈值已保存到: {threshold_path}")
                        model_info['optimal_threshold'] = float(model.optimal_threshold)
        
        # 保存预处理器
        preprocessor_path = os.path.join(config.best_model_save_dir, f"{best_model_name.replace(' ', '_')}_preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"  预处理器已保存到: {preprocessor_path}")
        model_info['preprocessor_path'] = preprocessor_path
        
        # 单独保存标准化器
        scaler = preprocessor.scaler
        if scaler is not None:
            scaler_path = os.path.join(config.best_model_save_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"  标准化器已保存到: {scaler_path}")
            model_info['scaler_path'] = scaler_path
        else:
            print("  警告: 预处理器中没有标准化器")
    
    # 保存性能指标
    metrics_path = os.path.join(config.best_model_save_dir, "performance_metrics.json")
    performance_metrics = {}
    
    if best_model_name in all_results:
        result = all_results[best_model_name]
        if 'metrics' in result:
            # 提取数值型指标
            for key, value in result['metrics'].items():
                if key not in ['Model']:
                    performance_metrics[key] = value
    
    # 添加外部验证指标
    if best_model_name in external_results:
        external_result = external_results[best_model_name]
        if 'metrics' in external_result:
            for key, value in external_result['metrics'].items():
                if key not in ['Model']:
                    performance_metrics[f'external_{key}'] = value
    
    # 保存性能指标
    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4, ensure_ascii=False)
    print(f"  性能指标已保存到: {metrics_path}")
    
    # 保存模型信息
    model_info_path = os.path.join(config.best_model_save_dir, "model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False, default=str)
    print(f"  模型信息已保存到: {model_info_path}")
    
    # 保存配置
    config_info = {
        'feature_columns': config.feature_cols,
        'sequence_length': config.sequence_length,
        'label_column': config.label_col,
        'input_dim': config.input_dim,
        'model_name': best_model_name,
        'saved_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    config_path = os.path.join(config.best_model_save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=4, ensure_ascii=False, default=str)
    print(f"  配置信息已保存到: {config_path}")
    
    # 保存示例数据用于测试（重置随机种子确保可复现）
    np.random.seed(42)
    example_data = {
        'feature_names': config.feature_cols,
        'example_sequence': np.random.randn(config.sequence_length, len(config.feature_cols)).tolist(),
        'label_mapping': {'Gram Positive': 1, 'Gram Negative': 0}
    }
    example_path = os.path.join(config.best_model_save_dir, "example_data.json")
    with open(example_path, 'w') as f:
        json.dump(example_data, f, indent=4, ensure_ascii=False, default=str)
    print(f"  示例数据已保存到: {example_path}")
    
    print(f"\n最佳模型已保存到目录: {config.best_model_save_dir}")
    
    return model_info

# ==================== 数据加载函数 ====================
def load_and_preprocess_data():
    """加载和预处理数据"""
    print("加载数据...")
    
    # 加载数据
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    external_df = pd.read_csv(config.external_path)
    
    print(f"训练集: {len(train_df['subject_id'].unique())} 例患者")
    print(f"测试集: {len(test_df['subject_id'].unique())} 例患者")
    print(f"外部验证集: {len(external_df['subject_id'].unique())} 例患者")
    
    # 转换标签
    label_mapping = {'Gram Positive': 1, 'Gram Negative': 0}
    for df in [train_df, test_df, external_df]:
        df[config.label_col] = df[config.label_col].map(label_mapping)
    
    # 检查数据分布
    print("\n数据分布:")
    for name, df in [('训练集', train_df), ('测试集', test_df), ('外部验证集', external_df)]:
        unique_patients = df.drop_duplicates('subject_id')
        pos_count = unique_patients[config.label_col].sum()
        total_count = len(unique_patients)
        pos_ratio = pos_count / total_count if total_count > 0 else 0
        print(f"{name}: {pos_ratio:.1%} 阳性 ({pos_count}/{total_count})")
    
    return train_df, test_df, external_df

# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("革兰阳性菌和革兰阴性菌脓毒症患者血流感染的早期鉴别模型（修复数据泄露版）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用设备: {config.device}")
    print(f"交叉验证: {config.n_folds}折交叉验证")
    print("\n类别不平衡处理策略:")
    print("  - 传统ML: class_weight='balanced' + 动态scale_pos_weight")
    print("  - 深度学习: BCEWithLogitsLoss with pos_weight = n_negative/n_positive")
    print("  - 阈值校准: 在验证集上优化F1-score")
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.best_model_save_dir, exist_ok=True)
    print(f"\n结果将保存到: {config.save_dir}")
    print(f"最佳模型将保存到: {config.best_model_save_dir}")
    
    # 1. 加载数据
    print("\n" + "="*60)
    print("1. 加载和预处理数据")
    print("="*60)
    
    train_df, test_df, external_df = load_and_preprocess_data()
    
    # 2. 划分训练集和验证集（患者级别）
    print("\n" + "="*60)
    print("2. 划分训练集和验证集")
    print("="*60)
    
    train_patient_ids = train_df['subject_id'].unique()
    patient_labels = []
    for pid in train_patient_ids:
        label = train_df[train_df['subject_id'] == pid][config.label_col].iloc[0]
        patient_labels.append(label)
    patient_labels_array = np.array(patient_labels)
    
    # 打印类别分布
    pos_count = sum(patient_labels_array)
    neg_count = len(patient_labels_array) - pos_count
    print(f"训练集总患者: {len(patient_labels_array)}")
    print(f"  阳性 (Gram Positive): {pos_count} ({pos_count/len(patient_labels_array)*100:.1f}%)")
    print(f"  阴性 (Gram Negative): {neg_count} ({neg_count/len(patient_labels_array)*100:.1f}%)")
    print(f"  不平衡比例: {max(pos_count, neg_count)/min(pos_count, neg_count):.1f}:1")
    
    # 使用分层划分
    train_train_pids, train_val_pids = train_test_split(
        train_patient_ids, 
        test_size=config.val_ratio, 
        random_state=42,
        stratify=patient_labels_array
    )
    
    # 提取划分后的数据
    train_train_df = train_df[train_df['subject_id'].isin(train_train_pids)]
    train_val_df = train_df[train_df['subject_id'].isin(train_val_pids)]
    
    print(f"\n训练子集: {len(train_train_pids)} 例患者")
    print(f"验证子集: {len(train_val_pids)} 例患者")
    
    # 3. 训练传统机器学习模型（使用安全预处理器）
    print(f"\n" + "="*60)
    print(f"3. 训练传统机器学习模型（{config.n_folds}折交叉验证，安全预处理器）")
    print("="*60)
    
    traditional_ml = EnhancedTraditionalMLModels()
    ml_results = traditional_ml.train_and_evaluate_with_cv(train_df, test_df, n_folds=config.n_folds)
    
    # 4. 为深度学习模型创建安全预处理器
    print("\n" + "="*60)
    print("4. 为深度学习模型准备数据（使用安全预处理器）")
    print("="*60)
    
    # 在训练子集上拟合安全预处理器
    print("为深度学习模型创建安全预处理器...")
    dl_preprocessor = SecureDataPreprocessor(config.feature_cols, config.sequence_length)
    dl_preprocessor.fit(train_train_df)
    
    # 使用安全预处理器创建数据集
    train_train_data = create_dataset_dict(
        train_train_df, config.feature_cols, config.label_col, preprocessor=dl_preprocessor
    )
    train_train_dataset = SepsisTimeSeriesDataset(train_train_data)
    
    train_val_data = create_dataset_dict(
        train_val_df, config.feature_cols, config.label_col, preprocessor=dl_preprocessor
    )
    train_val_dataset = SepsisTimeSeriesDataset(train_val_data)
    
    # 使用相同的预处理器处理测试集和外部验证集
    test_patients_data = create_dataset_dict(
        test_df, config.feature_cols, config.label_col, preprocessor=dl_preprocessor
    )
    test_dataset = SepsisTimeSeriesDataset(test_patients_data)
    
    external_patients_data = create_dataset_dict(
        external_df, config.feature_cols, config.label_col, preprocessor=dl_preprocessor
    )
    external_dataset = SepsisTimeSeriesDataset(external_patients_data)
    
    # 保存安全预处理器（使用 joblib 确保兼容性）
    joblib.dump(dl_preprocessor, f"{config.save_dir}/secure_preprocessor.pkl")
    
    # 5. 训练深度学习模型
    print("\n" + "-"*40)
    print("训练单独的LSTM和GRU模型（使用安全预处理数据）")
    print("-"*40)
    
    # 单独的LSTM模型
    print("\n训练单独的LSTM模型...")
    standalone_lstm_model = StandaloneLSTMModel(
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_num_layers=config.lstm_num_layers,
        dropout_rate=config.dropout_rate,
        bidirectional=config.lstm_bidirectional
    )
    standalone_lstm_model.train(train_train_dataset, train_val_dataset)
    
    # 在测试集上评估单独的LSTM模型（使用最优阈值）
    y_test_lstm, y_pred_lstm, y_prob_lstm = standalone_lstm_model.predict(test_dataset, use_optimal_threshold=True)
    lstm_metrics = calculate_metrics_with_ci_fixed(y_test_lstm, y_pred_lstm, y_prob_lstm, "LSTM", standalone_lstm_model.optimal_threshold)
    lstm_metrics['CV_AUROC_range'] = 'N/A'
    lstm_metrics['Optimal_Threshold'] = f"{standalone_lstm_model.optimal_threshold:.3f}"
    
    lstm_results = {
        'y_true': y_test_lstm,
        'y_pred': y_pred_lstm,
        'y_pred_prob': y_prob_lstm,
        'metrics': lstm_metrics,
        'model': standalone_lstm_model,
        'optimal_threshold': standalone_lstm_model.optimal_threshold
    }
    
    # 单独的GRU模型
    print("\n训练单独的GRU模型...")
    standalone_gru_model = StandaloneGRUModel(
        gru_hidden_dim=config.gru_hidden_dim,
        gru_num_layers=config.gru_num_layers,
        dropout_rate=config.dropout_rate,
        bidirectional=config.gru_bidirectional
    )
    standalone_gru_model.train(train_train_dataset, train_val_dataset)
    
    # 在测试集上评估单独的GRU模型（使用最优阈值）
    y_test_gru, y_pred_gru, y_prob_gru = standalone_gru_model.predict(test_dataset, use_optimal_threshold=True)
    gru_metrics = calculate_metrics_with_ci_fixed(y_test_gru, y_pred_gru, y_prob_gru, "GRU", standalone_gru_model.optimal_threshold)
    gru_metrics['CV_AUROC_range'] = 'N/A'
    gru_metrics['Optimal_Threshold'] = f"{standalone_gru_model.optimal_threshold:.3f}"
    
    gru_results = {
        'y_true': y_test_gru,
        'y_pred': y_pred_gru,
        'y_pred_prob': y_prob_gru,
        'metrics': gru_metrics,
        'model': standalone_gru_model,
        'optimal_threshold': standalone_gru_model.optimal_threshold
    }
    
    # 6. 合并所有结果
    print("\n" + "="*60)
    print("6. 模型性能比较（测试集）")
    print("="*60)
    
    # 收集所有结果
    all_results = {
        'LSTM': lstm_results,
        'GRU': gru_results
    }
    
    for name, result in ml_results.items():
        all_results[name] = result
    
    # 创建性能比较表格
    metrics_list = []
    for model_name, results in all_results.items():
        metrics = results['metrics'].copy()
        metrics['Model'] = model_name
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('Model', inplace=True)
    
    # 保存结果
    metrics_df.to_csv(f"{config.save_dir}/test_set_performance_fixed_leakage.csv", index=True)
    
    print("\n测试集性能指标:")
    print("-" * 140)
    display_cols = ['AUROC', 'AUPRC', 'CV_AUROC_mean', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_Score', 'Brier_Score', 'Optimal_Threshold']
    available_cols = [col for col in display_cols if col in metrics_df.columns]
    print(metrics_df[available_cols].to_string())
    
    # 7. 外部验证
    print("\n" + "="*60)
    print("7. 外部验证")
    print("="*60)
    
    external_results = {}
    
    # 传统机器学习模型的外部验证
    print("传统机器学习模型的外部验证...")
    for name, model in traditional_ml.fitted_models.items():
        print(f"  验证 {name}...")
        try:
            # 使用该模型对应的预处理器
            preprocessor = traditional_ml.preprocessors.get(name)
            if preprocessor is None:
                print(f"    警告: {name}没有对应的预处理器，跳过外部验证")
                continue
            
            # 准备外部验证特征（使用训练集预处理器）
            X_external = []
            y_external = []
            
            external_patient_ids = external_df['subject_id'].unique()
            for pid in external_patient_ids:
                patient_df = external_df[external_df['subject_id'] == pid]
                features = preprocessor.extract_features_for_ml(patient_df)
                label = patient_df[config.label_col].iloc[0]
                X_external.append(features)
                y_external.append(label)
            
            X_external = np.array(X_external)
            y_external = np.array(y_external)
            
            # 预测（使用最优阈值）
            if hasattr(model, 'predict_proba'):
                y_pred_prob = model.predict_proba(X_external)[:, 1]
            else:
                y_pred = model.predict(X_external)
                y_pred_prob = y_pred.astype(float)
            
            optimal_threshold = traditional_ml.optimal_thresholds.get(name, 0.5)
            y_pred = (y_pred_prob >= optimal_threshold).astype(int)
            
            metrics = calculate_metrics_with_ci_fixed(y_external, y_pred, y_pred_prob, f"{name} (External)", optimal_threshold)
            external_results[name] = {
                'y_true': y_external,
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'metrics': metrics,
                'optimal_threshold': optimal_threshold
            }
            
            print(f"    外部验证AUROC: {metrics['AUROC']}")
            print(f"    外部验证AUPRC: {metrics['AUPRC']}")
            print(f"    最优阈值: {optimal_threshold:.3f}")
        except Exception as e:
            print(f"    验证{name}时出错: {str(e)}")
    
    # LSTM外部验证
    print("\n单独LSTM模型的外部验证...")
    try:
        y_ext_true, y_ext_pred, y_ext_prob = standalone_lstm_model.predict(external_dataset, use_optimal_threshold=True)
        
        lstm_ext_metrics = calculate_metrics_with_ci_fixed(y_ext_true, y_ext_pred, y_ext_prob, "LSTM (External)", standalone_lstm_model.optimal_threshold)
        external_results['LSTM'] = {
            'y_true': y_ext_true,
            'y_pred': y_ext_pred,
            'y_pred_prob': y_ext_prob,
            'metrics': lstm_ext_metrics,
            'optimal_threshold': standalone_lstm_model.optimal_threshold
        }
        print(f"    外部验证AUROC: {lstm_ext_metrics['AUROC']}")
        print(f"    外部验证AUPRC: {lstm_ext_metrics['AUPRC']}")
    except Exception as e:
        print(f"    验证LSTM时出错: {str(e)}")
    
    # GRU外部验证
    print("\n单独GRU模型的外部验证...")
    try:
        y_ext_true, y_ext_pred, y_ext_prob = standalone_gru_model.predict(external_dataset, use_optimal_threshold=True)
        
        gru_ext_metrics = calculate_metrics_with_ci_fixed(y_ext_true, y_ext_pred, y_ext_prob, "GRU (External)", standalone_gru_model.optimal_threshold)
        external_results['GRU'] = {
            'y_true': y_ext_true,
            'y_pred': y_ext_pred,
            'y_pred_prob': y_ext_prob,
            'metrics': gru_ext_metrics,
            'optimal_threshold': standalone_gru_model.optimal_threshold
        }
        print(f"    外部验证AUROC: {gru_ext_metrics['AUROC']}")
        print(f"    外部验证AUPRC: {gru_ext_metrics['AUPRC']}")
    except Exception as e:
        print(f"    验证GRU时出错: {str(e)}")
    
    # 创建外部验证性能表格
    external_metrics_list = []
    for model_name, results in external_results.items():
        metrics = results['metrics'].copy()
        metrics['Model'] = model_name
        external_metrics_list.append(metrics)
    
    external_metrics_df = pd.DataFrame(external_metrics_list)
    external_metrics_df.set_index('Model', inplace=True)
    
    # 保存外部验证结果
    external_metrics_df.to_csv(f"{config.save_dir}/external_validation_performance_fixed_leakage.csv", index=True)
    
    print("\n外部验证性能指标:")
    print("-" * 120)
    display_cols = ['AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1_Score', 'Brier_Score', 'Optimal_Threshold']
    available_cols = [col for col in display_cols if col in external_metrics_df.columns]
    print(external_metrics_df[available_cols].to_string())
    
    # 8. 可视化
    print("\n" + "="*60)
    print("8. 生成可视化图表")
    print("="*60)
    
    # 准备测试集可视化数据
    test_visualization_data = {}
    for model_name, results in all_results.items():
        test_visualization_data[model_name] = {
            'y_true': results['y_true'],
            'y_pred_prob': results['y_pred_prob'],
            'metrics': results['metrics']
        }
    
    # 准备外部验证集可视化数据
    external_visualization_data = {}
    for model_name, results in external_results.items():
        external_visualization_data[model_name] = {
            'y_true': results['y_true'],
            'y_pred_prob': results['y_pred_prob'],
            'metrics': results['metrics']
        }
    
    # 生成测试集图表
    print("生成测试集图表...")
    plot_roc_curves_simple(test_visualization_data, f"{config.save_dir}/test_roc_curves_fixed_leakage.png", "Test Set ROC Curves")
    plot_pr_curves_simple(test_visualization_data, f"{config.save_dir}/test_pr_curves_fixed_leakage.png", "Test Set Precision-Recall Curves")
    plot_model_comparison_bar(metrics_df, f"{config.save_dir}/test_model_comparison_fixed_leakage.png", "Test Set Model Performance Comparison")
    plot_calibration_curves(test_visualization_data, f"{config.save_dir}/test_calibration_curves_fixed_leakage.png", "Test Set Calibration Curves")
    
    # 生成外部验证集图表
    print("生成外部验证集图表...")
    plot_roc_curves_simple(external_visualization_data, f"{config.save_dir}/external_roc_curves_fixed_leakage.png", "External Validation ROC Curves")
    plot_pr_curves_simple(external_visualization_data, f"{config.save_dir}/external_pr_curves_fixed_leakage.png", "External Validation Precision-Recall Curves")
    plot_model_comparison_bar(external_metrics_df, f"{config.save_dir}/external_model_comparison_fixed_leakage.png", "External Validation Model Performance Comparison")
    plot_calibration_curves(external_visualization_data, f"{config.save_dir}/external_calibration_curves_fixed_leakage.png", "External Validation Calibration Curves")
    
    # 9. 保存最佳模型
    # 选择最佳模型 - 优先选择LightGBM
    best_model_name = 'LightGBM'
    
    # 检查LightGBM模型是否存在
    if best_model_name in metrics_df.index:
        best_model_auroc_str = metrics_df.loc[best_model_name, 'AUROC']
        try:
            best_model_auroc = float(best_model_auroc_str.split('(')[0].strip())
        except:
            best_model_auroc = 0.5
    else:
        # 如果LightGBM不存在，选择AUROC最高的模型
        best_model_name = None
        best_model_auroc = 0
        for model_name, row in metrics_df.iterrows():
            if 'AUROC' in row:
                auroc_str = row['AUROC']
                try:
                    auroc_val = float(auroc_str.split('(')[0].strip())
                    if auroc_val > best_model_auroc:
                        best_model_auroc = auroc_val
                        best_model_name = model_name
                except:
                    continue
    
    # 保存最佳模型
    if best_model_name:
        print(f"\n{'='*60}")
        print(f"最佳模型: {best_model_name} (测试集AUROC: {best_model_auroc:.3f})")
        print(f"{'='*60}")
        
        # 获取对应的预处理器
        if best_model_name in traditional_ml.preprocessors:
            best_preprocessor = traditional_ml.preprocessors[best_model_name]
        else:
            best_preprocessor = dl_preprocessor
        
        # 保存最佳模型用于部署
        model_info = save_best_model_for_deployment(
            best_model_name=best_model_name,
            all_results=all_results,
            external_results=external_results,
            traditional_ml_models=traditional_ml,
            preprocessor=best_preprocessor,
            config=config
        )
    
    # 10. 生成详细报告
    print("\n" + "="*60)
    print("9. 生成详细报告")
    print("="*60)
    
    # 生成最终报告
    final_report = {
        'experiment_info': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device_used': str(config.device),
            'experiment_type': '革兰阳性菌和革兰阴性菌脓毒症患者血流感染的早期鉴别模型（修复数据泄露版）',
            'validation_strategy': f'{config.n_folds}折交叉验证（安全预处理器）+ 10%内部验证集早停',
            'data_leakage_fixes': {
                'secure_preprocessor': '使用安全预处理器确保所有统计量来自训练集',
                'feature_engineering_isolation': '特征工程完全隔离，验证集/测试集不使用自身信息',
                'imputation_isolation': '缺失值填充使用训练集统计量',
                'cross_validation_isolation': '交叉验证中每个fold使用独立的预处理器'
            },
            'class_imbalance_handling': {
                'strategy': '类别加权 + 阈值校准',
                'traditional_ml': 'class_weight=\'balanced\' + 动态scale_pos_weight',
                'deep_learning': 'BCEWithLogitsLoss with pos_weight = n_negative/n_positive',
                'threshold_calibration': '在验证集上优化F1-score',
                'evaluation_metrics': 'AUROC, AUPRC, Brier Score, Calibration Curves'
            }
        },
        'data_statistics': {
            'train_patients': len(train_df['subject_id'].unique()),
            'train_train_patients': len(train_train_pids),
            'train_val_patients': len(train_val_pids),
            'test_patients': len(test_df['subject_id'].unique()),
            'external_patients': len(external_df['subject_id'].unique()),
            'features_used': config.feature_cols,
            'train_pos_ratio': train_df.drop_duplicates('subject_id')[config.label_col].mean(),
            'train_train_pos_ratio': train_train_df.drop_duplicates('subject_id')[config.label_col].mean(),
            'train_val_pos_ratio': train_val_df.drop_duplicates('subject_id')[config.label_col].mean(),
            'test_pos_ratio': test_df.drop_duplicates('subject_id')[config.label_col].mean(),
            'external_pos_ratio': external_df.drop_duplicates('subject_id')[config.label_col].mean()
        },
        'model_configurations': {
            'lstm_standalone': {
                'lstm_hidden_dim': config.lstm_hidden_dim,
                'lstm_num_layers': config.lstm_num_layers,
                'dropout_rate': config.dropout_rate,
                'bidirectional': config.lstm_bidirectional
            },
            'gru_standalone': {
                'gru_hidden_dim': config.gru_hidden_dim,
                'gru_num_layers': config.gru_num_layers,
                'dropout_rate': config.dropout_rate,
                'bidirectional': config.gru_bidirectional
            },
            'traditional_ml_models': list(traditional_ml.models.keys())
        },
        'performance_summary': {
            'best_model': best_model_name,
            'best_model_test_auroc': float(best_model_auroc) if best_model_auroc else 0,
            'test_set_performance': metrics_df.to_dict(),
            'external_validation_performance': external_metrics_df.to_dict()
        }
    }
    
    with open(f"{config.save_dir}/final_report_fixed_leakage.json", 'w') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False, default=str)
    
    # 11. 总结
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    print(f"\n所有结果已保存到: {config.save_dir}")
    
    if best_model_name:
        print(f"最佳模型已保存到: {config.best_model_save_dir}")
    
    print(f"\n数据泄露修复总结:")
    print(f"1. 创建了SecureDataPreprocessor类，统一管理所有预处理步骤")
    print(f"2. 确保所有统计量（均值、标准差、填充值）都来自训练集")
    print(f"3. 特征工程完全隔离：验证集/测试集不使用自身信息生成特征")
    print(f"4. 缺失值填充使用训练集全局统计量，而非患者自身统计量")
    print(f"5. 交叉验证中每个fold使用独立的预处理器")
    print(f"6. 移除了趋势特征（容易泄露信息）")
    print(f"7. 外部验证集特征生成完全依赖训练集预处理器")
    
    print(f"\n类别不平衡处理总结:")
    print(f"1. 传统ML模型: class_weight='balanced' + 动态scale_pos_weight")
    print(f"2. 深度学习模型: BCEWithLogitsLoss with pos_weight = n_negative/n_positive")
    print(f"3. 阈值校准: 在验证集上优化F1-score")
    print(f"4. 评估指标: 报告AUPRC（对不平衡更鲁棒）和校准曲线")
    
    print(f"\n生成的文件:")
    print(f"1. 测试集性能指标: {config.save_dir}/test_set_performance_fixed_leakage.csv")
    print(f"2. 外部验证性能指标: {config.save_dir}/external_validation_performance_fixed_leakage.csv")
    print(f"3. 测试集可视化:")
    print(f"   - ROC曲线: {config.save_dir}/test_roc_curves_fixed_leakage.png")
    print(f"   - PR曲线: {config.save_dir}/test_pr_curves_fixed_leakage.png")
    print(f"   - 模型比较图: {config.save_dir}/test_model_comparison_fixed_leakage.png")
    print(f"   - 校准曲线: {config.save_dir}/test_calibration_curves_fixed_leakage.png")
    print(f"4. 外部验证集可视化:")
    print(f"   - ROC曲线: {config.save_dir}/external_roc_curves_fixed_leakage.png")
    print(f"   - PR曲线: {config.save_dir}/external_pr_curves_fixed_leakage.png")
    print(f"   - 模型比较图: {config.save_dir}/external_model_comparison_fixed_leakage.png")
    print(f"   - 校准曲线: {config.save_dir}/external_calibration_curves_fixed_leakage.png")
    print(f"5. 详细报告: {config.save_dir}/final_report_fixed_leakage.json")
    print(f"6. 安全预处理器: {config.save_dir}/secure_preprocessor.pkl")
    
    if best_model_name:
        print(f"\n最佳模型相关文件:")
        print(f"1. 标准化器: {config.best_model_save_dir}/scaler.pkl")
        print(f"2. 预处理器: {config.best_model_save_dir}/{best_model_name.replace(' ', '_')}_preprocessor.pkl")
        print(f"3. 模型文件: {config.best_model_save_dir}/{best_model_name.replace(' ', '_')}_model.pkl")
        print(f"4. 最优阈值: {config.best_model_save_dir}/{best_model_name.replace(' ', '_')}_threshold.pkl")
        print(f"5. 模型信息: {config.best_model_save_dir}/model_info.json")
        print(f"6. 配置信息: {config.best_model_save_dir}/config.json")
        print(f"7. 性能指标: {config.best_model_save_dir}/performance_metrics.json")
        print(f"8. 示例数据: {config.best_model_save_dir}/example_data.json")
    
    print("\n感谢使用脓毒症血流感染鉴别模型（修复数据泄露版）!")

if __name__ == "__main__":
    main()
```