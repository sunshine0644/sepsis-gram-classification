# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import shap
import matplotlib
matplotlib.use('Agg')  # 解决GUI问题
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from model_predictor import predictor

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'message': 'Sepsis Prediction API',
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """单次预测接口"""
    try:
        data = request.get_json()
        
        # 提取特征
        features = data.get('features', {})
        
        # 转换为DataFrame
        df = pd.DataFrame([features])
        
        # 预测
        proba = predictor.predict_single(df)
        
        # 获取特征重要性
        importance = predictor.get_feature_importance()
        
        return jsonify({
            'success': True,
            'probability': float(proba[0]),
            'risk_level': 'high' if proba[0] > 0.5 else 'low',
            'feature_importance': importance
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        df = pd.DataFrame(samples)
        probas = predictor.predict_single(df)
        
        results = []
        for i, prob in enumerate(probas):
            results.append({
                'sample_id': i,
                'probability': float(prob),
                'risk_level': 'high' if prob > 0.5 else 'low'
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/shap/analyze', methods=['POST'])
def shap_analyze():
    """SHAP分析接口"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        # 转换为3D数组 (n_samples, 3, n_features)
        if len(samples) == 1:
            # 单样本，复制成3个时间点
            sample = samples[0]
            X_3d = np.array([[sample] * 3])
        else:
            # 多样本，假设每个样本是3个时间点
            X_3d = np.array(samples)
        
        # 计算SHAP值
        shap_values = predictor.compute_shap_values(X_3d, n_background=50)
        
        if shap_values is None:
            return jsonify({
                'success': False,
                'error': 'SHAP computation failed'
            }), 500
        
        # 生成SHAP图
        figs = {}
        
        # 1. 蜂群图
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=predictor.feature_cols,
            show=False,
            max_display=10
        )
        plt.tight_layout()
        
        # 转换为base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        figs['beeswarm'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 2. 条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values.values,
            shap_values.data,
            feature_names=predictor.feature_cols,
            show=False,
            plot_type="bar",
            max_display=10
        )
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        figs['bar'] = base64.b64encode(buf.read()).decode('utf-8')
        
        # 计算每个样本的预测概率
        probas = predictor.predict_temporal(X_3d)
        
        return jsonify({
            'success': True,
            'figures': figs,
            'shap_summary': {
                'mean_abs_shap': np.mean(np.abs(shap_values.values), axis=0).tolist(),
                'feature_names': predictor.feature_cols
            },
            'predictions': probas.tolist()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/shap/waterfall', methods=['POST'])
def shap_waterfall():
    """为特定病例生成瀑布图"""
    try:
        data = request.get_json()
        sample = data.get('sample', [])
        sample_idx = data.get('sample_idx', 0)
        
        # 转换为3D数组
        if len(sample) == 3 and isinstance(sample[0], list):
            X_3d = np.array([sample])
        else:
            X_3d = np.array([[sample] * 3])
        
        # 计算SHAP值
        shap_values = predictor.compute_shap_values(X_3d, n_background=50)
        
        if shap_values is None:
            return jsonify({
                'success': False,
                'error': 'SHAP computation failed'
            }), 500
        
        # 获取基准值
        base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.5
        
        # 为每个时间点生成瀑布图
        waterfall_figs = []
        for t in range(3):
            # 创建Explanation对象
            explanation = shap.Explanation(
                values=shap_values.values[0, t*14:(t+1)*14] if len(shap_values.values.shape) > 1 else shap_values.values[0],
                base_values=base_value,
                data=X_3d[0, t],
                feature_names=predictor.feature_cols
            )
            
            # 生成瀑布图
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.waterfall_plot(explanation, show=False, max_display=8)
            plt.title(f'Time Period {t+1} SHAP Waterfall', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            waterfall_figs.append(base64.b64encode(buf.read()).decode('utf-8'))
        
        # 预测概率
        proba = predictor.predict_temporal(X_3d)[0]
        
        return jsonify({
            'success': True,
            'waterfall_plots': waterfall_figs,
            'base_value': float(base_value),
            'probability': float(proba),
            'feature_names': predictor.feature_cols
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    try:
        importance = predictor.get_feature_importance()
        
        return jsonify({
            'success': True,
            'features': predictor.feature_cols,
            'expected_features': predictor.expected_features,
            'feature_importance': importance,
            'model_type': type(predictor.model).__name__
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)