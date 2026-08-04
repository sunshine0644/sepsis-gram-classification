"""
Publication-quality plots for Top-5 feature LightGBM model
— AUROC, AUPRC, and DCA curves —
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.style as style
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# Global style settings
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette — scientific, colorblind-friendly
C_TRAIN = '#2C5F8A'     # deep blue
C_TEST  = '#D44C3A'     # muted red
C_EXT   = '#3B8E5B'     # forest green
C_REF   = '#7F8C8D'     # grey
C_FILL_TRAIN = '#2C5F8A'
C_FILL_TEST  = '#D44C3A'
C_FILL_EXT   = '#3B8E5B'

# ══════════════════════════════════════════════════════════════
# Data loading & feature engineering (same pipeline as paper)
# ══════════════════════════════════════════════════════════════
TOP5 = ["pt", "platelet", "hemoglobin", "bicarbonate", "resp_rate"]

def load_data(path):
    df = pd.read_csv(path)
    df['label'] = (df['gram_type'] == 'Gram Positive').astype(int)
    return df

def extract_features(group, feature_cols):
    feats = group[feature_cols].values
    if len(feats) == 0:
        return None
    col_med = np.nanmedian(feats, axis=0)
    for j in range(feats.shape[1]):
        mask = np.isnan(feats[:, j])
        feats[mask, j] = col_med[j] if not np.isnan(col_med[j]) else 0
    mean_f  = np.mean(feats, axis=0)
    std_f   = np.std(feats, axis=0) if len(feats) > 1 else np.zeros(feats.shape[1])
    max_f   = np.max(feats, axis=0)
    min_f   = np.min(feats, axis=0)
    median_f = np.median(feats, axis=0)
    return np.concatenate([mean_f, std_f, max_f, min_f, median_f])

def build_dataset(df, feature_cols):
    X_list, y_list = [], []
    for sid, group in df.groupby('subject_id'):
        feats = extract_features(group, feature_cols)
        if feats is not None:
            X_list.append(feats)
            y_list.append(group['label'].iloc[0])
    return np.array(X_list), np.array(y_list)

# Load
train = load_data('/Users/lizeqi/Desktop/机器学习所有数据/data analysis/train_data.csv')
test  = load_data('/Users/lizeqi/Desktop/机器学习所有数据/data analysis/test_data.csv')
ext   = load_data('/Users/lizeqi/Desktop/机器学习所有数据/data analysis/external validation.csv')

X_train, y_train = build_dataset(train, TOP5)
X_test,  y_test  = build_dataset(test,  TOP5)
X_ext,   y_ext   = build_dataset(ext,   TOP5)

# Impute & scale
for j in range(X_train.shape[1]):
    med = np.nanmedian(X_train[:, j])
    X_train[:, j] = np.nan_to_num(X_train[:, j], nan=med)
    X_test[:, j]  = np.nan_to_num(X_test[:, j],  nan=med)
    X_ext[:, j]   = np.nan_to_num(X_ext[:, j],   nan=med)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_ext_s   = scaler.transform(X_ext)

# Train LightGBM
spw = (y_train == 0).sum() / (y_train == 1).sum()
model = LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=spw,
    random_state=42, n_jobs=-1, verbose=-1
)
model.fit(X_train_s, y_train)

# Probabilities
proba_train = model.predict_proba(X_train_s)[:, 1]
proba_test  = model.predict_proba(X_test_s)[:, 1]
proba_ext   = model.predict_proba(X_ext_s)[:, 1]

# ══════════════════════════════════════════════════════════════
# FIGURE 1 — AUROC Curves
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.8))

datasets = [
    (proba_train, y_train, 'Training set', C_TRAIN),
    (proba_test,  y_test,  'Test set',     C_TEST),
    (proba_ext,   y_ext,   'External validation', C_EXT),
]

for proba, y_true, label, color in datasets:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)

    # Bootstrap CI
    np.random.seed(42)
    boots = [roc_auc_score(y_true[np.random.choice(len(y_true), len(y_true), replace=True)],
                           proba[np.random.choice(len(y_true), len(y_true), replace=True)])
             for _ in range(1000)]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    ax.plot(fpr, tpr, color=color, lw=2.2, alpha=0.95,
            label=f'{label}  (AUROC = {auc_val:.3f}, 95% CI: {ci_low:.3f}–{ci_high:.3f})')

# Diagonal
ax.plot([0, 1], [0, 1], '--', color=C_REF, lw=1.0, alpha=0.55)

ax.set_xlabel('1 – Specificity (False Positive Rate)', labelpad=10)
ax.set_ylabel('Sensitivity (True Positive Rate)', labelpad=10)
ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold', pad=14)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.set_yticks(np.arange(0, 1.1, 0.2))

legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                   framealpha=0.92, edgecolor='#CCCCCC')
legend.get_frame().set_linewidth(0.5)

ax.grid(True, alpha=0.18, linestyle='-', linewidth=0.5)
ax.tick_params(length=4, width=1.0)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.1)

plt.tight_layout()
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure1_AUROC.png', dpi=400)
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure1_AUROC.pdf')
plt.close()
print("✓  Figure 1 — AUROC saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 2 — AUPRC Curves
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.8))

for proba, y_true, label, color in datasets:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    auc_val = average_precision_score(y_true, proba)

    # Bootstrap CI
    np.random.seed(42)
    boots = []
    for _ in range(1000):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        try:
            boots.append(average_precision_score(y_true[idx], proba[idx]))
        except:
            pass
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    ax.plot(recall, precision, color=color, lw=2.2, alpha=0.95,
            label=f'{label}  (AUPRC = {auc_val:.3f}, 95% CI: {ci_low:.3f}–{ci_high:.3f})')

# No-skill line (positive class prevalence)
no_skill = y_test.mean()
ax.axhline(no_skill, color=C_REF, linestyle='--', lw=1.0, alpha=0.55,
           label=f'No-skill classifier ({no_skill:.2f})')

ax.set_xlabel('Recall (Sensitivity)', labelpad=10)
ax.set_ylabel('Precision (Positive Predictive Value)', labelpad=10)
ax.set_title('Precision–Recall (PR) Curves', fontweight='bold', pad=14)

ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.set_yticks(np.arange(0, 1.1, 0.2))

legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                   framealpha=0.92, edgecolor='#CCCCCC')
legend.get_frame().set_linewidth(0.5)

ax.grid(True, alpha=0.18, linestyle='-', linewidth=0.5)
ax.tick_params(length=4, width=1.0)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.1)

plt.tight_layout()
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure2_AUPRC.png', dpi=400)
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure2_AUPRC.pdf')
plt.close()
print("✓  Figure 2 — AUPRC saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Decision Curve Analysis (DCA)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6.8))

thresholds = np.linspace(0.01, 0.80, 200)

for proba, y_true, label, color in [
    (proba_test, y_test, 'Test set', C_TEST),
    (proba_ext,  y_ext,  'External validation', C_EXT),
]:
    net_benefit = []
    for pt in thresholds:
        y_pred_t = (proba >= pt).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_true == 1))
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        n  = len(y_true)
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)

    ax.plot(thresholds, net_benefit, color=color, lw=2.2, alpha=0.95, label=label)

# Treat-all reference
avg_prev = (y_test.mean() + y_ext.mean()) / 2
nb_treat_all = avg_prev - (1 - avg_prev) * thresholds / (1 - thresholds)
ax.plot(thresholds, nb_treat_all, color=C_REF, lw=1.2, linestyle='--', alpha=0.7,
        label='Treat all')

# Treat-none reference
ax.axhline(0, color=C_REF, lw=1.2, linestyle=':', alpha=0.5, label='Treat none')

ax.set_xlabel('Threshold Probability', labelpad=10)
ax.set_ylabel('Net Benefit', labelpad=10)
ax.set_title('Decision Curve Analysis (DCA)', fontweight='bold', pad=14)

ax.set_xlim(0, 0.80)
ax.set_xticks(np.arange(0, 0.85, 0.10))

legend = ax.legend(loc='lower right', frameon=True, fancybox=True,
                   framealpha=0.92, edgecolor='#CCCCCC')
legend.get_frame().set_linewidth(0.5)

ax.grid(True, alpha=0.18, linestyle='-', linewidth=0.5)
ax.tick_params(length=4, width=1.0)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(1.1)

ax.set_ylim(bottom=-0.05)

plt.tight_layout()
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure3_DCA.png', dpi=400)
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure3_DCA.pdf')
plt.close()
print("✓  Figure 3 — DCA saved")

# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Combined panel (3-in-1 for manuscript)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

# --- Panel A: AUROC ---
ax = axes[0]
for proba, y_true, label, color in datasets:
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_val = roc_auc_score(y_true, proba)
    ax.plot(fpr, tpr, color=color, lw=2.0, alpha=0.95,
            label=f'{label}\nAUROC={auc_val:.3f}')
ax.plot([0, 1], [0, 1], '--', color=C_REF, lw=0.8, alpha=0.5)
ax.set_xlabel('1 – Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('A  —  ROC Curves', fontweight='bold', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=8.5, framealpha=0.9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.15, lw=0.5)
ax.tick_params(length=3, width=0.8)

# --- Panel B: AUPRC ---
ax = axes[1]
for proba, y_true, label, color in datasets:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    auc_val = average_precision_score(y_true, proba)
    ax.plot(recall, precision, color=color, lw=2.0, alpha=0.95,
            label=f'{label}\nAUPRC={auc_val:.3f}')
ax.axhline(y_test.mean(), color=C_REF, linestyle='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('B  —  PR Curves', fontweight='bold', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=8.5, framealpha=0.9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.15, lw=0.5)
ax.tick_params(length=3, width=0.8)

# --- Panel C: DCA ---
ax = axes[2]
for proba, y_true, label, color in [
    (proba_test, y_test, 'Test set', C_TEST),
    (proba_ext,  y_ext,  'External validation', C_EXT),
]:
    nb_list = []
    for pt in thresholds:
        y_pred_t = (proba >= pt).astype(int)
        tp = np.sum((y_pred_t == 1) & (y_true == 1))
        fp = np.sum((y_pred_t == 1) & (y_true == 0))
        n  = len(y_true)
        nb_list.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    ax.plot(thresholds, nb_list, color=color, lw=2.0, alpha=0.95, label=label)

avg_prev = (y_test.mean() + y_ext.mean()) / 2
ax.plot(thresholds, avg_prev - (1 - avg_prev) * thresholds / (1 - thresholds),
        color=C_REF, lw=1.0, linestyle='--', alpha=0.55, label='Treat all')
ax.axhline(0, color=C_REF, lw=1.0, linestyle=':', alpha=0.4, label='Treat none')

ax.set_xlabel('Threshold Probability')
ax.set_ylabel('Net Benefit')
ax.set_title('C  —  Decision Curve Analysis', fontweight='bold', loc='left')
ax.legend(loc='lower right', frameon=True, fontsize=8.5, framealpha=0.9)
ax.set_xlim(0, 0.80)
ax.grid(True, alpha=0.15, lw=0.5)
ax.tick_params(length=3, width=0.8)

plt.tight_layout(pad=2.5)
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure4_Combined_Panel.png', dpi=400)
fig.savefig('/Users/lizeqi/Desktop/Top5_特征分析结果/Figure4_Combined_Panel.pdf')
plt.close()
print("✓  Figure 4 — Combined panel saved")

print("\n" + "="*50)
print("All figures saved to:")
print("  /Users/lizeqi/Desktop/Top5_特征分析结果/")
print("="*50)
