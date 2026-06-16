"""Clean comparison: Best model(s) vs external baselines. All 5 metrics."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, json, csv

OUT = os.path.join(os.path.dirname(__file__), "results", "plots", "experiment_summary")
os.makedirs(OUT, exist_ok=True)

C_BEST = '#E53935'; C_OURS = '#1A237E'; C_EXT = '#78909C'
M_COLORS = {'f1': '#E53935', 'precision': '#1E88E5', 'recall': '#43A047',
            'auc': '#FB8C00', 'aupr': '#7B1FA2'}
METRICS = ['f1', 'precision', 'recall', 'auc', 'aupr']
METRIC_LABELS = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall', 'auc': 'AUC', 'aupr': 'AUPR'}
N_M = len(METRICS)
W = 0.14

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

# ============================================================
# BEST MODEL vs EXTERNAL (clean comparison)
# ============================================================
models = [
    # (label, f1, p, r, auc, aupr, color, category)
    ("tri_branch\n(OURS)",               0.7587, 0.8123, 0.7117, 0.9329, 0.7281, C_BEST, "Ours"),
    ("DCdetector",       0.7553, 0.7530, 0.7577, 0.9337, 0.0, C_EXT, "External"),
    ("USAD",             0.7417, 0.7846, 0.7032, 0.9471, 0.0, C_EXT, "External"),
    ("MTAD-GAT",         0.7194, 0.7642, 0.6795, 0.9376, 0.0, C_EXT, "External"),
    ("CAN",              0.7057, 0.6258, 0.8090, 0.9534, 0.0, C_EXT, "External"),
    ("DAGMM",            0.7048, 0.6853, 0.7254, 0.9431, 0.0, C_EXT, "External"),
    ("TranAD",           0.6958, 0.5998, 0.8285, 0.9513, 0.0, C_EXT, "External"),
    ("MAD-GAN",          0.6851, 0.6979, 0.6726, 0.9325, 0.0, C_EXT, "External"),
]

fig, ax = plt.subplots(figsize=(13, 6))
data = [{'f1': f1, 'precision': p, 'recall': r, 'roc_auc': auc, 'pr_auc': aupr}
        for _, f1, p, r, auc, aupr, _, _ in models]
labels = [m[0] for m in models]
colors = [m[6] for m in models]

x = np.arange(len(models))
for j, metric in enumerate(METRICS):
    key = 'roc_auc' if metric == 'auc' else ('pr_auc' if metric == 'aupr' else metric)
    vals = [d[key] for d in data]
    offset = (j - 2) * W
    bars = ax.bar(x + offset, vals, W, label=METRIC_LABELS[metric],
                  color=M_COLORS[metric], edgecolor='white', alpha=0.9)
    if metric == 'aupr': continue  # skip AUPR annotation for ext models (no data)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                    f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', rotation=90)

# Highlight ours
ax.axvspan(-0.5, 0.5, alpha=0.08, color=C_BEST)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Score'); ax.set_ylim(0.55, 1.02)
ax.set_title('Best Model vs External Baselines: All Metrics', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=5, loc='lower right')
ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Delta annotations
for metric, mlabel in [('f1', 'F1')]:
    key = 'roc_auc' if metric == 'auc' else ('pr_auc' if metric == 'aupr' else metric)
    best_v = data[0][key]
    dc_v = data[1][key]
    delta = best_v - dc_v
    # Add text at top
    ax.annotate(f'{mlabel}: {best_v:.4f} (vs DCdetector {dc_v:.4f}, +{delta:.4f})',
                xy=(0, best_v), xytext=(0, 1.0),
                fontsize=8, fontweight='bold', color=C_BEST, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_BEST))

save(fig, 'best_vs_external.png')
print(f'[OK] best_vs_external.png')

# ============================================================
# ABLATION: Key module contributions (clean, 5 metrics)
# ============================================================
BASE = os.path.join(os.path.dirname(__file__), "outputs", "swat_normal_train_merged_test")
def load_raw(name):
    path_map = {
        'baseline': 'metrics.json',
        'temporal_attn': 'temporal_attn/metrics.json',
        'dynamic_prior_feat': 'dynamic_prior_feat/metrics.json',
        'prior_fusion': 'prior_fusion/metrics.json',
        'ms_tcn': 'ms_tcn/metrics.json',
        'tri_branch': 'tri_branch/metrics.json',
    }
    path = os.path.join(BASE, path_map[name])
    if os.path.exists(path):
        return json.load(open(path))['raw']
    return None

ablation = [
    ('baseline', 'Baseline\nGATv2+TCN+GRU'),
    ('temporal_attn', '+ Temporal\nAttention'),
    ('dynamic_prior_feat', '+ DynPrior\nFeatFusion'),
    ('prior_fusion', '+ Prior\nFusion'),
    ('ms_tcn', '+ MultiScale\nTCN'),
    ('tri_branch', '+ Tri-Branch\n(OURS best)'),
]
ab_data = [load_raw(k) for k, _ in ablation]
ab_labels = [l for _, l in ablation]

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ab_data))
for j, metric in enumerate(METRICS):
    key = 'roc_auc' if metric == 'auc' else ('pr_auc' if metric == 'aupr' else metric)
    vals = [d[key] for d in ab_data]
    offset = (j - 2) * W
    bars = ax.bar(x + offset, vals, W, label=METRIC_LABELS[metric],
                  color=M_COLORS[metric], edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f'{v:.3f}', ha='center', fontsize=5, fontweight='bold', rotation=90)

# F1 deltas
base_f1 = ab_data[0]['f1']
for i, d in enumerate(ab_data[1:], 1):
    delta = d['f1'] - base_f1
    sign = '+' if delta > 0 else ''
    ax.annotate(f'F1 {sign}{delta:.4f}', xy=(i, d['f1']),
                xytext=(i, max(d['roc_auc'], d['f1']) + 0.02),
                fontsize=7, fontweight='bold', ha='center',
                color='#2E7D32' if delta > 0 else '#C62828')

ax.set_xticks(x); ax.set_xticklabels(ab_labels, fontsize=8)
ax.set_ylabel('Score'); ax.set_ylim(0.55, 1.0)
ax.set_title('Ablation Study: Module Contributions', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=5, loc='lower right')
ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'ablation_clean.png')
print(f'[OK] ablation_clean.png')

# ============================================================
# SCORE EVOLUTION (3 stages, 5 metrics)
# ============================================================
stages_data = [
    {'f1': 0.7546, 'precision': 0.7957, 'recall': 0.7176, 'roc_auc': 0.9374, 'pr_auc': 0.7281},
    {'f1': 0.7587, 'precision': 0.8123, 'recall': 0.7117, 'roc_auc': 0.9329, 'pr_auc': 0.7281},
    {'f1': 0.8260, 'precision': 0.9372, 'recall': 0.7383, 'roc_auc': 0.9390, 'pr_auc': 0.7489},
]
stage_labels = ['IQR+k=5\n(original)', 'IQR+k=1\n(optimized k)', 'raw+max\n(val_th q=0.9999)']
stage_colors = [C_EXT, C_OURS, C_BEST]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(stages_data))
for j, metric in enumerate(METRICS):
    key = 'roc_auc' if metric == 'auc' else ('pr_auc' if metric == 'aupr' else metric)
    vals = [d[key] for d in stages_data]
    offset = (j - 2) * W
    bars = ax.bar(x + offset, vals, W, label=METRIC_LABELS[metric],
                  color=M_COLORS[metric], edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f'{v:.4f}', ha='center', fontsize=6, fontweight='bold', rotation=90)

# Deltas vs stage 0
for metric in METRICS:
    key = 'roc_auc' if metric == 'auc' else ('pr_auc' if metric == 'aupr' else metric)
    delta = stages_data[2][key] - stages_data[0][key]
    if abs(delta) > 0.001:
        sign = '+' if delta > 0 else ''
        j = METRICS.index(metric)
        ax.text(2 + (j-2)*W, stages_data[2][key] + 0.008,
                f'{sign}{delta:.4f}', ha='center', fontsize=6, fontweight='bold',
                color='#2E7D32' if delta > 0 else '#C62828')

ax.set_xticks(x); ax.set_xticklabels(stage_labels, fontsize=9)
ax.set_ylabel('Score'); ax.set_ylim(0.68, 1.0)
ax.set_title('Score Evolution: Original -> Optimized', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, ncol=5, loc='lower right')
ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'score_evolution_clean.png')
print(f'[OK] score_evolution_clean.png')

print(f'\nClean comparison charts saved to: {OUT}')
