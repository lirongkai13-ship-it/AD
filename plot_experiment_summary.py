"""Comprehensive experiment summary charts for paper.
Every chart includes all 5 metrics: F1, Precision, Recall, AUC, AUPR.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, json

OUT = os.path.join(os.path.dirname(__file__), "results", "plots", "experiment_summary")
os.makedirs(OUT, exist_ok=True)

# Colors
C_BEST = '#E53935'; C_2ND = '#FF6F00'; C_OURS = '#1A237E'
C_EXT = '#78909C'; C_GREEN = '#2E7D32'; C_RED = '#C62828'
C_BLUE = '#1565C0'; C_ORANGE = '#EF6C00'; C_PURPLE = '#7B1FA2'
M_COLORS = {'f1': C_RED, 'precision': C_BLUE, 'recall': C_GREEN,
            'auc': C_ORANGE, 'aupr': C_PURPLE}

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  [OK] {name}')

def load_metrics(name):
    """Load raw metrics dict from a model checkpoint dir."""
    BASE = os.path.join(os.path.dirname(__file__), "outputs", "swat_normal_train_merged_test")
    path_map = {
        'tri_branch': 'tri_branch/metrics.json',
        'parallel_usad_prior': 'parallel_usad_prior/metrics.json',
        'parallel_usad_ms': 'parallel_usad_ms/metrics.json',
        'dynamic_usad': 'dynamic_usad/metrics.json',
        'dyn_ms_usad': 'dyn_ms_usad/metrics.json',
        'parallel_usad': 'parallel_usad/metrics.json',
        'usad_dual': 'usad_dual/metrics.json',
        'parallel_usad_d': 'parallel_usad_d/metrics.json',
        'static_usad': 'static_usad/metrics.json',
        'dyn_usad_prior_graph': 'dyn_usad_prior_graph/metrics.json',
        'baseline': 'metrics.json',
        'temporal_attn': 'temporal_attn/metrics.json',
        'prior_fusion': 'prior_fusion/metrics.json',
        'ms_tcn': 'ms_tcn/metrics.json',
        'dynamic_graph': 'dynamic_graph/metrics.json',
        'prior_dynamic': 'prior_dynamic/metrics.json',
        'dynamic_prior_feat': 'dynamic_prior_feat/metrics.json',
        'tri_branch_gamma_0_0': 'tri_branch_gamma_0_0/metrics.json',
        'tri_branch_gamma_0_02': 'tri_branch_gamma_0_02/metrics.json',
        'tri_branch_gate_0_5': 'tri_branch_gate_0_5/metrics.json',
        'parallel_usad_b': 'parallel_usad_b/metrics.json',
        'abc_b': 'abc_b/metrics.json',
        'abc_b_v2': 'abc_b_v2/metrics.json',
        'parallel_usad_fast': 'parallel_usad_fast/metrics.json',
        'dynamic_usad_prior': 'dynamic_usad_prior/metrics.json',
    }
    if name not in path_map:
        return None
    path = os.path.join(BASE, path_map[name])
    if os.path.exists(path):
        return json.load(open(path))['raw']
    return None

def mget(metrics, key, default=0.0):
    """Safely get metric value. Handles roc_auc->auc, pr_auc->aupr key mapping."""
    if metrics is None: return default
    key_map = {'auc': 'roc_auc', 'aupr': 'pr_auc'}
    actual_key = key_map.get(key, key)
    return metrics.get(actual_key, default)

# ============================================================
# Shared plotter: grouped bar chart with all 5 metrics
# ============================================================
METRICS = ['f1', 'precision', 'recall', 'auc', 'aupr']
METRIC_LABELS = {'f1': 'F1', 'precision': 'Precision', 'recall': 'Recall', 'auc': 'AUC', 'aupr': 'AUPR'}
N_METRICS = len(METRICS)
WIDTH = 0.14

def plot_grouped_bars(ax, data, xlabels, title, ylim=None, annotate=True):
    """
    data: list of dicts with keys f1, precision, recall, auc, aupr
    xlabels: list of strings for x-axis
    """
    x = np.arange(len(data))
    for j, metric in enumerate(METRICS):
        vals = [mget(d, metric) for d in data]
        offset = (j - (N_METRICS - 1) / 2) * WIDTH
        bars = ax.bar(x + offset, vals, WIDTH, label=METRIC_LABELS[metric],
                      color=M_COLORS[metric], edgecolor='white', alpha=0.9)
        if annotate:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                        f'{v:.3f}', ha='center', fontsize=4.5, fontweight='bold', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_title(title, fontweight='bold', fontsize=12)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=5, loc='lower right')
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)


# ============================================================
# 1. ALL MODELS COMPARISON (all 5 metrics)
# ============================================================
def plot_all_models():
    our_models = [
        ('tri_branch', 'tri_branch\n(OURS best)', C_BEST),
        ('parallel_usad_prior', 'parallel_usad\n_prior', C_2ND),
        ('parallel_usad_ms', 'parallel_usad\n_ms', C_OURS),
        ('dynamic_usad', 'dynamic_usad\n(serial)', '#5C6BC0'),
        ('dyn_ms_usad', 'dyn_ms_usad\n(serial)', '#5C6BC0'),
        ('parallel_usad', 'parallel_usad', C_OURS),
        ('usad_dual', 'usad_dual\n(serial base)', '#5C6BC0'),
        ('parallel_usad_d', 'parallel_usad_d', C_OURS),
        ('static_usad', 'static_usad\n(serial)', '#5C6BC0'),
        ('dyn_usad_prior_graph', 'dyn_usad_prior\n_graph', '#C62828'),
    ]
    external = [
        ('DCdetector', 'DCdetector\n(ext)', None, 0.7553, 0.7530, 0.7577, 0.9337, 0.0),
        ('USAD_ext', 'USAD\n(ext)', None, 0.7417, 0.7846, 0.7032, 0.9471, 0.0),
        ('MTAD-GAT', 'MTAD-GAT\n(ext)', None, 0.7194, 0.7642, 0.6795, 0.9376, 0.0),
        ('CAN', 'CAN\n(ext)', None, 0.7057, 0.6258, 0.8090, 0.9534, 0.0),
        ('DAGMM', 'DAGMM\n(ext)', None, 0.7048, 0.6853, 0.7254, 0.9431, 0.0),
        ('TranAD', 'TranAD\n(ext)', None, 0.6958, 0.5998, 0.8285, 0.9513, 0.0),
        ('MAD-GAN', 'MAD-GAN\n(ext)', None, 0.6851, 0.6979, 0.6726, 0.9325, 0.0),
    ]

    data = []
    labels = []
    colors_bar = []
    for key, label, color in our_models:
        m = load_metrics(key)
        if m:
            data.append(m); labels.append(label); colors_bar.append(color)

    for _, label, _, f1, p, r, auc, aupr in external:
        data.append({'f1': f1, 'precision': p, 'recall': r, 'roc_auc': auc, 'pr_auc': aupr})
        labels.append(label); colors_bar.append(C_EXT)

    # Sort by F1
    combined = list(zip(data, labels, colors_bar))
    combined.sort(key=lambda x: x[0]['f1'], reverse=True)
    data, labels, colors_bar = zip(*combined)

    fig, ax = plt.subplots(figsize=(18, 6.5))
    plot_grouped_bars(ax, list(data), list(labels),
                      'All Models: Full Metrics Comparison (F1 / P / R / AUC / AUPR)',
                      ylim=(0.55, 1.02))
    save(fig, '01_all_models_full_metrics.png')


# ============================================================
# 2. ABLATION STUDY (full metrics)
# ============================================================
def plot_ablation():
    keys = ['baseline', 'temporal_attn', 'dynamic_prior_feat', 'prior_fusion', 'ms_tcn', 'dynamic_graph', 'prior_dynamic']
    labels = ['Baseline\nGATv2+TCN+GRU', '+Temporal\nAttention', '+DynPrior\nFeatFusion',
              '+Prior\nFusion', '+MultiScale\nTCN', '+Dynamic\nPearson', '+Prior\nDynamic']
    data = []
    for k in keys:
        m = load_metrics(k)
        data.append(m if m else {'f1': 0, 'precision': 0, 'recall': 0, 'roc_auc': 0, 'pr_auc': 0})

    fig, ax = plt.subplots(figsize=(15, 6))
    plot_grouped_bars(ax, data, labels, 'Ablation Study: All Metrics', ylim=(0.55, 0.98))
    # Delta annotation
    base_f1 = data[0]['f1']
    for i, d in enumerate(data[1:], 1):
        delta = d['f1'] - base_f1
        sign = '+' if delta > 0 else ''
        ax.text(i, max(d.values()) + 0.008, f'F1:{sign}{delta:.4f}', ha='center', fontsize=8, fontweight='bold')
    save(fig, '02_ablation_full_metrics.png')


# ============================================================
# 3. USAD SERIES (full metrics)
# ============================================================
def plot_usad_series():
    keys = ['tri_branch', 'parallel_usad_prior', 'parallel_usad_ms', 'dynamic_usad',
            'dyn_ms_usad', 'parallel_usad', 'usad_dual', 'parallel_usad_d',
            'static_usad', 'dyn_usad_prior_graph']
    data = [load_metrics(k) for k in keys]
    labels = [k.replace('_', '\n') for k in keys]
    # Sort by F1
    pairs = sorted(zip(data, labels, keys), key=lambda x: x[0]['f1'] if x[0] else 0, reverse=True)
    data, labels, _ = zip(*pairs)

    fig, ax = plt.subplots(figsize=(15, 6))
    plot_grouped_bars(ax, list(data), list(labels), 'USAD Series: Full Metrics', ylim=(0.55, 0.98))
    save(fig, '03_usad_series_full_metrics.png')


# ============================================================
# 4. TOP-K SWEEP (all metrics from CSV)
# ============================================================
def plot_topk_sweep():
    csv_path = os.path.join(os.path.dirname(__file__), "results", "score_optimization", "topk_score_results.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] topk_score_results.csv not found")
        return
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f): rows.append(r)

    K = [1, 3, 5, 8, 10, 15, 20, 51]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for err_type, color in [('e1', C_BEST), ('e2', C_BLUE), ('e12', C_GREEN)]:
        k_map = {int(r['topk']): {m: float(r[m]) for m in METRICS}
                 for r in rows if r['error_type'] == err_type}
        for j, metric in enumerate(METRICS):
            ax = axes[j // 3][j % 3]
            vals = [k_map[k][metric] for k in K]
            ax.plot(K, vals, 'o-', color=color, label=err_type, markersize=6, linewidth=2)
            ax.set_xlabel('Top-K'); ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_title(METRIC_LABELS[metric]); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Hide extra subplot
    axes[1][2].set_visible(False)
    fig.suptitle('Top-K Sweep: All Metrics (e1/e2/e12)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, '04_topk_sweep_all_metrics.png')


# ============================================================
# 5. SCORE METHOD COMPARISON
# ============================================================
def plot_score_methods():
    # From raw_max_reliability results: best per method with val_th
    csv_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "threshold_sweep_all_methods.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] threshold_sweep_all_methods.csv")
        return
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f): rows.append(r)

    reportable = [r for r in rows if 'test_sweep' not in r['method']]
    methods = ['IQR+k=1 (val_th)', 'IQR+k=5 (val_th)', 'raw+max (train_th)', 'raw+max (val_th)']
    short_labels = ['IQR+k=1\n(val_th)', 'IQR+k=5\n(val_th)', 'raw+max\n(train_th)', 'raw+max\n(val_th)']

    data = []
    for m in methods:
        m_rows = [r for r in reportable if r['method'] == m]
        best = max(m_rows, key=lambda r: float(r['f1']))
        data.append({metric: float(best[metric]) for metric in METRICS})

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_grouped_bars(ax, data, short_labels, 'Score Method Comparison (Val/Train Threshold Only)', ylim=(0.45, 1.0))
    save(fig, '05_score_method_comparison.png')


# ============================================================
# 6. GAMMA ABLATION (full metrics)
# ============================================================
def plot_gamma():
    keys = ['tri_branch_gamma_0_0', 'tri_branch_gamma_0_02', 'tri_branch']
    data = [load_metrics(k) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    for j, metric in enumerate(METRICS):
        vals = [mget(d, metric) for d in data]
        ax.plot(x, vals, 'o-', color=M_COLORS[metric], label=METRIC_LABELS[metric], markersize=8, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f'{v:.4f}', ha='center', fontsize=7, fontweight='bold', color=M_COLORS[metric])

    ax.set_xticks(x); ax.set_xticklabels(['gamma=0', 'gamma=0.02', 'gamma=0.05 (best)'], fontsize=10)
    ax.set_ylabel('Score'); ax.set_title('Gamma Ablation: All Metrics', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, ncol=5); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, '06_gamma_ablation_all_metrics.png')


# ============================================================
# 7. GATE SCALE ABLATION (full metrics)
# ============================================================
def plot_gate_scale():
    data = [load_metrics('tri_branch_gate_0_5'), load_metrics('tri_branch')]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    for j, metric in enumerate(METRICS):
        vals = [mget(d, metric) for d in data]
        offset = (j - (N_METRICS - 1) / 2) * (WIDTH + 0.02)
        ax.bar(x + offset, vals, WIDTH, label=METRIC_LABELS[metric], color=M_COLORS[metric], edgecolor='white')
        for i, v in enumerate(vals):
            ax.text(i + offset, v + 0.004, f'{v:.4f}', ha='center', fontsize=6.5, fontweight='bold', rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(['gate_scale=0.5', 'gate_scale=1.0 (best)'], fontsize=10)
    ax.set_ylabel('Score'); ax.set_title('Gate Scale Ablation: All Metrics', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9, ncol=5); ax.set_ylim(0.65, 1.0)
    ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, '07_gate_scale_ablation_all_metrics.png')


# ============================================================
# 8. TEMPORAL BRANCH ABLATION (full metrics)
# ============================================================
def plot_temporal():
    keys = ['parallel_usad_prior', 'parallel_usad_ms', 'tri_branch', 'abc_b_v2', 'parallel_usad_d', 'parallel_usad_b']
    labels = ['Conv1d k=3\n(A: Best, ~25min)', 'Conv1d k=3,5,7\n(MS, ~15min)',
              '+Global Attn\n(tri_branch, ~20min)', 'Dilated Conv\n(B: 1,2,4, 6h)',
              'TCN x3\n(D: 1,2,4, 4h)', 'Global GRU\nBroadcast, ~10min)']
    data = [load_metrics(k) for k in keys]
    # Keep original order — simpler at top
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_grouped_bars(ax, data, labels, 'Temporal Branch Ablation: All Metrics', ylim=(0.62, 0.98))
    save(fig, '08_temporal_branch_all_metrics.png')


# ============================================================
# 9. TRI-BRANCH vs PRIOR (2-model side-by-side, full metrics)
# ============================================================
def plot_tribranch_vs_prior():
    data = [load_metrics('parallel_usad_prior'), load_metrics('tri_branch')]
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_grouped_bars(ax, data, ['Parallel USAD\n+ Prior', 'Tri-Branch\n+ Global Gate'],
                      'Tri-Branch vs Prior: All Metrics', ylim=(0.7, 0.98))
    # Add deltas
    for j, metric in enumerate(METRICS):
        delta = mget(data[1], metric) - mget(data[0], metric)
        if abs(delta) > 0.0001:
            sign = '+' if delta > 0 else ''
            ax.text(1 + (j - 2) * WIDTH, max(mget(data[1], metric), mget(data[0], metric)) + 0.006,
                    f'{sign}{delta:.4f}', ha='center', fontsize=6.5, fontweight='bold',
                    color=C_GREEN if delta > 0 else C_RED)
    save(fig, '09_tribranch_vs_prior.png')


# ============================================================
# 10. SCORE EVOLUTION (3 stages, full metrics)
# ============================================================
def plot_evolution():
    stages = [
        {'f1': 0.7546, 'precision': 0.7957, 'recall': 0.7176, 'roc_auc': 0.9374, 'pr_auc': 0.7281},
        {'f1': 0.8260, 'precision': 0.9372, 'recall': 0.7383, 'roc_auc': 0.9390, 'pr_auc': 0.7281},
    ]
    # raw+max from reliability data
    stages[1] = {'f1': 0.8260, 'precision': 0.9372, 'recall': 0.7383, 'roc_auc': 0.9390, 'pr_auc': 0.7489}
    # Actually let me get the real number from our sweep
    csv_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "threshold_sweep_all_methods.csv")
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            for r in csv.DictReader(f):
                if r['method'] == 'raw+max (val_th)' and abs(float(r['q']) - 0.9999) < 0.0001:
                    stages[1] = {m: float(r[m]) for m in METRICS}
                    break

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = ['Original\n(IQR+k=5, val_th)', 'Optimized\n(raw+max, val_th, q=0.9999)']
    plot_grouped_bars(ax, stages, labels, 'Score Evolution: Baseline -> Optimized', ylim=(0.68, 1.0))
    # Deltas
    for j, metric in enumerate(METRICS):
        delta = mget(stages[1], metric) - mget(stages[0], metric)
        if abs(delta) > 0.0001:
            sign = '+' if delta > 0 else ''
            ax.text(1 + (j - 2) * WIDTH, max(mget(stages[1], metric), mget(stages[0], metric)) + 0.006,
                    f'{sign}{delta:.4f}', ha='center', fontsize=7, fontweight='bold',
                    color=C_GREEN if delta > 0 else C_RED)
    save(fig, '10_score_evolution.png')


# ============================================================
# 11. PARAMETER EFFICIENCY (F1 + AUC bubble)
# ============================================================
def plot_param_efficiency():
    models = [
        ("tri_branch", 1.01, C_BEST),
        ("parallel_usad_prior", 1.00, C_2ND),
        ("parallel_usad_ms", 1.00, C_OURS),
        ("parallel_usad", 1.00, C_OURS),
        ("parallel_usad_d", 1.00, C_OURS),
        ("dynamic_usad", 0.43, '#5C6BC0'),
        ("dyn_ms_usad", 0.43, '#5C6BC0'),
        ("usad_dual", 0.43, '#5C6BC0'),
        ("static_usad", 0.43, '#5C6BC0'),
        ("baseline", 0.21, '#9FA8DA'),
    ]
    # External
    externals = [
        ("DCdetector", 0.10, 0.7553, 0.9337, C_EXT),
        ("USAD (ext)", 0.02, 0.7417, 0.9471, C_EXT),
        ("TranAD", 0.05, 0.6958, 0.9513, C_EXT),
        ("MTAD-GAT", 0.15, 0.7194, 0.9376, C_EXT),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    for name, params, color in models:
        m = load_metrics(name) if name != 'baseline' else load_metrics('baseline')
        if m:
            f1 = mget(m, 'f1'); auc = mget(m, 'auc')
            ax.scatter(params, f1, s=200, c=color, edgecolors='white', linewidth=1, zorder=5)
            ax.scatter(params, auc, s=80, c=color, edgecolors='white', linewidth=1, zorder=5, marker='s', alpha=0.6)
            ax.annotate(name.replace('_', '\n'), (params, f1), textcoords="offset points",
                       xytext=(6, -2), fontsize=6.5)

    for name, params, f1, auc, color in externals:
        ax.scatter(params, f1, s=120, c=color, edgecolors='white', linewidth=1, zorder=5, marker='D')
        ax.scatter(params, auc, s=50, c=color, edgecolors='white', linewidth=1, zorder=5, marker='s', alpha=0.4)
        ax.annotate(name, (params, f1), textcoords="offset points", xytext=(4, -8), fontsize=7)

    # Legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='F1'),
              Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='AUC')]
    ax.legend(handles=legend, fontsize=10)

    ax.set_xlabel('Parameters (Million)', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Efficiency: Params vs F1+AUC', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, '11_param_efficiency.png')


# ============================================================
# 12. THRESHOLD SWEEP (all metrics vs q)
# ============================================================
def plot_threshold_full():
    csv_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "threshold_sweep_all_methods.csv")
    if not os.path.exists(csv_path): return
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f): rows.append(r)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    method_pairs = [("raw+max (val_th)", "raw+max (val)"), ("IQR+k=5 (val_th)", "IQR+k=5 (val)")]

    for j, metric in enumerate(METRICS):
        ax = axes[j // 3][j % 3]
        for method_key, label in method_pairs:
            m_rows = sorted([r for r in rows if r['method'] == method_key], key=lambda r: float(r['q']))
            if m_rows:
                qs = [float(r['q']) for r in m_rows]
                vals = [float(r[metric]) for r in m_rows]
                ax.plot(qs, vals, 'o-', label=label, markersize=3, linewidth=1.5)
        ax.set_xlabel('Quantile q'); ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric]); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    axes[1][2].set_visible(False)
    fig.suptitle('Threshold Sweep: All Metrics vs Quantile q', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, '12_threshold_sweep_all_metrics.png')


# ============================================================
# 13. VARIABLE ANALYSIS (top ratio + frequency combined)
# ============================================================
def plot_variable_analysis():
    csv_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "variable_error_scale_stats.csv")
    freq_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "top1_variable_frequency.csv")
    if not os.path.exists(csv_path):
        print("  [SKIP] variable CSVs not found")
        return
    import csv

    # Error ratio
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f): rows.append(r)
    rows.sort(key=lambda r: float(r['ratio']), reverse=True)
    top20 = rows[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Error ratio
    names = [r['node_name'] for r in reversed(top20)]
    ratios = [float(r['ratio']) for r in reversed(top20)]
    colors = [C_RED if r > 30 else C_ORANGE if r > 15 else C_GREEN for r in ratios]
    ax1.barh(range(len(names)), ratios, color=colors, edgecolor='white')
    ax1.set_yticks(range(len(names))); ax1.set_yticklabels(names, fontsize=7)
    ax1.set_xlabel('Attack/Normal Mean Error Ratio'); ax1.set_title('Error Separation Ratio (Top-20)')
    ax1.axvline(15, color='gray', linestyle='--', alpha=0.5)

    # Right: Frequency
    freq_rows = []
    with open(freq_path, 'r') as f:
        for r in csv.DictReader(f): freq_rows.append(r)
    freq_attack = sorted(freq_rows, key=lambda r: float(r['freq_attack']), reverse=True)[:15]
    names_a = [r['node_name'] for r in freq_attack][::-1]
    freqs_a = [float(r['freq_attack']) for r in freq_attack][::-1]
    # Also get ratio for these
    ratio_map = {r['node_name']: float(r['ratio']) for r in rows}
    freq_colors = [C_RED if ratio_map.get(n, 0) > 30 else C_ORANGE for n in names_a]
    ax2.barh(range(len(names_a)), freqs_a, color=freq_colors, edgecolor='white')
    ax2.set_yticks(range(len(names_a))); ax2.set_yticklabels(names_a, fontsize=7)
    ax2.set_xlabel('Top1 Frequency (Attack)'); ax2.set_title('Top1 Variable Frequency - Attack')

    fig.suptitle('Variable-Level Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, '13_variable_analysis.png')


# ============================================================
# 14. EXTERNAL MODELS COMPARISON
# ============================================================
def plot_external_comparison():
    externals = [
        ("DCdetector", 0.7553, 0.7530, 0.7577, 0.9337, 0.0),
        ("tri_branch\n(OURS)", 0.7546, 0.7957, 0.7176, 0.9374, 0.7281),
        ("USAD", 0.7417, 0.7846, 0.7032, 0.9471, 0.0),
        ("MTAD-GAT", 0.7194, 0.7642, 0.6795, 0.9376, 0.0),
        ("CAN", 0.7057, 0.6258, 0.8090, 0.9534, 0.0),
        ("DAGMM", 0.7048, 0.6853, 0.7254, 0.9431, 0.0),
        ("TranAD", 0.6958, 0.5998, 0.8285, 0.9513, 0.0),
        ("MAD-GAN", 0.6851, 0.6979, 0.6726, 0.9325, 0.0),
    ]
    data = [{'f1': f1, 'precision': p, 'recall': r, 'auc': auc, 'aupr': aupr}
            for _, f1, p, r, auc, aupr in externals]
    labels = [n for n, _, _, _, _, _ in externals]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(data))
    width = 0.14
    for j, metric in enumerate(METRICS):
        vals = [mget(d, metric) for d in data]
        offset = (j - 2) * width
        bars = ax.bar(x + offset, vals, width, label=METRIC_LABELS[metric],
                      color=M_COLORS[metric], edgecolor='white', alpha=0.9)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                        f'{v:.3f}', ha='center', fontsize=6, fontweight='bold', rotation=90)

    # Highlight ours
    ax.axvspan(0.5, 1.5, alpha=0.08, color=C_BEST)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score'); ax.set_title('Best Models vs External Baselines', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=5); ax.set_ylim(0.55, 1.02)
    ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, '14_external_comparison.png')


# ============================================================
# 15. SCORE DISTRIBUTION STATS
# ============================================================
def plot_score_distribution():
    csv_path = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability", "score_distribution_stats.csv")
    if not os.path.exists(csv_path): return
    import csv
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f): rows.append(r)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = [r['method'] for r in rows]
    sep = [float(r['separation_ratio']) for r in rows]
    n_mean = [float(r['normal_mean']) for r in rows]
    a_mean = [float(r['attack_mean']) for r in rows]

    x = np.arange(len(methods))
    width = 0.3
    ax.bar(x - width/2, n_mean, width, label='Normal Mean', color='#4CAF50', edgecolor='white')
    ax.bar(x + width/2, a_mean, width, label='Attack Mean', color='#F44336', edgecolor='white')
    for i, s in enumerate(sep):
        ax.text(i, max(n_mean[i], a_mean[i]) + 2, f'Sep={s:.1f}x', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Mean Score'); ax.legend(fontsize=10)
    ax.set_title('Score Distribution: Normal vs Attack Mean', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    save(fig, '15_score_distribution.png')


# ============================================================
if __name__ == '__main__':
    print("Generating comprehensive experiment summary charts...")
    plot_all_models()
    plot_ablation()
    plot_usad_series()
    plot_topk_sweep()
    plot_score_methods()
    plot_gamma()
    plot_gate_scale()
    plot_temporal()
    plot_tribranch_vs_prior()
    plot_evolution()
    plot_param_efficiency()
    plot_threshold_full()
    plot_variable_analysis()
    plot_external_comparison()
    plot_score_distribution()
    print(f'\nAll 15 charts saved to: {OUT}')
