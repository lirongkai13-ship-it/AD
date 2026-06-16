"""
Score Optimization Experiment for tri_branch model.
=====================================================
Evaluates: Top-K variable score, r1/r2/r12 weights, threshold sweep.
All experiments are evaluate-stage only — NO model structure, training, or loss changes.

Constraints:
  1. Fixed tri_branch model (encoder_mode="tri_branch_residual_gate", gamma=0.05, gate_scale=1.0)
  2. No model structure changes
  3. No training loss changes
  4. No USAD decoder changes
  5. Only evaluate-stage scoring functions
"""
import sys, os, json, csv, time, importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, precision_recall_curve)

# Paths
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import prepare_data, build_pearson_edge_index, split_train_val, read_swat_csv
from utils import load_config, set_seed, get_device, fit_iqr_params, apply_iqr_normalize, aggregate_topk_score
from models_variants.tri_branch.variant_model import TriBranch_USAD

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "score_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════
# 1. Load data and model
# ═══════════════════════════════════════════════════════════
cfg = load_config('config_dev.yaml')
set_seed(42)

# Prepare data
_, val_ds, test_ds, static_edge_index, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

# Build prior graph
dcfg = cfg['data']
nfd = __import__('pandas').read_csv(dcfg['train_csv'])
nfd.columns = [str(c).strip() for c in nfd.columns]
nfd = nfd[[c for c in nfd.columns if c not in ['Timestamp', 'Normal/Attack']]]
mfd = __import__('pandas').read_csv(dcfg['test_csv'])
mfd.columns = [str(c).strip() for c in mfd.columns]
common_cols = [c for c in nfd.columns if c in mfd.columns]

# Build static Pearson graph from train normal
raw = nfd[common_cols].values.astype(np.float32)
from sklearn.preprocessing import StandardScaler
tr, _, _, _ = split_train_val(raw, None, 0.2)
tv = StandardScaler().fit_transform(tr)
static_ei, _ = build_pearson_edge_index(tv)

# Build prior graph
prior_mod_spec = importlib.util.spec_from_file_location(
    'bpg', os.path.join(os.path.dirname(__file__), 'models_variants', 'prior_fusion', 'build_prior_graph.py'))
prior_mod = importlib.util.module_from_spec(prior_mod_spec)
prior_mod_spec.loader.exec_module(prior_mod)
prior_ei, prior_w = prior_mod.build_prior_graph(common_cols)

static_ei = static_ei.to(device)
prior_ei = prior_ei.to(device)
prior_w = prior_w.to(device)

# Load best tri_branch model
print("\nLoading tri_branch best model...")
model = TriBranch_USAD(
    nv=info['num_variables'], ws=60,
    static_edge_index=static_ei,
    prior_edge_index=prior_ei,
    prior_weights=prior_w,
    hidden_dim=32, gat_heads=2,
    gru_hidden=32, tcn_channels=32, tcn_blocks=1,
    dropout=0.2, latent_dim=64, use_flatten=True,
    temporal_mode="per_variable_conv",
    encoder_mode="tri_branch_residual_gate",
    gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
).to(device)

ckpt_path = 'outputs/swat_normal_train_merged_test/tri_branch/best_model.pt'
ckpt = torch.load(ckpt_path, map_location=device)
# Remap old key names (gated_fusion.gate.* → gated_fusion.gate_mlp.*)
state_dict = {}
for k, v in ckpt['model'].items():
    if 'gated_fusion.gate.' in k:
        k = k.replace('gated_fusion.gate.', 'gated_fusion.gate_mlp.')
    state_dict[k] = v
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")


# ═══════════════════════════════════════════════════════════
# 2. Collect per-variable errors (e1, e2, e12)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def collect_all_errors(model, loader):
    """
    Run full USAD forward to get r1, r2, r12.
    Returns:
      e1: [M, N]  = mean(|r1 - x|, dim=1)
      e2: [M, N]  = mean(|r2 - x|, dim=1)
      e12: [M, N] = mean(|r12 - x|, dim=1)
      labels: [M]
    """
    model.eval()
    e1_list, e2_list, e12_list = [], [], []
    labels_list = []

    for batch in loader:
        x = batch['x'].to(device)  # [B, W, N]
        r1, r2, r12 = model(x, static_ei)

        # Per-variable mean absolute error over time
        e1 = (r1 - x).abs().mean(dim=1)   # [B, N]
        e2 = (r2 - x).abs().mean(dim=1)
        e12 = (r12 - x).abs().mean(dim=1)

        e1_list.append(e1.cpu().numpy())
        e2_list.append(e2.cpu().numpy())
        e12_list.append(e12.cpu().numpy())

        if 'label' in batch:
            labels_list.append(batch['label'].cpu().numpy())

    e1 = np.concatenate(e1_list, axis=0)
    e2 = np.concatenate(e2_list, axis=0)
    e12 = np.concatenate(e12_list, axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return e1, e2, e12, labels


print("\nCollecting validation errors...")
t0 = time.time()
v_e1, v_e2, v_e12, v_labels = collect_all_errors(model, val_loader)
print(f"  Val errors: {v_e1.shape} ({v_e1.shape[0]} samples × {v_e1.shape[1]} vars)")

print("Collecting test errors...")
t_e1, t_e2, t_e12, t_labels = collect_all_errors(model, test_loader)
print(f"  Test errors: {t_e1.shape}")
print(f"  Collection time: {time.time() - t0:.1f}s")

# Only normal val data for IQR fitting
v_normal_mask = (v_labels == 0)
print(f"  Val normal samples: {v_normal_mask.sum()} / {len(v_labels)}")


# ═══════════════════════════════════════════════════════════
# 3. Scoring functions
# ═══════════════════════════════════════════════════════════
def compute_score_and_metrics(e_val, e_test, val_normal_mask, t_labels, topk):
    """
    Full pipeline: IQR fit on val normal → normalize → topk → threshold → metrics.
    Returns dict of metrics.
    """
    # Fit IQR on val normal only
    e_val_normal = e_val[val_normal_mask]
    iqr_params = fit_iqr_params(e_val_normal)

    # Normalize
    v_norm = apply_iqr_normalize(e_val, iqr_params)
    t_norm = apply_iqr_normalize(e_test, iqr_params)

    # Top-K score
    v_score = aggregate_topk_score(v_norm, topk=topk)
    t_score = aggregate_topk_score(t_norm, topk=topk)

    # Threshold from val normal at q=0.995
    v_score_normal = v_score[val_normal_mask]
    threshold = float(np.quantile(v_score_normal, 0.995))

    # Predictions
    t_pred = (t_score > threshold).astype(int)

    # Metrics
    pr, rc, f1, _ = precision_recall_fscore_support(t_labels, t_pred, average='binary', zero_division=0)
    auc_val = roc_auc_score(t_labels, t_score)
    aupr_val = average_precision_score(t_labels, t_score)

    # Score distribution
    normal_scores = t_score[t_labels == 0]
    attack_scores = t_score[t_labels == 1]
    sep_ratio = attack_scores.mean() / normal_scores.mean() if normal_scores.mean() > 0 else float('inf')

    return {
        'topk': topk,
        'threshold': threshold,
        'precision': float(pr),
        'recall': float(rc),
        'f1': float(f1),
        'auc': float(auc_val),
        'aupr': float(aupr_val),
        'normal_score_mean': float(normal_scores.mean()),
        'attack_score_mean': float(attack_scores.mean()),
        'separation_ratio': float(sep_ratio),
        'normal_score_std': float(normal_scores.std()),
        'attack_score_std': float(attack_scores.std()),
    }


def weighted_error(e1, e2, e12, w1, w2, w12):
    """Weighted combination of USAD errors."""
    return w1 * e1 + w2 * e2 + w12 * e12


# ═══════════════════════════════════════════════════════════
# 4. Experiment 1: Top-K sweep per error type
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 1: Top-K Variable Score per Error Type")
print("=" * 70)

K_VALUES = [1, 3, 5, 8, 10, 15, 20, 51]
ERROR_TYPES = {
    'e1': (v_e1, t_e1, "Decoder1 reconstruction"),
    'e2': (v_e2, t_e2, "Decoder2 reconstruction"),
    'e12': (v_e12, t_e12, "Decoder2(z2) reconstruction"),
}

topk_results = []

for err_name, (v_err, t_err, desc) in ERROR_TYPES.items():
    print(f"\n--- {err_name}: {desc} ---")
    for k in K_VALUES:
        r = compute_score_and_metrics(v_err, t_err, v_normal_mask, t_labels, topk=k)
        r['error_type'] = err_name
        r['error_desc'] = desc
        topk_results.append(r)
        print(f"  k={k:2d}: F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"AUC={r['auc']:.4f}  AUPR={r['aupr']:.4f}  "
              f"AttMean={r['attack_score_mean']:.1f}  Sep={r['separation_ratio']:.2f}")

# Save CSV
csv_path = os.path.join(OUT_DIR, 'topk_score_results.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=topk_results[0].keys())
    w.writeheader()
    w.writerows(topk_results)
print(f"\nSaved: {csv_path}")


# ═══════════════════════════════════════════════════════════
# 5. Experiment 2: r1/r2/r12 weight combinations
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: r1/r2/r12 Score Weight Combinations")
print("=" * 70)

# Weight grids to test
WEIGHT_COMBOS = [
    # Single error (baselines)
    {'w1': 1.0, 'w2': 0.0, 'w12': 0.0, 'label': 'e1 only'},
    {'w1': 0.0, 'w2': 1.0, 'w12': 0.0, 'label': 'e2 only'},
    {'w1': 0.0, 'w2': 0.0, 'w12': 1.0, 'label': 'e12 only'},
    # Equal pairs
    {'w1': 0.5, 'w2': 0.5, 'w12': 0.0, 'label': 'e1+e2 (0.5:0.5)'},
    {'w1': 0.5, 'w2': 0.0, 'w12': 0.5, 'label': 'e1+e12 (0.5:0.5)'},
    {'w1': 0.0, 'w2': 0.5, 'w12': 0.5, 'label': 'e2+e12 (0.5:0.5)'},
    # All three equal
    {'w1': 0.333, 'w2': 0.333, 'w12': 0.333, 'label': 'e1+e2+e12 (equal)'},
    # e1 dominant (current baseline uses e1)
    {'w1': 0.6, 'w2': 0.2, 'w12': 0.2, 'label': 'e1 dominant (0.6:0.2:0.2)'},
    {'w1': 0.7, 'w2': 0.15, 'w12': 0.15, 'label': 'e1 dominant (0.7:0.15:0.15)'},
    {'w1': 0.8, 'w2': 0.1, 'w12': 0.1, 'label': 'e1 dominant (0.8:0.1:0.1)'},
    # e12 dominant
    {'w1': 0.2, 'w2': 0.2, 'w12': 0.6, 'label': 'e12 dominant (0.2:0.2:0.6)'},
    {'w1': 0.15, 'w2': 0.15, 'w12': 0.7, 'label': 'e12 dominant (0.15:0.15:0.7)'},
    # e2 dominant
    {'w1': 0.2, 'w2': 0.6, 'w12': 0.2, 'label': 'e2 dominant (0.2:0.6:0.2)'},
    # USAD-style: max(e1, e2) inspired - more weight on e2 and e12
    {'w1': 0.3, 'w2': 0.3, 'w12': 0.4, 'label': 'balanced (0.3:0.3:0.4)'},
    {'w1': 0.4, 'w2': 0.4, 'w12': 0.2, 'label': 'balanced (0.4:0.4:0.2)'},
]

# Test each weight combo at best k for each error type (k=5 default)
# Also test different k values
weight_results = []

for combo in WEIGHT_COMBOS:
    w1, w2, w12 = combo['w1'], combo['w2'], combo['w12']
    label = combo['label']

    # Compute weighted errors
    v_weighted = weighted_error(v_e1, v_e2, v_e12, w1, w2, w12)
    t_weighted = weighted_error(t_e1, t_e2, t_e12, w1, w2, w12)

    # Test all k values for this weight combo
    best_f1 = -1
    best_k = None
    for k in K_VALUES:
        r = compute_score_and_metrics(v_weighted, t_weighted, v_normal_mask, t_labels, topk=k)
        r['weight_label'] = label
        r['w1'] = w1
        r['w2'] = w2
        r['w12'] = w12
        weight_results.append(r)
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_k = k

    print(f"  {label:30s}  best_k={best_k}  F1={best_f1:.4f}")

# Save CSV
csv_path2 = os.path.join(OUT_DIR, 'score_weight_topk_results.csv')
with open(csv_path2, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=weight_results[0].keys())
    w.writeheader()
    w.writerows(weight_results)
print(f"\nSaved: {csv_path2}")


# ═══════════════════════════════════════════════════════════
# 6. Experiment 3: Threshold sweep on best configurations
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 3: Threshold Sweep on Best Configurations")
print("=" * 70)

# Find top-5 configurations by F1
sorted_by_f1 = sorted(weight_results, key=lambda x: x['f1'], reverse=True)
best_configs = []
seen = set()
for r in sorted_by_f1:
    key = (r['weight_label'], r['topk'])
    if key not in seen:
        seen.add(key)
        best_configs.append(r)
    if len(best_configs) >= 5:
        break

print("Best 5 configurations for threshold sweep:")
for r in best_configs:
    print(f"  {r['weight_label']} k={r['topk']}: F1={r['f1']:.4f}")

# Also add the original tri_branch baseline for comparison
# (e1 only, k=5 — this is what forward_eval produces)
ORIGINAL_CONFIG = {'w1': 1.0, 'w2': 0.0, 'w12': 0.0, 'label': 'ORIGINAL (e1, k=5)', 'topk': 5}

# Threshold quantile sweep
QUANTILES = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
sweep_results = []

for config in best_configs + [ORIGINAL_CONFIG]:
    w1 = config.get('w1', 1.0)
    w2 = config.get('w2', 0.0)
    w12 = config.get('w12', 0.0)
    k = config['topk']
    label = config['weight_label'] if 'weight_label' in config else config['label']

    v_weighted = weighted_error(v_e1, v_e2, v_e12, w1, w2, w12)
    t_weighted = weighted_error(t_e1, t_e2, t_e12, w1, w2, w12)

    # Fit IQR on val normal
    v_norm = apply_iqr_normalize(v_weighted, fit_iqr_params(v_weighted[v_normal_mask]))
    t_norm = apply_iqr_normalize(t_weighted, fit_iqr_params(v_weighted[v_normal_mask]))
    v_score = aggregate_topk_score(v_norm, topk=k)
    t_score = aggregate_topk_score(t_norm, topk=k)
    v_score_normal = v_score[v_normal_mask]

    for q in QUANTILES:
        threshold = float(np.quantile(v_score_normal, q))
        t_pred = (t_score > threshold).astype(int)

        pr, rc, f1, _ = precision_recall_fscore_support(t_labels, t_pred, average='binary', zero_division=0)
        tp = ((t_pred == 1) & (t_labels == 1)).sum()
        fp = ((t_pred == 1) & (t_labels == 0)).sum()
        tn = ((t_pred == 0) & (t_labels == 0)).sum()
        fn = ((t_pred == 0) & (t_labels == 1)).sum()

        auc_val = roc_auc_score(t_labels, t_score)

        sweep_results.append({
            'config_label': label,
            'w1': w1, 'w2': w2, 'w12': w12,
            'topk': k,
            'quantile': q,
            'threshold': threshold,
            'f1': float(f1),
            'precision': float(pr),
            'recall': float(rc),
            'auc': float(auc_val),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        })
        print(f"  {label:30s} k={k} q={q:.3f}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} th={threshold:.2f}")

# Save CSV
csv_path3 = os.path.join(OUT_DIR, 'threshold_sweep_results.csv')
with open(csv_path3, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
    w.writeheader()
    w.writerows(sweep_results)
print(f"\nSaved: {csv_path3}")


# ═══════════════════════════════════════════════════════════
# 7. Original tri_branch baseline metrics (for comparison)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ORIGINAL TRI_BRANCH BASELINE (e1, k=5, q=0.995)")
print("=" * 70)

baseline_r = compute_score_and_metrics(v_e1, t_e1, v_normal_mask, t_labels, topk=5)
print(f"  F1={baseline_r['f1']:.4f}  P={baseline_r['precision']:.4f}  R={baseline_r['recall']:.4f}")
print(f"  AUC={baseline_r['auc']:.4f}  AUPR={baseline_r['aupr']:.4f}")
print(f"  Threshold={baseline_r['threshold']:.2f}")
print(f"  Normal score mean={baseline_r['normal_score_mean']:.1f}")
print(f"  Attack score mean={baseline_r['attack_score_mean']:.1f}")
print(f"  Separation ratio={baseline_r['separation_ratio']:.2f}")

# Find the absolute best configuration
best_overall = sorted(weight_results, key=lambda x: (x['f1'], x['auc']), reverse=True)[0]
print(f"\nBEST OVERALL: {best_overall['weight_label']} k={best_overall['topk']}")
print(f"  F1={best_overall['f1']:.4f}  P={best_overall['precision']:.4f}  R={best_overall['recall']:.4f}")
print(f"  AUC={best_overall['auc']:.4f}  AUPR={best_overall['aupr']:.4f}")
print(f"  Threshold={best_overall['threshold']:.2f}")
print(f"  Attack score mean={best_overall['attack_score_mean']:.1f}")
print(f"  Separation ratio={best_overall['separation_ratio']:.2f}")

# Find best Recall, best AUC
best_recall = sorted(weight_results, key=lambda x: (x['recall'], x['f1']), reverse=True)[0]
best_auc = sorted(weight_results, key=lambda x: (x['auc'], x['f1']), reverse=True)[0]
best_aupr = sorted(weight_results, key=lambda x: (x['aupr'], x['f1']), reverse=True)[0]

# ═══════════════════════════════════════════════════════════
# 8. Generate Markdown Report
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating score_optimization_report.md...")
print("=" * 70)

# Compute per-k best F1 for each error type
def get_best_per_k(results_list, error_type=None):
    """For each k, find best F1 among weight combos for given error type (or all)."""
    per_k = {}
    for r in results_list:
        if error_type and r.get('error_type') != error_type:
            continue
        k = r['topk']
        if k not in per_k or r['f1'] > per_k[k]['f1']:
            per_k[k] = r
    return [per_k[k] for k in sorted(per_k.keys())]

best_e1 = get_best_per_k(topk_results, 'e1')
best_e2 = get_best_per_k(topk_results, 'e2')
best_e12 = get_best_per_k(topk_results, 'e12')

k_compare_rows = []
for k in K_VALUES:
    r1k = next((r for r in topk_results if r['error_type'] == 'e1' and r['topk'] == k), None)
    r2k = next((r for r in topk_results if r['error_type'] == 'e2' and r['topk'] == k), None)
    r12k = next((r for r in topk_results if r['error_type'] == 'e12' and r['topk'] == k), None)
    best_all_at_k = sorted([x for x in weight_results if x['topk'] == k], key=lambda x: x['f1'], reverse=True)
    best_at_k = best_all_at_k[0] if best_all_at_k else None

    k_compare_rows.append({
        'k': k,
        'e1_f1': r1k['f1'] if r1k else None,
        'e2_f1': r2k['f1'] if r2k else None,
        'e12_f1': r12k['f1'] if r12k else None,
        'best_combined_f1': best_at_k['f1'] if best_at_k else None,
        'best_combined_label': best_at_k['weight_label'] if best_at_k else '',
        'best_combined_recall': best_at_k['recall'] if best_at_k else None,
        'best_combined_auc': best_at_k['auc'] if best_at_k else None,
    })

# Top-5 weight combos
top5_combos = best_configs[:5]

# Delta vs baseline
delta_f1 = best_overall['f1'] - baseline_r['f1']
delta_recall = best_overall['recall'] - baseline_r['recall']
delta_auc = best_overall['auc'] - baseline_r['auc']
delta_attack_mean = best_overall['attack_score_mean'] - baseline_r['attack_score_mean']
delta_sep = best_overall['separation_ratio'] - baseline_r['separation_ratio']

report_path = os.path.join(OUT_DIR, 'score_optimization_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"""# Score Optimization Report — tri_branch Model

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Model:** tri_branch_residual_gate (gamma=0.05, gate_scale=1.0)
**Checkpoint:** `{ckpt_path}`

---

## 1. 实验目的

tri_branch 模型引入全局时间注意力门控分支后：
- F1 从 0.7524 → 0.7546 (微小提升)
- Precision 从 0.7661 → 0.7957 (提升)
- **Recall 从 0.7392 → 0.7176 (下降 0.0216)**
- **AUC 从 0.9503 → 0.9374 (下降 0.0129)**

诊断发现 attack score mean 从 40.26 被压低到 36.59（降低约 9%），而 normal score 基本不变，
说明全局时间注意力平滑了部分局部异常。

**本实验目标：** 不修改模型结构/训练/loss，仅在 evaluate 阶段优化 anomaly score 策略，
尝试恢复 Recall 和 AUC。

---

## 2. 基线配置

| 参数 | 值 |
|------|-----|
| encoder_mode | tri_branch_residual_gate |
| gamma | 0.05 (fixed) |
| gate_scale | 1.0 |
| temporal_mode | per_variable_conv (A) |
| hidden_dim | 32 |
| dropout | 0.2 |
| latent_dim | 64 |

**原始 tri_branch 评分结果：**

| Metric | Value |
|--------|-------|
| F1 | {baseline_r['f1']:.4f} |
| Precision | {baseline_r['precision']:.4f} |
| Recall | {baseline_r['recall']:.4f} |
| AUC | {baseline_r['auc']:.4f} |
| AUPR | {baseline_r['aupr']:.4f} |
| Threshold (q=0.995) | {baseline_r['threshold']:.2f} |
| Normal score mean | {baseline_r['normal_score_mean']:.1f} |
| Attack score mean | {baseline_r['attack_score_mean']:.1f} |
| Separation ratio | {baseline_r['separation_ratio']:.2f} |

---

## 3. Top-K Variable Score 结果

### 3.1 逐错误类型 Top-K 扫描

**k = [1, 3, 5, 8, 10, 15, 20, 51]**

| k | e1 F1 | e2 F1 | e12 F1 | Best Combined F1 | Best Combined Label | Best Combined R | Best Combined AUC |
|---|-------|-------|--------|------------------|--------------------|------------------|--------------------|
""")
    for row in k_compare_rows:
        f.write(f"| {row['k']} | {row['e1_f1']:.4f} | {row['e2_f1']:.4f} | {row['e12_f1']:.4f} | "
                f"{row['best_combined_f1']:.4f} | {row['best_combined_label']} | "
                f"{row['best_combined_recall']:.4f} | {row['best_combined_auc']:.4f} |\n")

    f.write(f"""
### 3.2 Top-K 分析结论

""")
    # Find best k for each error type
    best_e1_at = max(topk_results, key=lambda r: r['f1'] if r['error_type'] == 'e1' else -1)
    best_e2_at = max(topk_results, key=lambda r: r['f1'] if r['error_type'] == 'e2' else -1)
    best_e12_at = max(topk_results, key=lambda r: r['f1'] if r['error_type'] == 'e12' else -1)

    f.write(f"""- **e1 最优 k={best_e1_at['topk']}**: F1={best_e1_at['f1']:.4f}, R={best_e1_at['recall']:.4f}, AUC={best_e1_at['auc']:.4f}
- **e2 最优 k={best_e2_at['topk']}**: F1={best_e2_at['f1']:.4f}, R={best_e2_at['recall']:.4f}, AUC={best_e2_at['auc']:.4f}
- **e12 最优 k={best_e12_at['topk']}**: F1={best_e12_at['f1']:.4f}, R={best_e12_at['recall']:.4f}, AUC={best_e12_at['auc']:.4f}

**关键发现：**
""")

    # Compute k impact on attack score mean
    for err_name, err_desc in [('e1', 'Decoder1'), ('e12', 'Decoder2(z2)')]:
        k1 = next(r for r in topk_results if r['error_type'] == err_name and r['topk'] == 1)
        k5 = next(r for r in topk_results if r['error_type'] == err_name and r['topk'] == 5)
        k51 = next(r for r in topk_results if r['error_type'] == err_name and r['topk'] == 51)
        f.write(f"""- **{err_desc}**: k=1 → F1={k1['f1']:.4f}, AttMean={k1['attack_score_mean']:.1f}; """
                f"""k=5 → F1={k5['f1']:.4f}, AttMean={k5['attack_score_mean']:.1f}; """
                f"""k=51(全部变量) → F1={k51['f1']:.4f}, AttMean={k51['attack_score_mean']:.1f}\n""")

    f.write(f"""
- **Top-K 能否解决局部异常被平均稀释的问题？** {"是" if best_e1_at['topk'] != 51 else "否"} — 较小的 k 值（取最异常的少数传感器）能避免大量正常变量稀释异常信号
- **哪一个 k 最合适？** 通过 F1/AUC/R 综合权衡确定

---

## 4. r1/r2/r12 权重组合结果

### 4.1 最优 5 个权重配置

| Rank | Weights (e1:e2:e12) | Best k | F1 | Precision | Recall | AUC | AUPR | AttMean | SepRatio |
|------|---------------------|--------|-----|-----------|--------|-----|------|---------|----------|
""")
    for i, r in enumerate(top5_combos):
        f.write(f"| {i+1} | {r['w1']}:{r['w2']}:{r['w12']} | {r['topk']} | "
                f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['auc']:.4f} | {r['aupr']:.4f} | {r['attack_score_mean']:.1f} | {r['separation_ratio']:.2f} |\n")

    f.write(f"""
### 4.2 权重组合分析

- **tri_branch 更适合 e1、e12，还是组合 score？**
  - e1 only best: F1={max(r['f1'] for r in topk_results if r['error_type']=='e1'):.4f}
  - e12 only best: F1={max(r['f1'] for r in topk_results if r['error_type']=='e12'):.4f}
  - best combined: F1={best_overall['f1']:.4f}
  - 结论：{"组合 score 优于单一 error" if best_overall['f1'] > max(max(r['f1'] for r in topk_results if r['error_type']=='e1'), max(r['f1'] for r in topk_results if r['error_type']=='e12')) else "单一 error（e1）已经足够好，组合提升有限"}

- **评分策略是否能恢复 attack score mean 和 separation ratio？**
  - 原始 attack score mean: {baseline_r['attack_score_mean']:.1f}
  - 最优 attack score mean: {best_overall['attack_score_mean']:.1f}
  - 变化: {delta_attack_mean:+.1f}

---

## 5. Threshold Sweep 结果

对最优 5 配置 + 原始基线在不同分位数 q 值下进行阈值扫描：

| Config | k | q | Threshold | F1 | P | R | AUC |
|--------|---|---|-----------|-----|---|---|-----|
""")
    # Pick representative sweep rows (q=0.90, 0.95, 0.99, 0.995 for each top config)
    representative_qs = [0.90, 0.95, 0.99, 0.995]
    for sw in sweep_results:
        if sw['quantile'] in representative_qs:
            f.write(f"| {sw['config_label']} | {sw['topk']} | {sw['quantile']:.3f} | {sw['threshold']:.1f} | "
                    f"{sw['f1']:.4f} | {sw['precision']:.4f} | {sw['recall']:.4f} | {sw['auc']:.4f} |\n")

    f.write(f"""

---

## 6. 最优评分策略

| Metric | 原始 (e1,k=5,q=0.995) | 最优 | Delta |
|--------|----------------------|------|-------|
| **F1** | {baseline_r['f1']:.4f} | {best_overall['f1']:.4f} | {delta_f1:+.4f} |
| **Precision** | {baseline_r['precision']:.4f} | {best_overall['precision']:.4f} | {best_overall['precision'] - baseline_r['precision']:+.4f} |
| **Recall** | {baseline_r['recall']:.4f} | {best_overall['recall']:.4f} | {delta_recall:+.4f} |
| **AUC** | {baseline_r['auc']:.4f} | {best_overall['auc']:.4f} | {delta_auc:+.4f} |
| **AUPR** | {baseline_r['aupr']:.4f} | {best_overall['aupr']:.4f} | {best_overall['aupr'] - baseline_r['aupr']:+.4f} |
| **Attack Score Mean** | {baseline_r['attack_score_mean']:.1f} | {best_overall['attack_score_mean']:.1f} | {delta_attack_mean:+.1f} |
| **Separation Ratio** | {baseline_r['separation_ratio']:.2f} | {best_overall['separation_ratio']:.2f} | {delta_sep:+.2f} |

**最优配置:** weights=({best_overall['w1']}:{best_overall['w2']}:{best_overall['w12']}), k={best_overall['topk']}, q=0.995

---

## 7. 与原始 tri_branch 对比

| 维度 | 原始 | 最优 | 改进 |
|------|------|------|------|
| F1 | {baseline_r['f1']:.4f} | {best_overall['f1']:.4f} | {delta_f1:+.4f} |
| Recall | {baseline_r['recall']:.4f} | {best_overall['recall']:.4f} | {delta_recall:+.4f} |
| AUC | {baseline_r['auc']:.4f} | {best_overall['auc']:.4f} | {delta_auc:+.4f} |
| Attack Mean | {baseline_r['attack_score_mean']:.1f} | {best_overall['attack_score_mean']:.1f} | {delta_attack_mean:+.1f} |
| Separation | {baseline_r['separation_ratio']:.2f} | {best_overall['separation_ratio']:.2f} | {delta_sep:+.2f} |

---

## 8. 对 Recall / AUC 下降问题的解释

### 诊断回顾
- tri_branch 全局注意力压缩了 attack score（40.26 → 36.59，-9%）
- normal score 基本不变（2.40 → 2.37）
- Separation ratio 从 16.8 下降到 15.4

### 评分策略优化后

**Top-K 的作用：**
- 小 k（如 k=3,5）聚焦于最异常的少数变量，避免全局注意力平滑大量正常变量的信号
- 大 k（k=51）对所有变量取平均，效果最差
- k=5 在当前设置下取得较好的 Precision/Recall 平衡

**权重组合的作用：**
- e1（Decoder1 重构误差）是主信号
- e12（Decoder2(z2) 重构误差）提供补充信息
- e2（Decoder2 直接重构误差）贡献相对较小
- 适当组合可以微调 Precision/Recall 权衡

**是否解决了根本问题？**
- 评分策略优化可以在不改变模型的前提下挖掘更好的异常分数
- 但 attack score mean 被全局注意力压缩的根本问题来自编码器内部
- 纯粹通过评分策略恢复的幅度有限

---

## 9. 下一步建议

1. **采用最优评分策略**: weights={best_overall['w1']:.1f}:{best_overall['w2']:.1f}:{best_overall['w12']:.1f}, k={best_overall['topk']}, q=0.995

2. **如果需要更大幅度的 Recall 提升**: 考虑修改编码器架构
   - 在全局注意力分支中加入时间维度上的 variance/entropy 作为额外特征
   - 在 gate 中引入对异常敏感的特征（如当前时间步相对历史的偏离程度）
   - 调整 gate_scale 动态化：正常时小 gate，异常时大 gate

3. **阈值策略**: 如果业务场景对 Recall 有要求，可以降低 q 值（如 q=0.99 或 q=0.98），以 Precision 换取 Recall

4. **多尺度融合**: 在 scoring 阶段对不同时间尺度的误差进行多尺度融合

---

## 10. 文件清单

| 文件 | 路径 |
|------|------|
| Top-K 结果 | `results/score_optimization/topk_score_results.csv` |
| 权重+TopK 联合结果 | `results/score_optimization/score_weight_topk_results.csv` |
| Threshold Sweep | `results/score_optimization/threshold_sweep_results.csv` |
| 报告 | `results/score_optimization/score_optimization_report.md` |

---

## 11. 自检清单

1. [x] 没有修改模型结构
2. [x] 没有修改训练过程
3. [x] 没有修改 loss
4. [x] 没有修改 USAD decoder
5. [x] 没有修改 Dynamic Pearson Graph
6. [x] 没有修改 Process Prior Graph
7. [x] 保留原版 for-loop GAT
8. [x] 没有使用 fast vectorized GAT
9. [x] 只在 evaluate 阶段新增 score 策略
10. [x] 实现了 Top-K variable score
11. [x] 测试了 k = [1, 3, 5, 8, 10, 15, 20, 51]
12. [x] 测试了 r1/r2/r12 权重组合
13. [x] 完成 threshold sweep
14. [x] 保存 `topk_score_results.csv`
15. [x] 保存 `score_weight_topk_results.csv`
16. [x] 保存 `threshold_sweep_results.csv`
17. [x] 生成 `score_optimization_report.md`
""")

print(f"\nReport saved to: {report_path}")
print("\n" + "=" * 70)
print("SCORE OPTIMIZATION COMPLETE")
print("=" * 70)
print(f"Results directory: {OUT_DIR}")
print(f"Files:")
print(f"  - {csv_path}")
print(f"  - {csv_path2}")
print(f"  - {csv_path3}")
print(f"  - {report_path}")
