"""raw+max score 
=================================
:
1. q=0.9999 threshold  train/val normal
2. raw+max  score 
3. raw max 
4. top1 
5. normal/attack score 
6. //loss  --  evaluate-only
"""
import os, sys, csv, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index, split_train_val,
                         SWaTDynamicWindowDataset)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score)
from models_variants.tri_branch.variant_model import TriBranch_USAD

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "raw_max_reliability")
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
ensure_dir(OUT_DIR)
ensure_dir(DIAG_DIR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 
# 1. Load config & data (train + val + test)
# 
print("=" * 70)
print("1. Loading config and data")
print("=" * 70)
cfg = load_config('config_dev.yaml')
set_seed(42)

dcfg = cfg['data']
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util

# Use the same data loading pipeline as prepare_data for consistency
from data_loader import read_swat_csv, build_labels

# Read and clean CSVs (ffill/bfill NaN)
nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
mfd, merged_raw_labels = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
common_cols = [c for c in nfd.columns if c in mfd.columns]
nfd = nfd[common_cols]
mfd = mfd[common_cols]

normal_raw = nfd.values.astype(np.float32)
merged_raw = mfd.values.astype(np.float32)

# Split normal into train/val (same as prepare_data)
train_raw, val_raw, _, _ = split_train_val(normal_raw, None, 0.2)

# Build labels
merged_labels = build_labels(merged_raw_labels, normal_label=dcfg.get('normal_label', 'Normal'))

# Fit scaler on train normal only
scaler = StandardScaler()
scaler.fit(train_raw)
train_vals = scaler.transform(train_raw)
val_vals = scaler.transform(val_raw)
test_vals = scaler.transform(merged_raw)

print(f"  train_vals: {train_vals.shape}, has_nan={np.isnan(train_vals).any()}")
print(f"  val_vals: {val_vals.shape}, has_nan={np.isnan(val_vals).any()}")

# Create separate datasets for train/val/test
window_size = int(dcfg["window_size"])
stride_train = int(dcfg.get("train_stride", dcfg.get("stride", 10)))
stride_val = int(dcfg.get("val_stride", dcfg.get("stride", 10)))
stride_test = int(dcfg.get("test_stride", dcfg.get("stride", 5)))

train_ds = SWaTDynamicWindowDataset(train_vals, None, window_size, 1, stride_train, 'future')
val_ds   = SWaTDynamicWindowDataset(val_vals, None, window_size, 1, stride_val, 'future')
test_ds  = SWaTDynamicWindowDataset(test_vals, merged_labels, window_size, 1, stride_test, 'future')

bs = int(cfg['train']['batch_size'])
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, drop_last=False)

print(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows, Test: {len(test_ds)} windows")

# 
# 2. Build graphs & load model
# 
print("\n" + "=" * 70)
print("2. Loading tri_branch model")
print("=" * 70)

static_ei, _ = build_pearson_edge_index(train_vals, corr_threshold=0.3, self_loop=True)

bgp = importlib.util.spec_from_file_location('bpg',
    os.path.join(os.path.dirname(__file__), 'models_variants', 'prior_fusion', 'build_prior_graph.py'))
bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
prior_ei, prior_w = bpgm.build_prior_graph(common_cols)

static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

model = TriBranch_USAD(
    nv=len(common_cols), ws=window_size,
    static_edge_index=static_ei, prior_edge_index=prior_ei, prior_weights=prior_w,
    hidden_dim=32, gat_heads=2, gru_hidden=32, tcn_channels=32, tcn_blocks=1,
    dropout=0.2, latent_dim=64, use_flatten=True,
    temporal_mode="per_variable_conv",
    encoder_mode="tri_branch_residual_gate",
    gamma_mode="fixed", gamma_value=0.05, gate_scale=1.0,
).to(device)

ckpt_path = 'outputs/swat_normal_train_merged_test/tri_branch/best_model.pt'
ckpt = torch.load(ckpt_path, map_location=device)
sd = {k.replace('gated_fusion.gate.', 'gated_fusion.gate_mlp.'): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}")

# 
# 3. Collect e1 errors on train / val / test
# 
print("\n" + "=" * 70)
print("3. Collecting e1 errors (train / val / test)")
print("=" * 70)
t0 = time.time()

@torch.no_grad()
def collect_e1(model, loader):
    """e1 = |r1 - x|.mean(dim=1) -> [M, N] per-variable errors"""
    model.eval()
    errors_list, labels_list = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, _, _ = model(x, static_ei)
        e1 = (r1 - x).abs().mean(dim=1)  # [B, N]
        errors_list.append(e1.cpu().numpy())
        if 'label' in batch:
            labels_list.append(batch['label'].cpu().numpy())
    errors = np.concatenate(errors_list, axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None
    return errors, labels

train_e1, _ = collect_e1(model, train_loader)
val_e1, _   = collect_e1(model, val_loader)
test_e1, test_labels = collect_e1(model, test_loader)

print(f"  Train e1: {train_e1.shape}")
print(f"  Val e1:   {val_e1.shape}")
print(f"  Test e1:  {test_e1.shape}, labels: {len(test_labels)}")
print(f"  Time: {time.time() - t0:.1f}s")

# 
# 4. Compute scores by three methods
# 
print("\n" + "=" * 70)
print("4. Computing scores (3 methods)")
print("=" * 70)

# Method A: IQR + k=5 (original baseline)
iqr5_params = fit_iqr_params(train_e1)  # fit on TRAIN normal only
val_iqr5_norm = apply_iqr_normalize(val_e1, iqr5_params)
test_iqr5_norm = apply_iqr_normalize(test_e1, iqr5_params)
val_iqr5_score = aggregate_topk_score(val_iqr5_norm, topk=5)
test_iqr5_score = aggregate_topk_score(test_iqr5_norm, topk=5)

# Method B: IQR + k=1
val_iqr1_score = aggregate_topk_score(val_iqr5_norm, topk=1)  # same normalization
test_iqr1_score = aggregate_topk_score(test_iqr5_norm, topk=1)

# Method C: raw + max (top1 variable, no IQR)
val_rawmax_score = val_e1.max(axis=1)
train_rawmax_score = train_e1.max(axis=1)
test_rawmax_score = test_e1.max(axis=1)

print(f"  IQR+k=5:   val range [{val_iqr5_score.min():.2f}, {val_iqr5_score.max():.2f}]")
print(f"  IQR+k=1:   val range [{val_iqr1_score.min():.2f}, {val_iqr1_score.max():.2f}]")
print(f"  raw+max:    val range [{val_rawmax_score.min():.4f}, {val_rawmax_score.max():.4f}]")

# 
# 5. Thresholds from TRAIN normal vs VAL normal
# 
print("\n" + "=" * 70)
print("5. Threshold computation (train-normal vs val-normal)")
print("=" * 70)

QUANTILES = [0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
             0.985, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999]

def get_thresholds(score, source_name):
    """Compute thresholds at each quantile from the given score distribution."""
    return {q: float(np.quantile(score, q)) for q in QUANTILES}

# Thresholds from train normal and val normal
train_th_rawmax = get_thresholds(train_rawmax_score, "train")
val_th_rawmax = get_thresholds(val_rawmax_score, "val")
train_th_iqr5 = get_thresholds(val_iqr5_score, "train")  # val=IQR fitted; use train normal scores
val_th_iqr5 = get_thresholds(val_iqr5_score, "val")

print(f"\n{'q':>9s}  {'Train th':>10s}  {'Val th':>10s}  {'Delta':>10s}")
print(f"{''*45}")
for q in [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999]:
    dt = train_th_rawmax[q]
    dv = val_th_rawmax[q]
    print(f"{q:9.4f}  {dt:10.4f}  {dv:10.4f}  {(dv-dt):+10.4f}")

# 
# 6. Evaluate all methods at all thresholds
# 
print("\n" + "=" * 70)
print("6. Full evaluation across methods x thresholds")
print("=" * 70)

def eval_at_threshold(score, labels, threshold):
    pred = (score > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    pr, rc, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    auc_val = float(roc_auc_score(labels, score))
    aupr_val = float(average_precision_score(labels, score))
    return {
        'precision': float(pr), 'recall': float(rc), 'f1': float(f1),
        'auc': auc_val, 'aupr': aupr_val,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }

results = []

# Evaluate each method with train and val thresholds
for method_name, test_score, th_dict, th_source in [
    ("IQR+k=5 (train_th)", test_iqr5_score, train_th_iqr5, "train_normal"),
    ("IQR+k=5 (val_th)",   test_iqr5_score, val_th_iqr5,   "val_normal"),
    ("IQR+k=1 (train_th)", test_iqr1_score, train_th_iqr5, "train_normal"),
    ("IQR+k=1 (val_th)",   test_iqr1_score, val_th_iqr5,   "val_normal"),
    ("raw+max (train_th)", test_rawmax_score, train_th_rawmax, "train_normal"),
    ("raw+max (val_th)",   test_rawmax_score, val_th_rawmax,   "val_normal"),
]:
    for q in QUANTILES:
        th = th_dict[q]
        r = eval_at_threshold(test_score, test_labels, th)
        r['method'] = method_name
        r['q'] = q
        r['threshold'] = th
        r['th_source'] = th_source
        results.append(r)

# Also test-sweep (for comparison only, clearly marked as non-reportable)
print("  (also computing test-sweep for diagnostic comparison only)")
for method_name, test_score in [
    ("IQR+k=5 (test_sweep)", test_iqr5_score),
    ("IQR+k=1 (test_sweep)", test_iqr1_score),
    ("raw+max (test_sweep)", test_rawmax_score),
]:
    for q in QUANTILES:
        th = float(np.quantile(test_score, q))  #  FROM TEST -- NOT REPORTABLE
        r = eval_at_threshold(test_score, test_labels, th)
        r['method'] = method_name
        r['q'] = q
        r['threshold'] = th
        r['th_source'] = "TEST_SWEEP_[!]_NOT_REPORTABLE"
        results.append(r)

# 
# 7. Find key configurations
# 
print("\n" + "=" * 70)
print("7. Key results")
print("=" * 70)

# Separate reportable (train/val th) from test-sweep
trainval_results = [r for r in results if 'test_sweep' not in r['method']]
test_sweep_results = [r for r in results if 'test_sweep' in r['method']]

# Best F1 per method (reportable only)
methods_reportable = sorted(set(r['method'] for r in trainval_results))
print(f"\n{'Method':<25s}  {'q':>9s}  {'th':>10s}  {'F1':>8s}  {'P':>8s}  {'R':>8s}  {'AUC':>8s}")
print(f"{''*85}")
for m in methods_reportable:
    m_results = [r for r in trainval_results if r['method'] == m]
    best = max(m_results, key=lambda r: r['f1'])
    print(f"{m:<25s}  {best['q']:9.4f}  {best['threshold']:10.4f}  "
          f"{best['f1']:8.4f}  {best['precision']:8.4f}  {best['recall']:8.4f}  {best['auc']:8.4f}")

print(f"\n{''*85}")
print("[WARNING] TEST-SWEEP (not reportable - for diagnostic comparison only):")
for m in ['IQR+k=5 (test_sweep)', 'IQR+k=1 (test_sweep)', 'raw+max (test_sweep)']:
    m_results = [r for r in test_sweep_results if r['method'] == m]
    best = max(m_results, key=lambda r: r['f1'])
    print(f"  {m:<25s}  q={best['q']:.4f}  th={best['threshold']:.4f}  "
          f"F1={best['f1']:.4f}  P={best['precision']:.4f}  R={best['recall']:.4f}")

# 
# 8. Score distribution analysis
# 
print("\n" + "=" * 70)
print("8. Score distribution analysis")
print("=" * 70)

t_normal_mask = (test_labels == 0)
t_attack_mask = (test_labels == 1)

dist_stats = []
for name, score in [("IQR+k=5", test_iqr5_score), ("IQR+k=1", test_iqr1_score), ("raw+max", test_rawmax_score)]:
    ns = score[t_normal_mask]
    as_ = score[t_attack_mask]
    sep = as_.mean() / ns.mean() if ns.mean() > 0 else float('inf')
    # Overlap: fraction of attack scores below train-derived q=0.9999 threshold
    th_train = train_th_rawmax[0.9999] if name == "raw+max" else train_th_iqr5[0.9999]
    attack_below_th = (as_ < th_train).mean()

    dist_stats.append({
        'method': name,
        'normal_mean': float(ns.mean()), 'normal_median': float(np.median(ns)),
        'normal_std': float(ns.std()), 'normal_p99': float(np.percentile(ns, 99)),
        'attack_mean': float(as_.mean()), 'attack_median': float(np.median(as_)),
        'attack_std': float(as_.std()), 'attack_p99': float(np.percentile(as_, 99)),
        'separation_ratio': float(sep),
        'attack_below_train_th_9999': float(attack_below_th),
    })
    print(f"  {name}: sep_ratio={sep:.2f}  normal_mean={ns.mean():.3f}  attack_mean={as_.mean():.2f}")

# 
# 9. Top1 variable frequency (raw+max)
# 
print("\n" + "=" * 70)
print("9. Top1 variable frequency analysis (raw+max)")
print("=" * 70)

# For each test sample, which variable has max error?
top1_var_idx = np.argmax(test_e1, axis=1)  # [M_test]
n_vars = test_e1.shape[1]

# Frequency
var_freq = np.bincount(top1_var_idx, minlength=n_vars)
var_freq_normal = np.bincount(top1_var_idx[t_normal_mask], minlength=n_vars)
var_freq_attack = np.bincount(top1_var_idx[t_attack_mask], minlength=n_vars)

# Top-10 variables overall
top10_idx = np.argsort(var_freq)[::-1][:10]
print(f"\n  Top-10 variables (all):")
print(f"  {'Idx':>4s}  {'Name':>10s}  {'All':>8s}  {'Normal':>8s}  {'Attack':>8s}  {'A/N ratio':>10s}")
for idx in top10_idx:
    n_norm = var_freq_normal[idx] / max(1, t_normal_mask.sum())
    n_attk = var_freq_attack[idx] / max(1, t_attack_mask.sum())
    ratio = n_attk / max(1e-8, n_norm)
    print(f"  {idx:4d}  {common_cols[idx]:>10s}  {var_freq[idx]:8d}  "
          f"{var_freq_normal[idx]:8d}  {var_freq_attack[idx]:8d}  {ratio:10.2f}")

# Scale domination risk: does one variable dominate >50% of samples?
dom_ratio = var_freq.max() / var_freq.sum()
print(f"\n  Max single-variable frequency ratio: {dom_ratio:.2%} "
      f"({'[!] SCALE DOMINATION RISK' if dom_ratio > 0.5 else '[OK] No single-variable domination'})")

# 
# 10. Per-variable error scale statistics
# 
print("\n" + "=" * 70)
print("10. Per-variable error scale statistics")
print("=" * 70)

var_stats = []
for i in range(n_vars):
    ns = test_e1[t_normal_mask, i]
    as_ = test_e1[t_attack_mask, i]
    ratio = as_.mean() / ns.mean() if ns.mean() > 0 else float('inf')
    var_stats.append({
        'node_index': i,
        'node_name': common_cols[i],
        'normal_mean': float(ns.mean()), 'attack_mean': float(as_.mean()),
        'ratio': float(ratio),
        'normal_p99': float(np.percentile(ns, 99)),
        'attack_p99': float(np.percentile(as_, 99)),
        'normal_max': float(ns.max()), 'attack_max': float(as_.max()),
    })

# Sort by ratio (descending -- best separation variables)
var_stats_sorted = sorted(var_stats, key=lambda x: x['ratio'], reverse=True)
print(f"\n  Top-15 variables by attack/normal ratio:")
print(f"  {'Idx':>4s}  {'Name':>10s}  {'NormalMean':>12s}  {'AttackMean':>12s}  {'Ratio':>8s}  {'NormalP99':>12s}  {'AttackP99':>12s}")
for vs in var_stats_sorted[:15]:
    print(f"  {vs['node_index']:4d}  {vs['node_name']:>10s}  {vs['normal_mean']:12.4f}  "
          f"{vs['attack_mean']:12.4f}  {vs['ratio']:8.2f}  "
          f"{vs['normal_p99']:12.4f}  {vs['attack_p99']:12.4f}")

# Check: are top ratio variables also the top1 frequency variables?
print(f"\n  Overlap between top-ratio and top-frequency variables:")
top_ratio_idx = set(vs['node_index'] for vs in var_stats_sorted[:10])
top_freq_idx = set(int(i) for i in top10_idx)
overlap = top_ratio_idx & top_freq_idx
print(f"  Common variables: {len(overlap)}/10 -- {[common_cols[i] for i in overlap]}")

# 
# 11. Save all CSVs
# 
print("\n" + "=" * 70)
print("11. Saving CSVs")
print("=" * 70)

# Main results
csv1 = os.path.join(OUT_DIR, 'threshold_sweep_all_methods.csv')
with open(csv1, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader()
    w.writerows(results)
print(f"  {csv1} ({len(results)} rows)")

# Score distribution
csv2 = os.path.join(OUT_DIR, 'score_distribution_stats.csv')
with open(csv2, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=dist_stats[0].keys())
    w.writeheader()
    w.writerows(dist_stats)
print(f"  {csv2}")

# Top1 variable frequency
csv3 = os.path.join(OUT_DIR, 'top1_variable_frequency.csv')
with open(csv3, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['node_index', 'node_name', 'freq_all', 'freq_normal', 'freq_attack', 'freq_all_pct', 'freq_normal_pct', 'freq_attack_pct'])
    w.writeheader()
    for i in range(n_vars):
        w.writerow({
            'node_index': i, 'node_name': common_cols[i],
            'freq_all': int(var_freq[i]), 'freq_normal': int(var_freq_normal[i]), 'freq_attack': int(var_freq_attack[i]),
            'freq_all_pct': round(float(var_freq[i]) / max(1, len(test_labels)) * 100, 2),
            'freq_normal_pct': round(float(var_freq_normal[i]) / max(1, t_normal_mask.sum()) * 100, 2),
            'freq_attack_pct': round(float(var_freq_attack[i]) / max(1, t_attack_mask.sum()) * 100, 2),
        })
print(f"  {csv3}")

# Variable error scale
csv4 = os.path.join(OUT_DIR, 'variable_error_scale_stats.csv')
with open(csv4, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=var_stats[0].keys())
    w.writeheader()
    w.writerows(var_stats_sorted)
print(f"  {csv4}")

# 
# 12. Plots
# 
print("\n" + "=" * 70)
print("12. Generating diagnostic plots")
print("=" * 70)

# Colors
C_RAW = '#E53935'
C_IQR5 = '#1E88E5'
C_IQR1 = '#43A047'
C_THRESH = '#FB8C00'

# --- Plot 1: raw vs IQR score distribution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, name, score in [
    (ax1, "raw+max", test_rawmax_score),
    (ax2, "IQR+k=5", test_iqr5_score),
]:
    ns = score[t_normal_mask]
    as_ = score[t_attack_mask]
    bins = np.linspace(min(ns.min(), as_.min()), min(max(ns.max(), as_.max()), np.percentile(as_, 99.9)), 80)
    ax.hist(ns, bins=bins, alpha=0.6, label=f'Normal (n={len(ns)})', color='#4CAF50', density=True)
    ax.hist(as_, bins=bins, alpha=0.6, label=f'Attack (n={len(as_)})', color='#F44336', density=True)
    # Threshold lines
    for q_val, ls, lbl in [(0.996, '--', 'q=0.996'), (0.999, '-.', 'q=0.999'), (0.9999, ':', 'q=0.9999')]:
        if name == "raw+max":
            th = val_th_rawmax[q_val]
        else:
            th = val_th_iqr5[q_val]
        ax.axvline(th, color='black', linestyle=ls, alpha=0.5, label=f'{lbl} th={th:.2f}')
    ax.set_xlabel('Score'); ax.set_ylabel('Density'); ax.set_title(name)
    ax.legend(fontsize=7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

fig.suptitle('Score Distribution: raw+max vs IQR+k=5', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, 'raw_vs_iqr_score_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  raw_vs_iqr_score_distribution.png")

# --- Plot 2: raw+max normal/attack histogram with thresholds ---
fig, ax = plt.subplots(figsize=(11, 4.5))
ns = test_rawmax_score[t_normal_mask]
as_ = test_rawmax_score[t_attack_mask]
bins = np.linspace(0, np.percentile(as_, 99.5), 100)
ax.hist(ns, bins=bins, alpha=0.5, label=f'Normal (n={len(ns)})', color='#4CAF50', density=True)
ax.hist(as_, bins=bins, alpha=0.5, label=f'Attack (n={len(as_)})', color='#F44336', density=True)
for q_val, ls, color in [(0.996, '--', '#FF6F00'), (0.999, '-.', '#E65100'), (0.9999, ':', '#BF360C')]:
    th_train = train_th_rawmax[q_val]
    th_val = val_th_rawmax[q_val]
    ax.axvline(th_val, color=color, linestyle=ls, alpha=0.7, linewidth=1.5,
               label=f'val q={q_val} th={th_val:.2f}')
ax.set_xlabel('raw+max Score'); ax.set_ylabel('Density')
ax.set_title('raw+max Score Distribution with Threshold Lines')
ax.legend(fontsize=7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, 'raw_max_normal_attack_hist.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  raw_max_normal_attack_hist.png")

# --- Plot 3: threshold q compare (train vs val vs test F1) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax, method_key, label in [
    (ax1, "raw+max (val_th)", "raw+max (val threshold)"),
    (ax2, "IQR+k=5 (val_th)", "IQR+k=5 (val threshold)"),
]:
    m_results = sorted([r for r in trainval_results if r['method'] == method_key], key=lambda r: r['q'])
    qs = [r['q'] for r in m_results]
    ax.plot(qs, [r['f1'] for r in m_results], 'o-', color=C_RAW if 'raw' in method_key else C_IQR5,
            label='F1', markersize=3)
    ax.plot(qs, [r['precision'] for r in m_results], 's--', color='#1E88E5', label='P', markersize=3, alpha=0.7)
    ax.plot(qs, [r['recall'] for r in m_results], '^--', color='#43A047', label='R', markersize=3, alpha=0.7)
    ax.set_xlabel('Quantile q'); ax.set_ylabel('Score'); ax.set_title(label)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.suptitle('Threshold Quantile Impact on F1/P/R', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, 'raw_max_threshold_q_compare.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  raw_max_threshold_q_compare.png")

# --- Plot 4: top1 variable frequency (attack) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# Attack
top15_attack_idx = np.argsort(var_freq_attack)[::-1][:15]
ax1.barh(range(15), [var_freq_attack[i] for i in top15_attack_idx][::-1],
         color='#F44336', edgecolor='white')
ax1.set_yticks(range(15))
ax1.set_yticklabels([common_cols[i] for i in top15_attack_idx][::-1], fontsize=8)
ax1.set_xlabel('Frequency'); ax1.set_title('Top1 Variable Frequency -- Attack Samples')
# Normal
top15_normal_idx = np.argsort(var_freq_normal)[::-1][:15]
ax2.barh(range(15), [var_freq_normal[i] for i in top15_normal_idx][::-1],
         color='#4CAF50', edgecolor='white')
ax2.set_yticks(range(15))
ax2.set_yticklabels([common_cols[i] for i in top15_normal_idx][::-1], fontsize=8)
ax2.set_xlabel('Frequency'); ax2.set_title('Top1 Variable Frequency -- Normal Samples')
fig.suptitle('Top1 Variable Frequency Distribution', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, 'top1_variable_frequency.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  top1_variable_frequency.png")

# --- Plot 5: variable error scale ratio ---
fig, ax = plt.subplots(figsize=(14, 5))
top30 = var_stats_sorted[:30]
names = [vs['node_name'] for vs in reversed(top30)]
ratios = [vs['ratio'] for vs in reversed(top30)]
colors_bar = ['#F44336' if r > 15 else '#FF9800' if r > 5 else '#4CAF50' for r in ratios]
ax.barh(range(len(names)), ratios, color=colors_bar, edgecolor='white')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7)
ax.set_xlabel('Attack/Normal Mean Ratio')
ax.set_title('Variable Error Scale Separation (AttackMean / NormalMean)')
ax.axvline(10, color='gray', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(DIAG_DIR, 'variable_error_scale_ratio.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("  variable_error_scale_ratio.png")

# 
# 13. Generate Markdown Report
# 
print("\n" + "=" * 70)
print("13. Generating report")
print("=" * 70)

# Compute key numbers for report
best_rawmax_trainval = max([r for r in trainval_results if r['method'] == 'raw+max (val_th)'], key=lambda r: r['f1'])
best_iqr5_trainval = max([r for r in trainval_results if r['method'] == 'IQR+k=5 (val_th)'], key=lambda r: r['f1'])
best_rawmax_testsweep = max([r for r in test_sweep_results if r['method'] == 'raw+max (test_sweep)'], key=lambda r: r['f1'])

# recommended q from val threshold
rec_rawmax = [r for r in trainval_results if r['method'] == 'raw+max (val_th)' and r['f1'] >= 0.755]
if rec_rawmax:
    rec_q = max(rec_rawmax, key=lambda r: r['recall'])
else:
    rec_q = best_rawmax_trainval

report = os.path.join(OUT_DIR, 'raw_max_score_reliability_report.md')
with open(report, 'w', encoding='utf-8') as f:
    f.write(f"""# raw+max Score -- 

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Model:** tri_branch_residual_gate (gamma=0.05, gate_scale=1.0)

---

## 1. 

 threshold sweep 
- `e1 raw + max + q=0.9999` -> F1=0.8260, P=0.9372, R=0.7383
-  `IQR+k=5` (F1=0.7546)  `IQR+k=1` (F1=0.7587)


1. q=0.9999  train/val normal test-set tuning
2. raw+max  score 
3. raw max 
4. attack top1 

---

## 2. 

| Parameter | Value |
|-----------|-------|
| Model | tri_branch_residual_gate |
| gamma | 0.05 (fixed) |
| gate_scale | 1.0 |
| temporal_mode | per_variable_conv |
| Score | e1 raw max (no IQR) |
| Checkpoint | `tri_branch/best_model.pt` |

---

## 3. Score 

| Method | Normal Mean | Attack Mean | Separation Ratio | Attack below train_th(q=0.9999) |
|--------|-------------|-------------|------------------|----------------------------------|
""")
    for ds in dist_stats:
        f.write(f"| {ds['method']} | {ds['normal_mean']:.4f} | {ds['attack_mean']:.2f} | {ds['separation_ratio']:.2f} | {ds['attack_below_train_th_9999']:.2%} |\n")

    f.write(f"""

### 3.1 

- **raw+max  {'>' if dist_stats[-1]['separation_ratio'] > dist_stats[0]['separation_ratio'] else '<'} IQR**: "
            f"raw+max={dist_stats[-1]['separation_ratio']:.2f} vs IQR+k=5={dist_stats[0]['separation_ratio']:.2f}
- IQR  scale **/**
- raw max 

---

## 4. 

|  | q | Threshold Source | F1 | P | R | Reportable? |
|------|---|------------------|-----|----|-----|-------------|
| IQR+k=5 (val_th) | {best_iqr5_trainval['q']:.4f} | val_normal | {best_iqr5_trainval['f1']:.4f} | {best_iqr5_trainval['precision']:.4f} | {best_iqr5_trainval['recall']:.4f} | [OK] |
| raw+max (val_th) | {best_rawmax_trainval['q']:.4f} | val_normal | {best_rawmax_trainval['f1']:.4f} | {best_rawmax_trainval['precision']:.4f} | {best_rawmax_trainval['recall']:.4f} | [OK] |
| raw+max (test_sweep) | {best_rawmax_testsweep['q']:.4f} | [!] TEST | {best_rawmax_testsweep['f1']:.4f} | {best_rawmax_testsweep['precision']:.4f} | {best_rawmax_testsweep['recall']:.4f} |  |

### 4.1 q=0.9999 

 F1=0.8260  **test set quantile** train/val normal 

 **val normal** 
- q=0.9999 (val th={val_th_rawmax[0.9999]:.4f}): F1={best_rawmax_trainval['f1']:.4f}

**q=0.9999 + test sweep  0.8260 **

---

## 5.  val normal

 val normal  test label

| q (val) | Threshold | F1 | P | R | AUC |
|----------|-----------|-----|----|-----|-----|
""")
    for q in [0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9999]:
        r = [x for x in trainval_results if x['method'] == 'raw+max (val_th)' and abs(x['q'] - q) < 0.0001]
        if r:
            r = r[0]
            f.write(f"| {q:.4f} | {r['threshold']:.4f} | {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['auc']:.4f} |\n")

    f.write(f"""

### 5.1  q 

**q={rec_q['q']:.4f} (val normal threshold)**

- F1: {rec_q['f1']:.4f}
- Precision: {rec_q['precision']:.4f}
- Recall: {rec_q['recall']:.4f}
- AUC: {rec_q['auc']:.4f}

 vs IQR+k=5 (val_th):
- F1 : {rec_q['f1'] - best_iqr5_trainval['f1']:+.4f}
- Recall : {rec_q['recall'] - best_iqr5_trainval['recall']:+.4f}

---

## 6. Top1 

### 6.1 Attack  Top1 

| Rank | Variable | Attack Freq | Attack % | Normal % | A/N Ratio |
|------|----------|-------------|----------|----------|-----------|
""")
    top15_attack = np.argsort(var_freq_attack)[::-1][:15]
    for rank, idx in enumerate(top15_attack):
        a_pct = var_freq_attack[idx] / max(1, t_attack_mask.sum()) * 100
        n_pct = var_freq_normal[idx] / max(1, t_normal_mask.sum()) * 100
        ratio = a_pct / max(1e-8, n_pct)
        f.write(f"| {rank+1} | {common_cols[idx]} | {var_freq_attack[idx]} | {a_pct:.1f}% | {n_pct:.1f}% | {ratio:.1f}x |\n")

    f.write(f"""

### 6.2 Scale Domination 

- : {dom_ratio:.1%}
- : {'[!] **** --  top1' if dom_ratio > 0.5 else '[OK] '}

### 6.3 

""")
    if dom_ratio > 0.5:
        f.write(f"- raw max ****\n")
    else:
        f.write(f"- raw max ****\n")

    f.write(f"""
- Normal  attack  top1 {'' if len(set(top15_attack) & set(top15_normal_idx)) < 10 else ''}

---

## 7. 

### 7.1 attack/normal ratio  Top-15

| Rank | Variable | Normal Mean | Attack Mean | Ratio | Normal P99 | Attack P99 |
|------|----------|-------------|-------------|-------|------------|------------|
""")
    for rank, vs in enumerate(var_stats_sorted[:15]):
        f.write(f"| {rank+1} | {vs['node_name']} | {vs['normal_mean']:.4f} | {vs['attack_mean']:.4f} | {vs['ratio']:.1f}x | {vs['normal_p99']:.4f} | {vs['attack_p99']:.4f} |\n")

    f.write(f"""

### 7.2 Top-ratio vs Top-frequency 

- : {len(overlap)}/10
- : {[common_cols[i] for i in overlap]}

---

## 8. 

### 8.1 `raw+max`  `IQR+k=5`
**{'' if best_rawmax_trainval['f1'] > best_iqr5_trainval['f1'] else ''}** --
raw+max(val_th) F1={best_rawmax_trainval['f1']:.4f} vs IQR+k=5(val_th) F1={best_iqr5_trainval['f1']:.4f}

### 8.2  attack score separation
{'****' if dist_stats[-1]['separation_ratio'] > dist_stats[0]['separation_ratio'] else '****'} --
raw+max separation={dist_stats[-1]['separation_ratio']:.2f} vs IQR+k=5={dist_stats[0]['separation_ratio']:.2f}

### 8.3 q=0.9999  train/val normal
 val_normal  train_normal test_sweep  0.8260****

### 8.4  test-set threshold tuning 
 F1=0.8260  test quantile ->  test-set tuning  val normal quantile

### 8.5 raw max 
{'[!]  -- ' if dom_ratio > 0.5 else '[OK]  -- '}

### 8.6 Attack top1 
 SWaT  LITPFIT 

---

## 9. 

### 9.1  Score 

```
score = e1 raw max (top1 variable, no IQR)
threshold q = {rec_q['q']:.4f} (from VAL normal)
```

### 9.2 

| Setting | q | Threshold | F1 | P | R | AUC |
|---------|---|-----------|-----|----|-----|-----|
| **Recommended (val_th)** | {rec_q['q']:.4f} | {rec_q['threshold']:.4f} | {rec_q['f1']:.4f} | {rec_q['precision']:.4f} | {rec_q['recall']:.4f} | {rec_q['auc']:.4f} |

### 9.3 Full setting 

1. **`e1 raw max` with val normal q** -- 
2. **`IQR+k=5` with val normal q={best_iqr5_trainval['q']:.4f}** -- 
3.  full setting (stride=1)  val normal threshold 

### 9.4 

```
 score: score = e1 raw max
 q: q = {rec_q['q']:.4f} (val normal quantile)
```

****: q=0.9999  test-sweep F1=0.8260
 val normal threshold 

---

## 10. 

1. [x] 
2. [x] 
3. [x]  tri_branch
4. [x]  gamma=0.05
5. [x]  gate_scale=1.0
6. [x]  e1  score 
7. [x]  IQR+k=5 
8. [x]  IQR+k=1 
9. [x]  raw+max
10. [x]  train normal threshold / val normal threshold / test sweep threshold
11. [x]  test sweep 
12. [x]  top1 
13. [x] 
14. [x]  CSV
15. [x]  `raw_max_score_reliability_report.md`

---

## 11. 

| File | Path |
|------|------|
| Full sweep results | `threshold_sweep_all_methods.csv` |
| Score distribution stats | `score_distribution_stats.csv` |
| Top1 variable frequency | `top1_variable_frequency.csv` |
| Variable error scale | `variable_error_scale_stats.csv` |
| Raw vs IQR distribution plot | `diagnostics/raw_vs_iqr_score_distribution.png` |
| Raw max histogram | `diagnostics/raw_max_normal_attack_hist.png` |
| Threshold q comparison | `diagnostics/raw_max_threshold_q_compare.png` |
| Top1 var frequency plot | `diagnostics/top1_variable_frequency.png` |
| Variable error scale plot | `diagnostics/variable_error_scale_ratio.png` |
| Report | `raw_max_score_reliability_report.md` |
""")

print(f"Report: {report}")
print("\n" + "=" * 70)
print("RELIABILITY VALIDATION COMPLETE")
print("=" * 70)
print(f"Results: {OUT_DIR}")
