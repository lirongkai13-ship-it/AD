"""v5 + USAD Combined Consistency Score — evaluate only, no model changes."""
import sys, os, time, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv, SWaTDynamicWindowDataset, build_labels)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_consistency_score")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 1. Load v5 model
# ============================================================
def load_v5():
    cfg = load_config('config_dev.yaml'); dcfg = cfg['data']
    nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    mfd, _ = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    cc = [c for c in nfd.columns if c in mfd.columns]
    raw = nfd[cc].values.astype(np.float32)
    tr, _, _, _ = split_train_val(raw, None, 0.2)
    tv = StandardScaler().fit_transform(tr)
    sei, _ = build_pearson_edge_index(tv)
    bgp = importlib.util.spec_from_file_location('bpg','models_variants/prior_fusion/build_prior_graph.py')
    bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
    pei, pw = bpgm.build_prior_graph(cc)
    sei = sei.to(device); pei = pei.to(device); pw = pw.to(device)

    from models_variants.tri_branch_v5.variant_model import TriBranch_USAD_v5
    model = TriBranch_USAD_v5(
        nv=len(cc), ws=60, static_edge_index=sei,
        prior_edge_index=pei, prior_weights=pw,
        hidden_dim=32, gat_heads=2, dropout=0.2, latent_dim=64,
        encoder_mode="tri_branch_residual_gate",
        gamma_mode="learnable", gamma_value=0.05, gate_scale=1.0,
    ).to(device)
    ckpt = torch.load('outputs/swat_normal_train_merged_test/tri_branch_v5/best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, sei


# ============================================================
# 2. Build data (dev setting, stride=10)
# ============================================================
cfg = load_config('config_dev.yaml')
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)
print(f"Val: {len(val_ds)} Test: {len(test_ds)}")

model, static_ei = load_v5()
print(f"v5 loaded, gamma={model.encoder.gated_fusion.gamma.item():.4f}")


# ============================================================
# 3. Collect r1, r2, r12
# ============================================================
@torch.no_grad()
def collect_usad_outputs(model, loader):
    """Collect r1, r2, r12 and labels."""
    model.eval()
    r1s, r2s, r12s, lbls = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, r2, r12, _ = model(x, static_ei)
        r1s.append(r1.cpu().numpy()); r2s.append(r2.cpu().numpy())
        r12s.append(r12.cpu().numpy())
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return (np.concatenate(r1s), np.concatenate(r2s), np.concatenate(r12s)), \
           np.concatenate(lbls) if lbls else None

print("Collecting val outputs...")
t0 = time.time()
(r1_v, r2_v, r12_v), _ = collect_usad_outputs(model, val_loader)
print(f"  Val: {r1_v.shape} in {time.time()-t0:.1f}s")

print("Collecting test outputs...")
t0 = time.time()
(r1_t, r2_t, r12_t), test_lbls = collect_usad_outputs(model, test_loader)
print(f"  Test: {r1_t.shape} in {time.time()-t0:.1f}s")


# ============================================================
# 4. USAD Combined Consistency Score
# ============================================================
def usad_consistency_score(r1, r2, x, alpha=1.0, beta=0.5, gamma=0.5,
                           time_pool='max', topk=5, use_iqr=True,
                           iqr_params=None, fit_on=None):
    """
    Args:
        r1, r2: [M, T, F] decoder outputs
        x:      [M, T, F] input
    Returns:
        score_window: [M] anomaly scores
    """
    # Step 1: Reconstruction error per feature
    e1 = np.abs(x - r1)  # [M, T, F]
    e2 = np.abs(x - r2)  # [M, T, F]

    # Mean over features
    e1_mean = e1.mean(axis=-1)  # [M, T]
    e2_mean = e2.mean(axis=-1)  # [M, T]

    # Step 2: Consistency terms
    cons_error_gap = np.abs(e1_mean - e2_mean)          # [M, T]
    cons_output_gap = np.abs(r1 - r2).mean(axis=-1)     # [M, T]

    # Step 3: Combined score (per timestep)
    score = alpha * (e1_mean + e2_mean) + \
            beta * cons_error_gap + \
            gamma * cons_output_gap  # [M, T]

    # Step 4: Time pooling (window-level)
    if time_pool == 'max':
        score_window = score.max(axis=1)  # [M]
    elif time_pool == 'topk':
        k = min(topk, score.shape[1])
        score_window = np.sort(score, axis=1)[:, -k:].mean(axis=1)
    else:  # mean
        score_window = score.mean(axis=1)

    # Step 5: IQR normalization
    if use_iqr:
        if iqr_params is None and fit_on is not None:
            iqr_params = fit_iqr_params(fit_on.reshape(-1, 1))
        if iqr_params is None:
            iqr_params = fit_iqr_params(score_window.reshape(-1, 1))
        score_window = apply_iqr_normalize(
            score_window.reshape(-1, 1), iqr_params).flatten()

    return score_window


# ============================================================
# 5. Evaluate all variants
# ============================================================
def eval_config(label, r1_v, r2_v, x_v, r1_t, r2_t, x_t, test_lbls,
                alpha, beta, gamma, time_pool, topk, q=0.995):
    """Full pipeline: fit on val → threshold → test."""
    # Val scores
    val_score = usad_consistency_score(r1_v, r2_v, x_v,
        alpha, beta, gamma, time_pool, topk, use_iqr=True, fit_on=None)
    iqr_p = fit_iqr_params(val_score.reshape(-1, 1))

    # Test scores with val IQR
    test_score = usad_consistency_score(r1_t, r2_t, x_t,
        alpha, beta, gamma, time_pool, topk, use_iqr=True, iqr_params=iqr_p)

    # Threshold from val
    th = float(np.quantile(val_score, q))
    pred = (test_score > th).astype(int)

    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, test_score))
    aupr = float(average_precision_score(test_lbls, test_score))
    pa_pred = point_adjust(pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
    tp = int(((pred==1)&(test_lbls==1)).sum())
    fp = int(((pred==1)&(test_lbls==0)).sum())
    fn = int(((pred==0)&(test_lbls==1)).sum())

    return {
        'label': label, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'time_pool': time_pool, 'topk': topk, 'q': q, 'th': th,
        'f1': float(f1), 'p': float(pr), 'r': float(rc),
        'auc': auc, 'aupr': aupr,
        'pa_f1': float(pa_f1), 'pa_p': float(pa_pr), 'pa_r': float(pa_rc),
        'tp': tp, 'fp': fp, 'fn': fn,
    }


# Run ablation
x_v = r1_v  # same as input for self-supervised USAD (reconstruction target = x_v)
# Actually, we need the original input x. Let me re-collect with x.
@torch.no_grad()
def collect_x_r1_r2(model, loader):
    model.eval()
    xs_all, r1s, r2s, lbls = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, r2, r12, _ = model(x, static_ei)
        xs_all.append(x.cpu().numpy())
        r1s.append(r1.cpu().numpy())
        r2s.append(r2.cpu().numpy())
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return (np.concatenate(xs_all), np.concatenate(r1s), np.concatenate(r2s)), \
           np.concatenate(lbls) if lbls else None

print("\nRe-collecting with x...")
(x_v, r1_v, r2_v), _ = collect_x_r1_r2(model, val_loader)
(x_t, r1_t, r2_t), test_lbls2 = collect_x_r1_r2(model, test_loader)
print(f"  Val x={x_v.shape} Test x={x_t.shape}")

results = []
for label, alpha, beta, gamma in [
    ("baseline (a*e1+a*e2)", 1.0, 0.0, 0.0),
    ("consistency only (cons_err+cons_out)", 0.0, 0.5, 0.5),
    ("full (a=1,b=0.5,g=0.5)", 1.0, 0.5, 0.5),
]:
    for pool, topk in [("max", 0), ("topk", 5), ("mean", 0)]:
        r = eval_config(f"{label}+{pool}", r1_v, r2_v, x_v, r1_t, r2_t, x_t, test_lbls2,
                       alpha, beta, gamma, pool, topk, q=0.995)
        results.append(r)
        print(f"  {r['label']:50s} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f} AUC={r['auc']:.4f} PA-F1={r['pa_f1']:.4f}")

# Also raw max baseline
val_err = (r1_v - x_v).abs().mean(axis=-1).mean(axis=1)  # [M] mean over T and F
test_err_raw = (r1_t - x_t).abs().mean(axis=-1).mean(axis=1)
# Or use previous raw score definition
val_raw = (r1_v - x_v).abs().mean(axis=-1).max(axis=1)
test_raw = (r1_t - x_t).abs().mean(axis=-1).max(axis=1)
th_raw = float(np.quantile(val_raw, 0.9995))
pred_raw = (test_raw > th_raw).astype(int)
pr, rc, f1_raw, _ = precision_recall_fscore_support(test_lbls2, pred_raw, average='binary', zero_division=0)
auc_raw = float(roc_auc_score(test_lbls2, test_raw))
pa_raw = point_adjust(pred_raw, test_lbls2)
pa_pr, pa_rc, pa_f1_raw, _ = precision_recall_fscore_support(test_lbls2, pa_raw, average='binary', zero_division=0)
results.append({'label': 'raw max (v5 best)', 'f1': float(f1_raw), 'p': float(pr), 'r': float(rc),
                'auc': auc_raw, 'pa_f1': float(pa_f1_raw), 'q': 0.9995, 'th': th_raw,
                'alpha': '-', 'beta': '-', 'gamma': '-', 'time_pool': 'max', 'topk': 1})
print(f"  raw max (v5 best): F1={f1_raw:.4f} P={pr:.4f} R={rc:.4f} AUC={auc_raw:.4f} PA-F1={pa_f1_raw:.4f}")

# Save
csv_path = os.path.join(OUT, 'consistency_score_ablation.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)

# Best
best = max(results, key=lambda r: r['f1'])
print(f"\nBest: {best['label']} F1={best['f1']:.4f}")

save_json({'results': results, 'best': best}, os.path.join(OUT, 'results.json'))
print(f"Done: {OUT}")
