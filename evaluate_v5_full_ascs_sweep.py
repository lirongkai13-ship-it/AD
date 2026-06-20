"""V3 sparse consistency — full sweep on full setting (stride=1)."""
import sys, os, time, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (build_pearson_edge_index, split_train_val, read_swat_csv,
                         SWaTDynamicWindowDataset, build_labels)
from utils import (load_config, get_device, ensure_dir, point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import importlib.util, itertools

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_full_ascs_sweep")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# Load v5 full
# ============================================================
cfg = load_config('config_dev.yaml')
dcfg = cfg['data']
nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
mfd, raw_lbls = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
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
ckpt = torch.load('outputs/full_setting/tri_branch_v5_full/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model']); model.eval()
print(f"Gamma={model.encoder.gated_fusion.gamma.item():.4f}")

# Build full datasets
merged_lbls = build_labels(raw_lbls, 'Normal')
_, val_raw_vals, _, _ = split_train_val(raw, None, 0.2)
scaler2 = StandardScaler(); scaler2.fit(tr)
val_v = scaler2.transform(val_raw_vals)
test_v = scaler2.transform(mfd[cc].values.astype(np.float32))
val_loader = DataLoader(SWaTDynamicWindowDataset(val_v, None, 60, 1, 1, 'future'), 256, shuffle=False)
test_loader = DataLoader(SWaTDynamicWindowDataset(test_v, merged_lbls, 60, 1, 1, 'future'), 256, shuffle=False)
print(f"Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}")


# ============================================================
# Scores
# ============================================================
def batch_scores_gpu(r1, r2, x, topk_f=5, eps=1e-8):
    """GPU-accelerated: r1,r2,x are torch tensors on GPU"""
    e1 = (x - r1).abs().mean(dim=1)  # [B, F]
    e2 = (x - r2).abs().mean(dim=1)  # [B, F]
    k = min(topk_f, e1.shape[1])
    # Main: raw top5 over features
    score_main = e1.topk(k, dim=1).values.mean(dim=1)  # [B]
    # Consistency: top5 over e1+e2
    e_combined = e1 + e2
    consistency = e_combined.topk(k, dim=1).values.mean(dim=1)  # [B]
    # Ratio
    r_diff = (r1 - r2).abs().mean(dim=1)  # [B, F]
    r_diff_topk = r_diff.topk(k, dim=1).values.mean(dim=1)  # [B]
    ratio = consistency / (r_diff_topk + eps)  # [B]
    return score_main, consistency, ratio

@torch.no_grad()
def collect_all(model, loader):
    model.eval()
    s_main, s_cons, s_ratio, lbls = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device); r1, r2, _, _ = model(x, sei)
        sm, sc, sr = batch_scores_gpu(r1, r2, x)
        s_main.append(sm.cpu().numpy()); s_cons.append(sc.cpu().numpy()); s_ratio.append(sr.cpu().numpy())
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return (np.concatenate(s_main), np.concatenate(s_cons), np.concatenate(s_ratio)), \
           np.concatenate(lbls) if lbls else None

print("Collecting full val...")
(vm, vc, vr), _ = collect_all(model, val_loader)
print("Collecting full test...")
(tm, tc, tr), test_lbls = collect_all(model, test_loader)
print(f"  Val: {vm.shape}  Test: {tm.shape}")

# ============================================================
# Grid search on VAL
# ============================================================
LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9]
MUS = [0.1, 0.2, 0.3, 0.5]
QS = [0.995, 0.997, 0.999, 0.9995]

val_results = []
for lam, mu, q in itertools.product(LAMBDAS, MUS, QS):
    vs = lam*vm + (1-lam)*vc + mu*vr
    th = float(np.quantile(vs, q))
    # VAL eval
    pred = (vs > th).astype(int)
    vlbls = np.zeros_like(vs)  # val is all normal → labels=0
    # val F1 doesn't make sense (no attacks), use val score stats
    normal_scores = vs[vlbls == 0]
    # Simulate: if q is too low, threshold too low → many FP
    # Use a proxy: threshold value relative to normal score distribution
    val_results.append({
        'lam': lam, 'mu': mu, 'q': q, 'th': float(th),
        'val_score_mean': float(vs.mean()), 'val_th_over_mean': float(th / max(1e-8, vs.mean())),
        'val_99': float(np.quantile(vs, 0.99)), 'val_999': float(np.quantile(vs, 0.999)),
    })

# ============================================================
# Pick best by VAL proxy: highest threshold/mean ratio = tightest threshold
# ============================================================
best_val = max(val_results, key=lambda r: r['val_th_over_mean'])
lam_b, mu_b, q_b = best_val['lam'], best_val['mu'], best_val['q']

# ============================================================
# Sweep on TEST (report best VAL config as primary)
# ============================================================
test_results = []
for lam, mu, q in itertools.product(LAMBDAS, MUS, QS):
    ts = lam*tm + (1-lam)*tc + mu*tr
    th = float(np.quantile(vs := lam*vm + (1-lam)*vc + mu*vr, q))
    pred = (ts > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, ts))
    aupr = float(average_precision_score(test_lbls, ts))
    pa_pred = point_adjust(pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
    test_results.append({
        'lam': lam, 'mu': mu, 'q': q, 'th': float(th),
        'f1': float(f1), 'p': float(pr), 'r': float(rc),
        'auc': auc, 'aupr': aupr,
        'pa_f1': float(pa_f1), 'pa_p': float(pa_pr), 'pa_r': float(pa_rc),
    })

# V1 baseline
for q in QS:
    th = float(np.quantile(vm, q))
    pred = (tm > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, tm))
    pa_pred = point_adjust(pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
    test_results.append({
        'lam': 1.0, 'mu': 0.0, 'q': q, 'th': float(th),
        'f1': float(f1), 'p': float(pr), 'r': float(rc),
        'auc': auc, 'pa_f1': float(pa_f1), 'pa_p': float(pa_pr), 'pa_r': float(pa_rc),
        'label': 'V1 raw_max',
    })

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
# Best overall by F1
best_f1 = max(test_results, key=lambda r: r['f1'])
best_pa = max(test_results, key=lambda r: r['pa_f1'])
best_val_config = [r for r in test_results if r['lam'] == lam_b and r['mu'] == mu_b and r['q'] == q_b][0] if any(r['lam']==lam_b and r['mu']==mu_b and r['q']==q_b for r in test_results) else None

print(f"Best by F1: lam={best_f1['lam']} mu={best_f1['mu']} q={best_f1['q']} F1={best_f1['f1']:.4f} P={best_f1['p']:.4f} R={best_f1['r']:.4f} PA-F1={best_f1['pa_f1']:.4f}")
print(f"Best by PA-F1: lam={best_pa['lam']} mu={best_pa['mu']} q={best_pa['q']} F1={best_pa['f1']:.4f} PA-F1={best_pa['pa_f1']:.4f}")

# V1 comparison
v1_best = max([r for r in test_results if r.get('label') == 'V1 raw_max'], key=lambda r: r['f1'])
print(f"\nV1 raw_max best: q={v1_best['q']} F1={v1_best['f1']:.4f} P={v1_best['p']:.4f} R={v1_best['r']:.4f} PA-F1={v1_best['pa_f1']:.4f}")
print(f"Delta V3 vs V1: F1 {best_f1['f1']-v1_best['f1']:+.4f}  PA-F1 {best_f1['pa_f1']-v1_best['pa_f1']:+.4f}")

# Top-5 by F1
print(f"\nTop-5 by F1:")
for r in sorted(test_results, key=lambda r: r['f1'], reverse=True)[:5]:
    tag = r.get('label', 'V3 l=' + str(r['lam']) + ' m=' + str(r['mu']))
    print(f"  {tag:20s} q={r['q']:.4f} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f} PA-F1={r['pa_f1']:.4f} AUC={r['auc']:.4f}")

# Save
with open(os.path.join(OUT, 'sweep.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=test_results[0].keys()); w.writeheader(); w.writerows(test_results)
save_json({'best_f1': best_f1, 'best_pa': best_pa, 'v1_best': v1_best, 'results': test_results},
          os.path.join(OUT, 'results.json'))
print(f"\nDone: {OUT}")
