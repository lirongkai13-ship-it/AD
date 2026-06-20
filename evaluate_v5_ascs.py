"""V4: Adaptive Sparse Consistency Scoring (ASCS) for v5."""
import sys, os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv)
from utils import (load_config, get_device, ensure_dir, point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import importlib.util

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_ascs")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# 1. Load v5
# ============================================================
cfg = load_config('config_dev.yaml')
dcfg = cfg['data']
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
print(f"v5 gamma={model.encoder.gated_fusion.gamma.item():.4f}")

# ============================================================
# 2. Data
# ============================================================
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)
print(f"Val={len(val_ds)} Test={len(test_ds)}")


# ============================================================
# 3. Scoring functions
# ============================================================
def batch_scores(r1, r2, x, topk_f=5, eps=1e-8):
    """r1,r2,x: [B,T,F] numpy"""
    e1 = np.abs(x - r1).mean(axis=1)  # [B,F]
    e2 = np.abs(x - r2).mean(axis=1)  # [B,F]

    # Main signal: raw max
    score_main = e1.max(axis=1)  # [B]

    # Consistency: top-k over features
    e_combined = e1 + e2  # [B,F]
    k_f = min(topk_f, e1.shape[1])
    consistency = np.sort(e_combined, axis=1)[:, -k_f:].mean(axis=1)  # [B]

    # Disagreement ratio: same as V3 working version
    r_diff = np.abs(r1 - r2).mean(axis=1)  # [B,F]
    r_diff_topk = np.sort(r_diff, axis=1)[:, -k_f:].mean(axis=1)  # [B]
    ratio = consistency / (r_diff_topk + eps)  # [B] — same scale as consistency

    return e1, e2, score_main, consistency, ratio


def adaptive_lambda(score_main, consistency, ratio):
    """
    Adaptive lambda based on agreement.
    ratio = consistency / |r1-r2|_topk — larger means decoders agree more.
    """
    # ratio typically 1-20; use log-scale sigmoid
    ratio_norm = np.clip(np.log1p(ratio) / 3.0, 0.1, 3.0)
    lam = 1.0 / (1.0 + np.exp(-2.0 * (ratio_norm - 1.0)))
    lam = np.clip(lam, 0.3, 0.9)
    return lam


# ============================================================
# 4. Evaluate
# ============================================================
@torch.no_grad()
def collect_all(model, loader):
    model.eval()
    s_main, s_cons, s_ratio, lbls = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device); r1, r2, _, _ = model(x, sei)
        _, _, sm, sc, sr = batch_scores(r1.cpu().numpy(), r2.cpu().numpy(), x.cpu().numpy())
        s_main.append(sm); s_cons.append(sc); s_ratio.append(sr)
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return (np.concatenate(s_main), np.concatenate(s_cons), np.concatenate(s_ratio)), \
           np.concatenate(lbls) if lbls else None

print("Collecting val...")
(vm, vc, vr), _ = collect_all(model, val_loader)
print("Collecting test...")
(tm, tc, tr), test_lbls = collect_all(model, test_loader)

results = []

# V1: raw_max only
for q in [0.995, 0.999, 0.9995]:
    th = float(np.quantile(vm, q))
    pred = (tm > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, tm))
    results.append({'v': 'V1 raw_max', 'q': q, 'f1': float(f1), 'p': float(pr), 'r': float(rc), 'auc': auc})
    print(f"  V1 q={q:.4f}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f}")

# V3: fixed lambda
for lam, mu, label in [(0.7, 0.2, 'V3 fixed l=0.7')]:
    vs = lam*vm + (1-lam)*vc + mu*vr
    ts = lam*tm + (1-lam)*tc + mu*tr
    for q in [0.995, 0.999, 0.9995]:
        th = float(np.quantile(vs, q))
        pred = (ts > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, ts))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        results.append({'v': label, 'q': q, 'lam': lam, 'mu': mu, 'f1': float(f1), 'p': float(pr), 'r': float(rc), 'auc': auc, 'pa_f1': float(pa_f1), 'pa_p': float(pa_pr)})
        print(f"  {label} q={q:.4f}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f} PA-F1={pa_f1:.4f}")

# V4: adaptive lambda
mu_vals = [0.1, 0.2, 0.3]
for mu in mu_vals:
    lam_v = adaptive_lambda(vm, vc, vr)
    lam_t = adaptive_lambda(tm, tc, tr)
    vs = lam_v*vm + (1-lam_v)*vc + mu*vr
    ts = lam_t*tm + (1-lam_t)*tc + mu*tr
    label = f'V4 adaptive mu={mu}'
    for q in [0.995, 0.999, 0.9995]:
        th = float(np.quantile(vs, q))
        pred = (ts > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, ts))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        results.append({'v': label, 'q': q, 'lam_mean': float(lam_t.mean()), 'mu': mu, 'f1': float(f1), 'p': float(pr), 'r': float(rc), 'auc': auc})
        print(f"  {label} lam_mean={lam_t.mean():.3f} q={q:.4f}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f}")

# Best per variant
print(f"\n{'='*60}")
best = max(results, key=lambda r: r['f1'])
print(f"Best: {best['v']} F1={best['f1']:.4f} P={best['p']:.4f}")
for vname in ['V1 raw_max', 'V3 fixed l=0.7']:
    vr_list = [r for r in results if r['v'] == vname]
    if vr_list:
        b = max(vr_list, key=lambda r: r['f1'])
        print(f"  {vname}: F1={b['f1']:.4f} P={b['p']:.4f} R={b['r']:.4f} AUC={b['auc']:.4f}")
for mu in mu_vals:
    vr_list = [r for r in results if r['v'] == f'V4 adaptive mu={mu}']
    if vr_list:
        b = max(vr_list, key=lambda r: r['f1'])
        print(f"  V4 mu={mu}: F1={b['f1']:.4f} P={b['p']:.4f} R={b['r']:.4f} lam_mean={b.get('lam_mean',0):.3f}")

save_json(results, os.path.join(OUT, 'results.json'))
print(f"Done: {OUT}")
