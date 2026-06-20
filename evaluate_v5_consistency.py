"""USAD Combined Consistency Score for v5 — streaming (no OOM)."""
import sys, os, time, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_consistency")
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
print(f"v5 loaded, gamma={model.encoder.gated_fusion.gamma.item():.4f}")

# ============================================================
# 2. Data loaders
# ============================================================
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)
print(f"Val={len(val_ds)} Test={len(test_ds)}")


# ============================================================
# 3. Consistency score (per-batch)
# ============================================================
def batch_score(r1, r2, x, alpha, beta, gamma, time_pool, topk):
    """r1,r2,x: [B,T,F] numpy arrays"""
    e1 = np.abs(x - r1); e2 = np.abs(x - r2)
    e1_m = e1.mean(axis=-1); e2_m = e2.mean(axis=-1)  # [B,T]
    cons_e = np.abs(e1_m - e2_m)                       # [B,T]
    cons_o = np.abs(r1 - r2).mean(axis=-1)              # [B,T]
    score = alpha*(e1_m+e2_m) + beta*cons_e + gamma*cons_o  # [B,T]

    if time_pool == 'max':
        return score.max(axis=1)
    elif time_pool == 'topk':
        k = min(topk, score.shape[1])
        return np.sort(score, axis=1)[:, -k:].mean(axis=1)
    else:
        return score.mean(axis=1)


# ============================================================
# 4. Collect val scores (streaming) + compute IQR
# ============================================================
@torch.no_grad()
def collect_scores(model, loader, alpha, beta, gamma, time_pool, topk, iqr_params=None):
    model.eval()
    scores, lbls = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, r2, _, _ = model(x, sei)
        s = batch_score(r1.cpu().numpy(), r2.cpu().numpy(), x.cpu().numpy(),
                        alpha, beta, gamma, time_pool, topk)
        scores.append(s)
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(lbls) if lbls else None

    if iqr_params is not None:
        scores = apply_iqr_normalize(scores.reshape(-1,1), iqr_params).flatten()
    return scores, labels


# ============================================================
# 5. Ablation
# ============================================================
results = []

for label, alpha, beta, gamma in [
    ("baseline e1+e2", 1.0, 0.0, 0.0),
    ("consistency only", 0.0, 0.5, 0.5),
    ("full a=1 b=0.5 g=0.5", 1.0, 0.5, 0.5),
    ("full a=1 b=0.3 g=0.3", 1.0, 0.3, 0.3),
    ("full a=1 b=0.7 g=0.7", 1.0, 0.7, 0.7),
    ("full a=0.5 b=0.5 g=0.5", 0.5, 0.5, 0.5),
]:
    for pool, topk, pname in [('max', 0, 'max'), ('topk', 5, 'topk5'), ('mean', 0, 'mean')]:
        t0 = time.time()
        # Val scores
        val_s, _ = collect_scores(model, val_loader, alpha, beta, gamma, pool, topk)
        iqr_p = fit_iqr_params(val_s.reshape(-1, 1))

        # Test scores
        test_s, test_lbls = collect_scores(model, test_loader, alpha, beta, gamma, pool, topk, iqr_p)

        # Threshold + metrics
        th = float(np.quantile(val_s, 0.995))
        pred = (test_s > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, test_s))
        aupr = float(average_precision_score(test_lbls, test_s))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        tp = int(((pred==1)&(test_lbls==1)).sum())
        fp = int(((pred==1)&(test_lbls==0)).sum())
        fn = int(((pred==0)&(test_lbls==1)).sum())

        r = {'label': f'{label}+{pname}', 'alpha': alpha, 'beta': beta, 'gamma': gamma,
             'pool': pname, 'f1': float(f1), 'p': float(pr), 'r': float(rc),
             'auc': auc, 'aupr': aupr, 'pa_f1': float(pa_f1), 'th': float(th),
             'tp': tp, 'fp': fp, 'fn': fn, 'time_s': time.time()-t0}
        results.append(r)
        print(f"  {r['label']:<50s} F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f} PA-F1={pa_f1:.4f} ({r['time_s']:.0f}s)")

# raw max baseline
t0 = time.time()
val_raw, _ = collect_scores(model, val_loader, 1.0, 0.0, 0.0, 'max', 0)  # e1_max
test_raw, test_lbls2 = collect_scores(model, test_loader, 1.0, 0.0, 0.0, 'max', 0,
                                       fit_iqr_params(val_raw.reshape(-1, 1)))
th_r = float(np.quantile(val_raw, 0.9995))
pred_r = (test_raw > th_r).astype(int)
pr, rc, f1_r, _ = precision_recall_fscore_support(test_lbls2, pred_r, average='binary', zero_division=0)
auc_r = float(roc_auc_score(test_lbls2, test_raw))
results.append({'label': 'raw max (v5 best)', 'f1': float(f1_r), 'p': float(pr), 'r': float(rc),
                'auc': auc_r, 'pa_f1': 0, 'th': float(th_r), 'pool': 'max', 'time_s': time.time()-t0,
                'alpha': '-', 'beta': '-', 'gamma': '-'})
print(f"  raw max (v5 best): F1={f1_r:.4f} P={pr:.4f} R={rc:.4f} AUC={auc_r:.4f}")

# Print summary
print(f"\n{'='*80}")
best = max(results, key=lambda r: r['f1'])
print(f"Best: {best['label']} F1={best['f1']:.4f}")
print(f"  vs raw max: delta={best['f1']-f1_r:+.4f}")

# Save
csv_path = os.path.join(OUT, 'consistency_ablation.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
save_json({'results': results, 'best': best}, os.path.join(OUT, 'results.json'))
print(f"Done: {OUT}")
