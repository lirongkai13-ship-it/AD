"""V3 sparse consistency on full setting (stride=1) — single best config."""
import sys, os, time, json
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
import importlib.util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load v5 full
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
mfd_np = mfd[cc].values.astype(np.float32)
test_v = scaler2.transform(mfd_np)
val_ds = SWaTDynamicWindowDataset(val_v, None, 60, 1, 1, 'future')
test_ds = SWaTDynamicWindowDataset(test_v, merged_lbls, 60, 1, 1, 'future')
print(f"Full: val={len(val_ds)} test={len(test_ds)}")

def batch_scores(r1, r2, x, eps=1e-8):
    e1 = np.abs(x - r1).mean(axis=1); e2 = np.abs(x - r2).mean(axis=1)  # [B,F]
    raw_max = e1.max(axis=1)
    e_c = e1 + e2; k = min(5, e1.shape[1])
    consistency = np.sort(e_c, axis=1)[:, -k:].mean(axis=1)
    r_diff = np.abs(r1 - r2).mean(axis=1); k2 = min(5, e1.shape[1])
    r_diff_topk = np.sort(r_diff, axis=1)[:, -k2:].mean(axis=1)
    cons_dis = consistency / (r_diff_topk + eps)
    return raw_max, consistency, cons_dis

@torch.no_grad()
def collect_scores(model, loader):
    model.eval(); scores_raw, scores_cons, scores_dis, lbls = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device); r1, r2, _, _ = model(x, sei)
        rm, cs, cd = batch_scores(r1.cpu().numpy(), r2.cpu().numpy(), x.cpu().numpy())
        scores_raw.append(rm); scores_cons.append(cs); scores_dis.append(cd)
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return (np.concatenate(scores_raw), np.concatenate(scores_cons), np.concatenate(scores_dis)), \
           np.concatenate(lbls) if lbls else None

# Configs to test
configs = [
    ("V1 raw_max only", 1.0, 0.0),
    ("V3 best (l=0.7,u=0.2)", 0.7, 0.2),
]

print("Collecting val...")
(vr, vc, vd), _ = collect_scores(model, DataLoader(val_ds, 256, shuffle=False))
print("Collecting test...")
(tr_s, tc_s, td_s), test_lbls = collect_scores(model, DataLoader(test_ds, 256, shuffle=False))

results = []
for label, lam, mu in configs:
    val_s = lam * vr + (1-lam) * vc + mu * vd
    test_s = lam * tr_s + (1-lam) * tc_s + mu * td_s
    for q in [0.995, 0.999, 0.9995]:
        th = float(np.quantile(val_s, q))
        pred = (test_s > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, test_s))
        aupr = float(average_precision_score(test_lbls, test_s))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        r = {'label': label, 'q': q, 'f1': float(f1), 'p': float(pr), 'r': float(rc),
             'auc': auc, 'aupr': aupr, 'pa_f1': float(pa_f1)}
        results.append(r)
        print(f"  {label} q={q}: F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f} PA-F1={pa_f1:.4f}")

print(f"\n{'='*60}")
best = max(results, key=lambda r: r['f1'])
print(f"Best full: {best['label']} q={best['q']} F1={best['f1']:.4f}")
save_json(results, os.path.join(os.path.dirname(__file__), "results", "v5_full_sparse", "results.json"))
