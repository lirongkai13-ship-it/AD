"""v5 Sparse-Consistency Enhanced USAD Scoring — streaming, no IQR."""
import sys, os, time, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_sparse_consistency")
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
# 2. Data
# ============================================================
_, val_ds, test_ds, _, info = prepare_data(cfg)
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)
print(f"Val={len(val_ds)} Test={len(test_ds)}")


# ============================================================
# 3. Scoring functions
# ============================================================
def batch_scores(r1, r2, x, eps=1e-8):
    """Compute per-sample scores for one batch.
    r1,r2,x: [B,T,F] numpy
    All scores: mean over time first, then operate on features [B,F] or aggregate to [B].
    """
    B, T, F = r1.shape

    # Error per sample, per feature: mean over time -> [B, F]
    e1_feat = np.abs(x - r1).mean(axis=1)  # [B, F]
    e2_feat = np.abs(x - r2).mean(axis=1)  # [B, F]

    # V1: raw max — max over features, same as v5 best
    score_raw_max = e1_feat.max(axis=1)  # [B]

    # V2: top-k consistency over FEATURES (not time)
    e_combined = e1_feat + e2_feat  # [B, F]
    k = min(5, F)
    score_consistency = np.sort(e_combined, axis=1)[:, -k:].mean(axis=1)  # [B]

    # normalized disagreement over features
    r_diff = np.abs(r1 - r2).mean(axis=1)  # [B, F] mean over time
    k2 = min(5, F)
    r_diff_topk = np.sort(r_diff, axis=1)[:, -k2:].mean(axis=1)  # [B]
    cons_dis = score_consistency / (r_diff_topk + eps)  # [B]

    return {
        'raw_max': score_raw_max,
        'consistency': score_consistency,
        'cons_dis': cons_dis,
        'e1_feat': e1_feat,
        'e2_feat': e2_feat,
    }


def final_score(scores_dict, lam=0.7, mu=0.2):
    """lam * raw_max + (1-lam) * consistency + mu * cons_dis"""
    return lam * scores_dict['raw_max'] + \
           (1 - lam) * scores_dict['consistency'] + \
           mu * scores_dict['cons_dis']


# ============================================================
# 4. Collect scores (streaming) and evaluate
# ============================================================
@torch.no_grad()
def collect_final_scores(model, loader, score_fn):
    """Streaming: compute final scores for all batches."""
    model.eval()
    scores, lbls = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1, r2, _, _ = model(x, sei)
        sd = batch_scores(r1.cpu().numpy(), r2.cpu().numpy(), x.cpu().numpy())
        s = score_fn(sd)
        scores.append(s)
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return np.concatenate(scores), np.concatenate(lbls) if lbls else None


# ============================================================
# 5. Ablation study
# ============================================================
configs = [
    # (label, lambda, mu)
    ("V1 raw_max only",       1.0, 0.0),
    ("V2 consistency only",   0.0, 0.2),   # consistency + cons_dis only
    ("V3 proposed l=0.7",     0.7, 0.2),
    ("V3a l=0.8 u=0.15",      0.8, 0.15),
    ("V3b l=0.6 u=0.25",      0.6, 0.25),
]

results = []
for label, lam, mu in configs:
    t0 = time.time()

    def score_fn(sd):
        return final_score(sd, lam=lam, mu=mu)

    # Val
    val_s, _ = collect_final_scores(model, val_loader, score_fn)
    # Test
    test_s, test_lbls = collect_final_scores(model, test_loader, score_fn)

    # Threshold from val (percentile, no IQR)
    for q in [0.995, 0.999]:
        th = float(np.quantile(val_s, q))
        pred = (test_s > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, test_s))
        aupr = float(average_precision_score(test_lbls, test_s))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)

        r = {'label': label, 'lam': lam, 'mu': mu, 'q': q, 'th': float(th),
             'f1': float(f1), 'p': float(pr), 'r': float(rc),
             'auc': auc, 'aupr': aupr,
             'pa_f1': float(pa_f1), 'pa_p': float(pa_pr), 'pa_r': float(pa_rc),
             'time_s': time.time()-t0}
        results.append(r)
        best_q = " <--" if f1 == max(rr['f1'] for rr in results if rr['label']==label) else ""
        print(f"  {label:<35s} q={q:.4f} F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f}{best_q}")

# Also V1 baseline: e1 raw max over features (original v5 best)
t0 = time.time()
val_raw, _ = collect_final_scores(model, val_loader, lambda sd: sd['raw_max'])
test_raw, test_lbls2 = collect_final_scores(model, test_loader, lambda sd: sd['raw_max'])
for q in [0.995, 0.999, 0.9995]:
    th = float(np.quantile(val_raw, q))
    pred = (test_raw > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls2, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls2, test_raw))
    r = {'label': f'V1 raw_max (e1 only) q={q}', 'lam': 1.0, 'mu': 0.0, 'q': q, 'th': float(th),
         'f1': float(f1), 'p': float(pr), 'r': float(rc), 'auc': auc, 'time_s': time.time()-t0}
    results.append(r)
    print(f"  V1 raw_max (e1 only) q={q}       F1={f1:.4f} P={pr:.4f} R={rc:.4f} AUC={auc:.4f}")


# ============================================================
# 6. Summary
# ============================================================
print(f"\n{'='*80}")
print(f"BEST PER CONFIG")
print(f"{'='*80}")
print(f"{'Config':<40s} {'q':>7s} {'F1':>8s} {'P':>8s} {'R':>8s} {'AUC':>8s} {'AUPR':>8s}")
print(f"{'-'*85}")
for label, lam, mu in configs[:7]:
    config_results = [r for r in results if r['label'] == label]
    if config_results:
        best = max(config_results, key=lambda r: r['f1'])
        print(f"{label:<40s} {best['q']:7.4f} {best['f1']:8.4f} {best['p']:8.4f} {best['r']:8.4f} {best['auc']:8.4f} {best.get('aupr',0):8.4f}")

# Best overall
best_all = max(results, key=lambda r: r['f1'])
print(f"\nBest overall: {best_all['label']} F1={best_all['f1']:.4f}")

# Save
csv_path = os.path.join(OUT, 'sparse_consistency_ablation.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
save_json(results, os.path.join(OUT, 'results.json'))
print(f"Done: {OUT}")
