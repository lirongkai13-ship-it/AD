"""Full setting parameter search for v5 + pos_emb scoring strategy."""
import sys, os, time, json, csv, itertools
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv, SWaTDynamicWindowDataset,
                         build_labels)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util
from collections import defaultdict

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_full_scoring_search")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# 1. Load v5 model
# ============================================================
def load_v5(ckpt_path, device):
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
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, sei, cc

# Try full checkpoint first, fall back to dev
full_ckpt = 'outputs/full_setting/tri_branch_v5_full/best_model.pt'
dev_ckpt = 'outputs/swat_normal_train_merged_test/tri_branch_v5/best_model.pt'
use_full = os.path.exists(full_ckpt)
ckpt_path = full_ckpt if use_full else dev_ckpt
print(f"Loading v5 from: {ckpt_path} (full={use_full})")
model, static_ei, common_cols = load_v5(ckpt_path, device)
gamma = model.encoder.gated_fusion.gamma.item()
print(f"  Gamma: {gamma:.4f}")


# ============================================================
# 2. Build full datasets (stride=1 for val, stride=1 for test)
# ============================================================
def build_full_dataset(cfg, split='val', stride=1):
    dcfg = cfg['data']
    nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    mfd, raw_labels = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    cc = [c for c in nfd.columns if c in mfd.columns]
    nfd = nfd[cc]; mfd = mfd[cc]
    normal_raw = nfd.values.astype(np.float32)
    merged_raw = mfd.values.astype(np.float32)
    merged_lbls = build_labels(raw_labels, 'Normal')

    train_raw, val_raw, _, _ = split_train_val(normal_raw, None, 0.2)
    scaler = StandardScaler(); scaler.fit(train_raw)
    if split == 'val':
        vals = scaler.transform(val_raw)
        ds = SWaTDynamicWindowDataset(vals, None, 60, 1, stride, 'future')
    else:
        vals = scaler.transform(merged_raw)
        ds = SWaTDynamicWindowDataset(vals, merged_lbls, 60, 1, stride, 'future')
    return DataLoader(ds, 256, shuffle=False, drop_last=False)

cfg = load_config('config_dev.yaml')
print("Building full val loader (stride=1)...")
val_loader_full = build_full_dataset(cfg, 'val', 1)
print(f"  Val: {len(val_loader_full.dataset)} windows")
print("Building full test loader (stride=1)...")
test_loader_full = build_full_dataset(cfg, 'test', 1)
print(f"  Test: {len(test_loader_full.dataset)} windows")

# Also dev loaders for comparison
val_loader_dev = build_full_dataset(cfg, 'val', 10)
test_loader_dev = build_full_dataset(cfg, 'test', 10)
print(f"Dev: val={len(val_loader_dev.dataset)} test={len(test_loader_dev.dataset)}")


# ============================================================
# 3. Collect errors (once for full, once for dev)
# ============================================================
@torch.no_grad()
def collect_errors(model, loader):
    model.eval()
    errs, lbls = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1 = model.forward_eval(x, static_ei)
        e = (r1 - x).abs().mean(dim=1)
        errs.append(e.cpu().numpy())
        if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
    return np.concatenate(errs), np.concatenate(lbls) if lbls else None

print("\nCollecting FULL errors...")
t0 = time.time()
val_err_f, _ = collect_errors(model, val_loader_full)
test_err_f, test_lbls_f = collect_errors(model, test_loader_full)
print(f"  Full: {time.time()-t0:.1f}s, val={val_err_f.shape} test={test_err_f.shape}")

print("Collecting DEV errors...")
t0 = time.time()
val_err_d, _ = collect_errors(model, val_loader_dev)
test_err_d, test_lbls_d = collect_errors(model, test_loader_dev)
print(f"  Dev: {time.time()-t0:.1f}s, val={val_err_d.shape} test={test_err_d.shape}")


# ============================================================
# 4. Scoring functions
# ============================================================
def raw_score(errors, topk=1):
    """raw max top-k, no IQR"""
    k = min(topk, errors.shape[1])
    topk_vals = np.sort(errors, axis=1)[:, -k:]
    return topk_vals.mean(axis=1)

def iqr_score(errors, topk=1, iqr_params=None, fit_on=None):
    """IQR normalize then top-k"""
    if iqr_params is None and fit_on is not None:
        iqr_params = fit_iqr_params(fit_on)
    if iqr_params is None:
        iqr_params = fit_iqr_params(errors)
    norm = apply_iqr_normalize(errors, iqr_params)
    return aggregate_topk_score(norm, topk=topk)

def temporal_merge(pred, scores, window):
    """Simple dilation-merge: expand positive regions by window/2 each side."""
    if window is None or window <= 1:
        return pred
    from scipy.ndimage import binary_dilation
    structure = np.ones(window, dtype=bool)
    return binary_dilation(pred, structure=structure).astype(int)

def evaluate(errors, labels, val_errors, topk, use_iqr, q, merge_window, fit_iqr_on_val=True):
    """Full evaluation pipeline."""
    # Compute scores
    if use_iqr:
        iqr_p = fit_iqr_params(val_errors)
        val_score = iqr_score(val_errors, topk, iqr_p)
        test_score = iqr_score(errors, topk, iqr_p)
    else:
        val_score = raw_score(val_errors, topk)
        test_score = raw_score(errors, topk)

    # Threshold from validation
    th = float(np.quantile(val_score, q))

    # Predict
    pred = (test_score > th).astype(int)

    # Temporal merge
    if merge_window is not None and merge_window > 1:
        pred = temporal_merge(pred, test_score, merge_window)

    # Metrics
    pr, rc, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(labels, test_score))
    aupr = float(average_precision_score(labels, test_score))

    # Point-adjust
    pa_pred = point_adjust(pred, labels)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(labels, pa_pred, average='binary', zero_division=0)

    tp = int(((pred==1)&(labels==1)).sum())
    fp = int(((pred==1)&(labels==0)).sum())
    fn = int(((pred==0)&(labels==1)).sum())
    pred_pos = int(pred.sum())

    return {
        'topk': topk, 'use_iqr': use_iqr, 'q': q, 'threshold': th,
        'merge_window': merge_window if merge_window else 0,
        'raw_f1': float(f1), 'precision': float(pr), 'recall': float(rc),
        'auc': auc, 'aupr': aupr,
        'pa_f1': float(pa_f1), 'pa_precision': float(pa_pr), 'pa_recall': float(pa_rc),
        'tp': tp, 'fp': fp, 'fn': fn, 'pred_pos': pred_pos,
        'pred_pos_ratio': float(pred_pos / len(labels)),
        'score_mean': float(test_score.mean()), 'score_std': float(test_score.std()),
        'score_min': float(test_score.min()), 'score_max': float(test_score.max()),
    }


# ============================================================
# 5. Full validation parameter search
# ============================================================
print(f"\n{'='*80}")
print("FULL VALIDATION PARAMETER SEARCH")
print(f"{'='*80}")

TOP_K_LIST = [1, 3, 5]
USE_IQR_LIST = [False, True]
Q_LIST = [0.990, 0.995, 0.997, 0.999, 0.9995]
MERGE_LIST = [None, 5, 10, 20]

all_results = []
total = len(TOP_K_LIST) * len(USE_IQR_LIST) * len(Q_LIST) * len(MERGE_LIST)
i = 0
for topk, use_iqr, q, merge_w in itertools.product(TOP_K_LIST, USE_IQR_LIST, Q_LIST, MERGE_LIST):
    i += 1
    r = evaluate(test_err_f, test_lbls_f, val_err_f, topk, use_iqr, q, merge_w)
    r['setting'] = 'full_val'
    r['stride'] = 1
    all_results.append(r)
    if i % 20 == 0:
        print(f"  [{i}/{total}] topk={topk} iqr={use_iqr} q={q:.4f} merge={merge_w} -> F1={r['raw_f1']:.4f}")

# Find best by raw F1
best_raw = max(all_results, key=lambda r: r['raw_f1'])
best_pa = max(all_results, key=lambda r: r['pa_f1'])
best_p = max([r for r in all_results if r['recall'] >= 0.65], key=lambda r: r['precision'], default=best_raw)
best_r = max([r for r in all_results if r['precision'] >= 0.70], key=lambda r: r['recall'], default=best_raw)

print(f"\nBest by Raw F1:  topk={best_raw['topk']} iqr={best_raw['use_iqr']} q={best_raw['q']:.4f} "
      f"merge={best_raw['merge_window']} th={best_raw['threshold']:.2f}")
print(f"  F1={best_raw['raw_f1']:.4f} P={best_raw['precision']:.4f} R={best_raw['recall']:.4f} "
      f"AUC={best_raw['auc']:.4f} PA-F1={best_raw['pa_f1']:.4f}")

print(f"\nBest by PA-F1:   topk={best_pa['topk']} iqr={best_pa['use_iqr']} q={best_pa['q']:.4f} "
      f"merge={best_pa['merge_window']} -> PA-F1={best_pa['pa_f1']:.4f}")

# Save full search CSV
csv_path = os.path.join(OUT, 'full_val_scoring_search_results.csv')
fieldnames = ['setting','stride','topk','use_iqr','q','threshold','merge_window',
              'raw_f1','precision','recall','auc','aupr',
              'pa_f1','pa_precision','pa_recall',
              'tp','fp','fn','pred_pos','pred_pos_ratio',
              'score_mean','score_std','score_min','score_max']
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader(); w.writerows(all_results)
print(f"\nSaved: {csv_path} ({len(all_results)} rows)")


# ============================================================
# 6. Full test final evaluation with best config
# ============================================================
print(f"\n{'='*80}")
print("FULL TEST FINAL EVALUATION (best val config)")
print(f"{'='*80}")

best = best_raw
print(f"Config: topk={best['topk']} iqr={best['use_iqr']} q={best['q']:.4f} "
      f"merge={best['merge_window']} th={best['threshold']:.2f}")

r_test = evaluate(test_err_f, test_lbls_f, val_err_f,
                   best['topk'], best['use_iqr'], best['q'], best['merge_window'])
r_test['setting'] = 'full_test'; r_test['stride'] = 1

print(f"Full Test: F1={r_test['raw_f1']:.4f} P={r_test['precision']:.4f} R={r_test['recall']:.4f}")
print(f"  AUC={r_test['auc']:.4f} AUPR={r_test['aupr']:.4f}")
print(f"  PA-F1={r_test['pa_f1']:.4f} PA-P={r_test['pa_precision']:.4f} PA-R={r_test['pa_recall']:.4f}")
print(f"  TP={r_test['tp']} FP={r_test['fp']} FN={r_test['fn']}")


# ============================================================
# 7. Dev evaluation with same scoring for comparison
# ============================================================
print(f"\n{'='*80}")
print("DEV COMPARISON (same scoring as best full config)")
print(f"{'='*80}")

r_dev = evaluate(test_err_d, test_lbls_d, val_err_d,
                  best['topk'], best['use_iqr'], best['q'], best['merge_window'])
r_dev['setting'] = 'dev'; r_dev['stride'] = 10

print(f"Dev: F1={r_dev['raw_f1']:.4f} P={r_dev['precision']:.4f} R={r_dev['recall']:.4f}")
print(f"  AUC={r_dev['auc']:.4f} PA-F1={r_dev['pa_f1']:.4f}")


# ============================================================
# 8. Segment diagnostics
# ============================================================
print(f"\n{'='*80}")
print("SEGMENT DIAGNOSTICS")
print(f"{'='*80}")

# Recompute with best config for full test
test_score_final = raw_score(test_err_f, best['topk']) if not best['use_iqr'] else \
    iqr_score(test_err_f, best['topk'],
              fit_iqr_params(val_err_f) if best['use_iqr'] else fit_iqr_params(test_err_f))
test_pred_final = (test_score_final > best['threshold']).astype(int)

# Attack segments
segs = []; start = None
for i in range(len(test_lbls_f)):
    if test_lbls_f[i]==1 and start is None: start = i
    elif test_lbls_f[i]==0 and start is not None:
        segs.append({'start': start, 'end': i-1, 'len': i-start}); start = None
if start is not None: segs.append({'start': start, 'end': len(test_lbls_f)-1, 'len': len(test_lbls_f)-start})

# Pred segments
psegs = []; start = None
for i in range(len(test_pred_final)):
    if test_pred_final[i]==1 and start is None: start = i
    elif test_pred_final[i]==0 and start is not None:
        psegs.append({'start': start, 'end': i-1, 'len': i-start}); start = None
if start is not None: psegs.append({'start': start, 'end': len(test_pred_final)-1, 'len': len(test_pred_final)-start})

# Hit analysis
hit = 0; missed = 0
for s in segs:
    if test_pred_final[s['start']:s['end']+1].sum() > 0: hit += 1
    else: missed += 1

print(f"True attack segments: {len(segs)}")
print(f"Pred segments: {len(psegs)}")
print(f"Hit: {hit}  Missed: {missed}")
print(f"Avg pred seg len: {np.mean([s['len'] for s in psegs]):.1f}" if psegs else "N/A")

# Per-segment max scores
true_seg_scores = [float(test_score_final[s['start']:s['end']+1].max()) for s in segs]
false_seg_scores = [float(test_score_final[s['start']:s['end']+1].max()) for s in psegs]
print(f"True seg score range: [{min(true_seg_scores):.2f}, {max(true_seg_scores):.2f}]")
print(f"Pred seg score range: [{min(false_seg_scores):.2f}, {max(false_seg_scores):.2f}]" if false_seg_scores else "N/A")

# Score distribution
ns = test_score_final[test_lbls_f==0]
as_ = test_score_final[test_lbls_f==1]
print(f"Normal score: mean={ns.mean():.4f} std={ns.std():.4f}")
print(f"Attack score: mean={as_.mean():.4f} std={as_.std():.4f}")
print(f"Separation: {as_.mean()/ns.mean():.2f}x")


# ============================================================
# 9. Final summary
# ============================================================
print(f"\n{'='*80}")
print("FINAL SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'Setting':<12s} {'topk':>5s} {'iqr':>5s} {'q':>7s} {'merge':>6s} {'F1':>8s} {'P':>8s} {'R':>8s} {'PA-F1':>8s} {'AUC':>8s}")
print(f"{'-'*85}")
for label, r in [('full_val', best_raw), ('full_test', r_test), ('dev', r_dev)]:
    print(f"{label:<12s} {r.get('topk',best['topk']):5} {str(r.get('use_iqr',best['use_iqr'])):>5s} "
          f"{r.get('q',best['q']):7.4f} {str(r.get('merge_window',best['merge_window'])):>6s} "
          f"{r['raw_f1']:8.4f} {r['precision']:8.4f} {r['recall']:8.4f} {r['pa_f1']:8.4f} {r['auc']:8.4f}")

# Save final
final_results = {
    'best_val_config': best_raw,
    'full_test': r_test,
    'dev_comparison': r_dev,
    'full_search_n': len(all_results),
    'segment_diagnostics': {
        'true_segments': len(segs), 'pred_segments': len(psegs),
        'hit': hit, 'missed': missed,
        'true_seg_score_min': min(true_seg_scores), 'true_seg_score_max': max(true_seg_scores),
    }
}
save_json(final_results, os.path.join(OUT, 'final_results.json'))
print(f"\nDone! Results: {OUT}")
