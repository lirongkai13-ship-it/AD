"""Final evaluation: v5 + raw_max scoring, dev + full settings."""
import sys, os, time, json, csv
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv, SWaTDynamicWindowDataset)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import pandas as pd, importlib.util

OUT = os.path.join(os.path.dirname(__file__), "results", "v5_final_eval")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# 1. Load v5 models (dev + full checkpoints)
# ============================================================
from models_variants.tri_branch_v5.variant_model import TriBranch_USAD_v5

def load_model(ckpt_path, device):
    # Build graphs
    cfg = load_config('config_dev.yaml')
    dcfg = cfg['data']
    nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    mfd, _ = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    common_cols = [c for c in nfd.columns if c in mfd.columns]
    raw = nfd[common_cols].values.astype(np.float32)
    tr, _, _, _ = split_train_val(raw, None, 0.2)
    tv = StandardScaler().fit_transform(tr)
    static_ei, _ = build_pearson_edge_index(tv)
    bgp = importlib.util.spec_from_file_location('bpg', 'models_variants/prior_fusion/build_prior_graph.py')
    bpgm = importlib.util.module_from_spec(bgp); bgp.loader.exec_module(bpgm)
    prior_ei, prior_w = bpgm.build_prior_graph(common_cols)
    static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

    model = TriBranch_USAD_v5(
        nv=len(common_cols), ws=60, static_edge_index=static_ei,
        prior_edge_index=prior_ei, prior_weights=prior_w,
        hidden_dim=32, gat_heads=2, dropout=0.2, latent_dim=64,
        encoder_mode="tri_branch_residual_gate",
        gamma_mode="learnable", gamma_value=0.05, gate_scale=1.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, static_ei, common_cols

# Load dev model
print("Loading v5 dev...")
model_dev, static_ei, common_cols = load_model(
    'outputs/swat_normal_train_merged_test/tri_branch_v5/best_model.pt', device)
dev_gamma = model_dev.encoder.gated_fusion.gamma.item()
print(f"  Dev gamma: {dev_gamma:.4f}")

# Load full model
full_ckpt = 'outputs/full_setting/tri_branch_v5_full/best_model.pt'
have_full = os.path.exists(full_ckpt)
if have_full:
    print(f"Loading v5 full...")
    model_full, _, _ = load_model(full_ckpt, device)
    full_gamma = model_full.encoder.gated_fusion.gamma.item()
    print(f"  Full gamma: {full_gamma:.4f}")
else:
    print("  Full checkpoint not found, using dev for both")


# ============================================================
# 2. Data loaders
# ============================================================
def get_loaders(stride):
    cfg = load_config('config_dev.yaml')
    set_seed(42)
    _, val_ds, test_ds, _, info = prepare_data(cfg)
    # Rebuild with custom stride
    dcfg = cfg['data']
    nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    mfd, merged_labels = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    common_cols = [c for c in nfd.columns if c in mfd.columns]
    nfd = nfd[common_cols]; mfd = mfd[common_cols]
    normal_raw = nfd.values.astype(np.float32)
    merged_raw = mfd.values.astype(np.float32)
    from data_loader import build_labels
    merged_lbls = build_labels(merged_labels, 'Normal')
    # Split and scale
    train_raw, val_raw, _, _ = split_train_val(normal_raw, None, 0.2)
    scaler = StandardScaler(); scaler.fit(train_raw)
    val_vals = scaler.transform(val_raw); test_vals = scaler.transform(merged_raw)
    ws = 60
    val_ds = SWaTDynamicWindowDataset(val_vals, None, ws, 1, stride, 'future')
    test_ds = SWaTDynamicWindowDataset(test_vals, merged_lbls, ws, 1, stride, 'future')
    val_loader = DataLoader(val_ds, 256, shuffle=False)
    test_loader = DataLoader(test_ds, 256, shuffle=False)
    return val_loader, test_loader, test_ds

val_loader_dev, test_loader_dev, test_ds_dev = get_loaders(10)
val_loader_full, test_loader_full, test_ds_full = get_loaders(1)
print(f"Dev: val={len(val_loader_dev.dataset)} test={len(test_loader_dev.dataset)}")
print(f"Full: val={len(val_loader_full.dataset)} test={len(test_loader_full.dataset)}")


# ============================================================
# 3. Collect errors
# ============================================================
@torch.no_grad()
def collect_errors(model, loader, edge_idx):
    model.eval()
    errors, labels = [], []
    for batch in loader:
        x = batch['x'].to(device)
        r1 = model.forward_eval(x, edge_idx)
        e = (r1 - x).abs().mean(dim=1)  # [B, N]
        errors.append(e.cpu().numpy())
        if 'label' in batch: labels.append(batch['label'].cpu().numpy())
    return np.concatenate(errors), np.concatenate(labels) if labels else None


# ============================================================
# 4. Score functions
# ============================================================
def raw_max_score(errors):
    """e1 raw max, no IQR, top1"""
    return errors.max(axis=1)  # [M]

def evaluate_full(model, val_loader, test_loader, edge_idx, q=0.999, stride=1, ds_test=None):
    t0 = time.time()

    # Collect
    val_err, _ = collect_errors(model, val_loader, edge_idx)
    test_err, test_lbls = collect_errors(model, test_loader, edge_idx)
    print(f"  Errors collected in {time.time()-t0:.1f}s")

    # Score
    val_score = raw_max_score(val_err)
    test_score = raw_max_score(test_err)
    threshold = float(np.quantile(val_score, q))
    test_pred = (test_score > threshold).astype(int)

    # Window-level metrics
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, test_pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, test_score))
    aupr = float(average_precision_score(test_lbls, test_score))

    # Point-adjust
    from utils import point_adjust
    pa_pred = point_adjust(test_pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)

    # TP/FP/TN/FN
    tp = int(((test_pred==1)&(test_lbls==1)).sum())
    fp = int(((test_pred==1)&(test_lbls==0)).sum())
    tn = int(((test_pred==0)&(test_lbls==0)).sum())
    fn = int(((test_pred==0)&(test_lbls==1)).sum())

    # Score distribution
    ns = test_score[test_lbls==0]
    as_ = test_score[test_lbls==1]
    sep = float(as_.mean() / ns.mean()) if ns.mean() > 0 else float('inf')

    # Attack segment analysis
    segments = []
    start = None
    for i in range(len(test_lbls)):
        if test_lbls[i]==1 and start is None: start = i
        elif test_lbls[i]==0 and start is not None:
            segments.append((start, i-1, i-start)); start = None
    if start is not None: segments.append((start, len(test_lbls)-1, len(test_lbls)-start))

    pred_segments = []
    start = None
    for i in range(len(test_pred)):
        if test_pred[i]==1 and start is None: start = i
        elif test_pred[i]==0 and start is not None:
            pred_segments.append((start, i-1, i-start)); start = None
    if start is not None: pred_segments.append((start, len(test_pred)-1, len(test_pred)-start))

    # True attack segments hit
    true_seg_hit = 0; true_seg_missed = 0; true_seg_scores = []
    for st, en, length in segments:
        seg_pred = test_pred[st:en+1]
        detected = int(seg_pred.sum() > 0)
        if detected: true_seg_hit += 1
        else: true_seg_missed += 1
        true_seg_scores.append(float(test_score[st:en+1].max()))

    # Predicted segment scores
    pred_seg_scores = []
    for st, en, length in pred_segments:
        pred_seg_scores.append(float(test_score[st:en+1].max()))

    return {
        'threshold': threshold, 'q': q, 'stride': stride,
        'raw': {'f1': float(f1), 'precision': float(pr), 'recall': float(rc),
                'roc_auc': auc, 'pr_auc': aupr},
        'point_adjust': {'f1': float(pa_f1), 'precision': float(pa_pr), 'recall': float(pa_rc)},
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'normal_mean': float(ns.mean()), 'attack_mean': float(as_.mean()),
        'separation_ratio': sep,
        'true_attack_segments': len(segments),
        'true_seg_hit': true_seg_hit, 'true_seg_missed': true_seg_missed,
        'pred_segments': len(pred_segments),
        'avg_pred_seg_len': float(np.mean([l for _,_,l in pred_segments])) if pred_segments else 0,
        'true_seg_scores': true_seg_scores,
        'pred_seg_scores': pred_seg_scores,
        'eval_time_s': time.time() - t0,
    }


# ============================================================
# 5. Run evaluation
# ============================================================
results = {}

print(f"\n=== Dev (stride=10) ===")
results['dev_v5_raw_max'] = evaluate_full(model_dev, val_loader_dev, test_loader_dev, static_ei, q=0.999, stride=10)
r = results['dev_v5_raw_max']
print(f"  Raw F1={r['raw']['f1']:.4f} P={r['raw']['precision']:.4f} R={r['raw']['recall']:.4f}")
print(f"  PA  F1={r['point_adjust']['f1']:.4f} AUC={r['raw']['roc_auc']:.4f}")
print(f"  Segments: true={r['true_attack_segments']} hit={r['true_seg_hit']} missed={r['true_seg_missed']}")
print(f"  Pred segs: {r['pred_segments']} avg_len={r['avg_pred_seg_len']:.1f}")

if have_full:
    print(f"\n=== Full (stride=1) ===")
    results['full_v5_raw_max'] = evaluate_full(model_full, val_loader_full, test_loader_full, static_ei, q=0.999, stride=1, ds_test=test_ds_full)
    r = results['full_v5_raw_max']
    print(f"  Raw F1={r['raw']['f1']:.4f} P={r['raw']['precision']:.4f} R={r['raw']['recall']:.4f}")
    print(f"  PA  F1={r['point_adjust']['f1']:.4f} AUC={r['raw']['roc_auc']:.4f}")
    print(f"  Segments: true={r['true_attack_segments']} hit={r['true_seg_hit']} missed={r['true_seg_missed']}")
    print(f"  Pred segs: {r['pred_segments']} avg_len={r['pred_seg_len']:.1f}")

# Also run dev with q sweep
print(f"\n=== Dev q-sweep ===")
sweep = []
val_err, _ = collect_errors(model_dev, val_loader_dev, static_ei)
test_err, test_lbls = collect_errors(model_dev, test_loader_dev, static_ei)
val_score_raw = raw_max_score(val_err)
test_score_raw = raw_max_score(test_err)
for q in [0.99, 0.995, 0.997, 0.999, 0.9995]:
    th = float(np.quantile(val_score_raw, q))
    pred = (test_score_raw > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    pa_pred = __import__('utils').point_adjust(pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
    sweep.append({'q': q, 'th': th, 'raw_f1': f1, 'raw_p': pr, 'raw_r': rc, 'pa_f1': pa_f1})
    print(f"  q={q:.4f} th={th:.2f} raw F1={f1:.4f} P={pr:.4f} R={rc:.4f} PA F1={pa_f1:.4f}")


# ============================================================
# 6. Save results
# ============================================================
results['q_sweep_dev'] = sweep
save_json = __import__('utils').save_json
save_json(results, os.path.join(OUT, 'v5_final_eval.json'))

# CSV for q-sweep
with open(os.path.join(OUT, 'q_sweep.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['q','th','raw_f1','raw_p','raw_r','pa_f1'])
    w.writeheader(); w.writerows(sweep)

# Summary table
print(f"\n{'='*80}")
print(f"{'Setting':<12s} {'Stride':<8s} {'Raw F1':>8s} {'P':>8s} {'R':>8s} {'PA F1':>8s} {'PA P':>8s} {'AUC':>8s}")
print(f"{'-'*80}")
for key, r in results.items():
    if 'raw' not in r: continue
    raw = r['raw']; pa = r.get('point_adjust', {})
    print(f"{key:<12s} {r.get('stride','?'):<8} {raw['f1']:8.4f} {raw['precision']:8.4f} {raw['recall']:8.4f} "
          f"{pa.get('f1',0):8.4f} {pa.get('precision',0):8.4f} {raw.get('roc_auc',0):8.4f}")

print(f"\nResults saved to: {OUT}")
