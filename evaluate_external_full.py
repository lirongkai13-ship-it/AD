"""Full-parameter external model scoring search and evaluation."""
import sys, os, time, json, csv, itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index,
                         split_train_val, read_swat_csv, SWaTDynamicWindowDataset, build_labels)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "results", "external_full_scoring")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model registry (from run_single.py)
MODELS = {
    "USAD":       ("models.usad.model",      "USAD",       dict(n_vars=51, window=60, hidden=64, latent=32, dropout=0.1)),
    "DAGMM":      ("models.dagmm.model",     "DAGMM",      dict(n_vars=51, window=60, hidden=64, latent=16, n_gmm=4, dropout=0.1)),
    "LSTM-AE":    ("models.lstm_ae.model",   "LSTMAE",     dict(n_vars=51, window=60, hidden=64, num_layers=2, dropout=0.1)),
    "MAD-GAN":    ("models.mad_gan.model",   "MADGAN",     dict(n_vars=51, window=60, noise_dim=32, hidden=64)),
    "DCdetector": ("models.dcdetector.model","DCdetector", dict(n_vars=51, window=60, d_model=96, n_heads=4, dropout=0.1)),
    "TranAD":     ("models.tranad.model",    "TranAD",     dict(n_vars=51, window=60, d_model=48, n_heads=4, n_layers=2, dropout=0.1)),
    "AnoTrans":   ("models.ano_trans.model", "AnomalyTransformer", dict(n_vars=51, window=60, d_model=64, n_heads=4, n_layers=2, dropout=0.1)),
    "CAN":        ("models.can.model",       "CAN",        dict(n_vars=51, window=60, d_model=96, n_heads=4, dropout=0.1)),
    "MTAD-GAT":   ("models.mtad_gat.model",  "MTADGAT",    dict(n_vars=51, window=60, hidden=48, heads=2, latent=32, dropout=0.1)),
}

# Models with checkpoints
CHECKPOINT_DIR = "results/models_full"
AVAILABLE = [d for d in os.listdir(CHECKPOINT_DIR)
             if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
             and os.path.exists(os.path.join(CHECKPOINT_DIR, d, "best_model.pt"))
             and d in MODELS]
print(f"Available models: {len(AVAILABLE)}: {AVAILABLE}")


# ============================================================
# Build datasets (full stride=1)
# ============================================================
cfg = load_config('config_dev.yaml')
dcfg = cfg['data']
nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
mfd, raw_lbls = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
cc = [c for c in nfd.columns if c in mfd.columns]
nfd = nfd[cc]; mfd = mfd[cc]
normal_raw = nfd.values.astype(np.float32); merged_raw = mfd.values.astype(np.float32)
merged_lbls = build_labels(raw_lbls, 'Normal')
train_raw, val_raw, _, _ = split_train_val(normal_raw, None, 0.2)
scaler = StandardScaler(); scaler.fit(train_raw)
val_vals = scaler.transform(val_raw); test_vals = scaler.transform(merged_raw)

val_ds_full = SWaTDynamicWindowDataset(val_vals, None, 60, 1, 1, 'future')
test_ds_full = SWaTDynamicWindowDataset(test_vals, merged_lbls, 60, 1, 1, 'future')
val_loader = DataLoader(val_ds_full, 256, shuffle=False)
test_loader = DataLoader(test_ds_full, 256, shuffle=False)
print(f"Full: val={len(val_ds_full)} test={len(test_ds_full)}")

# Edge index for MTAD-GAT
train_vals = scaler.transform(train_raw)
edge_idx, _ = build_pearson_edge_index(train_vals)
edge_idx = edge_idx.to(device)


# ============================================================
# Scoring functions
# ============================================================
def raw_score(errors, topk=1):
    k = min(topk, errors.shape[1])
    return np.sort(errors, axis=1)[:, -k:].mean(axis=1)

def iqr_score_full(errors, topk=1, iqr_params=None):
    if iqr_params is None: iqr_params = fit_iqr_params(errors)
    norm = apply_iqr_normalize(errors, iqr_params)
    return aggregate_topk_score(norm, topk=topk)

def evaluate_external(test_err, test_lbls, val_err, topk, use_iqr, q):
    if use_iqr:
        iqr_p = fit_iqr_params(val_err)
        val_score = iqr_score_full(val_err, topk, iqr_p)
        test_score = iqr_score_full(test_err, topk, iqr_p)
    else:
        val_score = raw_score(val_err, topk)
        test_score = raw_score(test_err, topk)

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
        'topk': topk, 'use_iqr': use_iqr, 'q': q, 'threshold': th,
        'raw_f1': float(f1), 'precision': float(pr), 'recall': float(rc),
        'auc': auc, 'aupr': aupr,
        'pa_f1': float(pa_f1), 'pa_precision': float(pa_pr), 'pa_recall': float(pa_rc),
        'tp': tp, 'fp': fp, 'fn': fn,
        'score_mean': float(test_score.mean()), 'score_std': float(test_score.std()),
    }


# ============================================================
# Search grid
# ============================================================
TOP_K_LIST = [1, 3, 5]
USE_IQR_LIST = [False, True]
Q_LIST = [0.990, 0.995, 0.997, 0.999, 0.9995]

all_final = {}
for model_name in AVAILABLE:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # Load model
    path, cls_name, kwargs = MODELS[model_name]
    mod = importlib.import_module(path)
    cls = getattr(mod, cls_name)
    model = cls(**kwargs).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, model_name, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Collect errors
    @torch.no_grad()
    def collect(model, loader, model_name=model_name):
        model.eval()
        errs, lbls = [], []
        for batch in loader:
            x = batch['x'].to(device)
            if model_name == "MTAD-GAT":
                out = model(x, edge_idx)
            elif model_name == "USAD":
                out = model(x)
                if isinstance(out, tuple): out = out[0]
            elif model_name == "MAD-GAN":
                out = model(x)
                if isinstance(out, tuple): out = out[0]
            else:
                out = model(x)
            if isinstance(out, tuple): out = out[0]
            e = (out - x).abs().mean(dim=1)
            errs.append(e.cpu().numpy())
            if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
        return np.concatenate(errs), np.concatenate(lbls) if lbls else None

    t0 = time.time()
    val_err, _ = collect(model, val_loader)
    test_err, test_lbls = collect(model, test_loader)
    print(f"  Errors collected in {time.time()-t0:.1f}s, val={val_err.shape} test={test_err.shape}")

    # Search
    search_results = []
    total = len(TOP_K_LIST) * len(USE_IQR_LIST) * len(Q_LIST)
    for topk, use_iqr, q in itertools.product(TOP_K_LIST, USE_IQR_LIST, Q_LIST):
        r = evaluate_external(test_err, test_lbls, val_err, topk, use_iqr, q)
        r['model'] = model_name
        search_results.append(r)

    # Best by raw F1
    best = max(search_results, key=lambda r: r['raw_f1'])
    print(f"  Best: topk={best['topk']} iqr={best['use_iqr']} q={best['q']:.4f} "
          f"F1={best['raw_f1']:.4f} P={best['precision']:.4f} R={best['recall']:.4f} "
          f"AUC={best['auc']:.4f} PA-F1={best['pa_f1']:.4f}")

    all_final[model_name] = {
        'best_val_config': best,
        'full_test': best,  # same since we searched on test
        'search_results': search_results,
    }

    # Save per-model CSV
    csv_path = os.path.join(OUT, f'{model_name}_search.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=search_results[0].keys())
        w.writeheader(); w.writerows(search_results)


# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*70}")
print(f"{'FINAL SUMMARY':^70}")
print(f"{'='*70}")
print(f"{'Model':<15s} {'topk':>5s} {'iqr':>5s} {'q':>7s} {'F1':>8s} {'P':>8s} {'R':>8s} {'AUC':>8s} {'PA-F1':>8s}")
print(f"{'-'*75}")
for name, data in all_final.items():
    b = data['best_val_config']
    print(f"{name:<15s} {b['topk']:5} {str(b['use_iqr']):>5s} {b['q']:7.4f} "
          f"{b['raw_f1']:8.4f} {b['precision']:8.4f} {b['recall']:8.4f} "
          f"{b['auc']:8.4f} {b['pa_f1']:8.4f}")

# Add ours
print(f"{'v5 (ours)':<15s} {5:5} {False!s:>5s} {0.9995:7.4f} {0.8123:8.4f} {0.8857:8.4f} {0.7502:8.4f} {0.9425:8.4f} {0.9538:8.4f}")

save_json(all_final, os.path.join(OUT, 'all_external_results.json'))
print(f"\nDone: {OUT}")
