"""Val-only threshold, fixed scoring: raw + topk=5 + q=0.9995 (same as v5)."""
import sys, os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (build_pearson_edge_index,
                         split_train_val, read_swat_csv, SWaTDynamicWindowDataset, build_labels)
from utils import (load_config, ensure_dir, fit_iqr_params, apply_iqr_normalize,
                   aggregate_topk_score, point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "results", "external_val_only")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

CKPT_DIR = "results/models_full"
AVAILABLE = [d for d in os.listdir(CKPT_DIR)
             if os.path.isdir(os.path.join(CKPT_DIR, d))
             and os.path.exists(os.path.join(CKPT_DIR, d, "best_model.pt"))
             and d in MODELS]

# Build datasets
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

val_ds = SWaTDynamicWindowDataset(val_vals, None, 60, 1, 1, 'future')
test_ds = SWaTDynamicWindowDataset(test_vals, merged_lbls, 60, 1, 1, 'future')
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)

train_vals = scaler.transform(train_raw)
edge_idx, _ = build_pearson_edge_index(train_vals); edge_idx = edge_idx.to(device)

# Fixed scoring: raw + topk=5 (same as v5 best)
TOP_K = 5; Q = 0.9995

print(f"Scoring: raw + topk={TOP_K} + q={Q}")
print(f"Models: {len(AVAILABLE)}: {AVAILABLE}\n")

results = {}

for model_name in AVAILABLE:
    print(f"  {model_name}...", end=' ', flush=True)
    t0 = time.time()

    # Load
    path, cls_name, kwargs = MODELS[model_name]
    mod = importlib.import_module(path); cls = getattr(mod, cls_name)
    model = cls(**kwargs).to(device)
    ckpt = torch.load(os.path.join(CKPT_DIR, model_name, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    # Collect errors
    @torch.no_grad()
    def collect(loader):
        model.eval(); errs, lbls = [], []
        for batch in loader:
            x = batch['x'].to(device)
            if model_name == "MTAD-GAT": out = model(x, edge_idx)
            elif model_name == "USAD": out = model(x)
            elif model_name == "MAD-GAN": out = model(x)
            else: out = model(x)
            if isinstance(out, tuple): out = out[0]
            e = (out - x).abs().mean(dim=1)
            errs.append(e.cpu().numpy())
            if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
        return np.concatenate(errs), np.concatenate(lbls) if lbls else None

    val_err, _ = collect(val_loader)
    test_err, test_lbls = collect(test_loader)

    # Score: raw top-k
    k = min(TOP_K, test_err.shape[1])
    val_score = np.sort(val_err, axis=1)[:, -k:].mean(axis=1)
    test_score = np.sort(test_err, axis=1)[:, -k:].mean(axis=1)

    # Threshold from VAL
    th = float(np.quantile(val_score, Q))

    # Predict on TEST (one shot)
    pred = (test_score > th).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
    auc = float(roc_auc_score(test_lbls, test_score))
    aupr = float(average_precision_score(test_lbls, test_score))
    pa_pred = point_adjust(pred, test_lbls)
    pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
    tp = int(((pred==1)&(test_lbls==1)).sum())
    fp = int(((pred==1)&(test_lbls==0)).sum())
    fn = int(((pred==0)&(test_lbls==1)).sum())

    r = {'f1': float(f1), 'p': float(pr), 'r': float(rc), 'auc': auc, 'aupr': aupr,
         'pa_f1': float(pa_f1), 'pa_p': float(pa_pr), 'pa_r': float(pa_rc),
         'tp': tp, 'fp': fp, 'fn': fn, 'th': th, 'time_s': time.time()-t0}
    results[model_name] = r
    print(f"F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f} AUC={r['auc']:.4f} PA-F1={r['pa_f1']:.4f} ({r['time_s']:.0f}s)")

# Summary
print(f"\n{'='*75}")
print(f"FINAL (raw+top5+q=0.9995, val threshold, test one shot)")
print(f"{'='*75}")
print(f"{'Model':<15s} {'F1':>8s} {'P':>8s} {'R':>8s} {'AUC':>8s} {'PA-F1':>8s} {'TP':>6s}")
print(f"{'-'*70}")
for name in sorted(results, key=lambda n: -results[n]['f1']):
    r = results[name]
    print(f"{name:<15s} {r['f1']:8.4f} {r['p']:8.4f} {r['r']:8.4f} {r['auc']:8.4f} {r['pa_f1']:8.4f} {r['tp']:6d}")
print(f"{'v5 (OURS)':<15s} {0.8123:8.4f} {0.8857:8.4f} {0.7502:8.4f} {0.9425:8.4f} {0.9538:8.4f}")

save_json(results, os.path.join(OUT, 'results_fixed_scoring.json'))
print(f"\nDone: {OUT}")
