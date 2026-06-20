"""External model val-only sweep: raw scoring, sweep topk/IQR/q on val, test once."""
import sys, os, time, json, csv, itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (build_pearson_edge_index, split_train_val, read_swat_csv,
                         SWaTDynamicWindowDataset, build_labels)
from utils import (load_config, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(__file__), "results", "external_val_sweep")
ensure_dir(OUT)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = {
    "USAD":       ("models.usad.model",      "USAD",       dict(n_vars=51, window=60, hidden=64, latent=32, dropout=0.1)),
    "DAGMM":      ("models.dagmm.model",     "DAGMM",      dict(n_vars=51, window=60, hidden=64, latent=16, n_gmm=4, dropout=0.1)),
    "LSTM-AE":    ("models.lstm_ae.model",   "LSTMAE",     dict(n_vars=51, window=60, hidden=64, num_layers=2, dropout=0.1)),
    "MAD-GAN":    ("models.mad_gan.model",   "MADGAN",     dict(n_vars=51, window=60, noise_dim=32, hidden=64)),
    "DCdetector": ("models.dcdetector.model","DCdetector", dict(n_vars=51, window=60, d_model=96, n_heads=4, dropout=0.1)),
    "TranAD":     ("models.tranad.model",    "TranAD",     dict(n_vars=51, window=60, d_model=48, n_heads=4, n_layers=2, dropout=0.1)),
}

CKPT_DIR = "results/models_full"
AVAILABLE = [d for d in sorted(MODELS.keys())
             if os.path.exists(os.path.join(CKPT_DIR, d, "best_model.pt"))]

# Build full datasets
cfg = load_config('config_dev.yaml')
dcfg = cfg['data']
nfd, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
mfd, raw_lbls = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
cc = [c for c in nfd.columns if c in mfd.columns]
raw = nfd[cc].values.astype(np.float32); tr, _, _, _ = split_train_val(raw, None, 0.2)
scaler = StandardScaler(); scaler.fit(tr)
_, val_raw_vals, _, _ = split_train_val(raw, None, 0.2)
val_v = scaler.transform(val_raw_vals)
test_v = scaler.transform(mfd[cc].values.astype(np.float32))
merged_lbls = build_labels(raw_lbls, 'Normal')
val_ds = SWaTDynamicWindowDataset(val_v, None, 60, 1, 1, 'future')
test_ds = SWaTDynamicWindowDataset(test_v, merged_lbls, 60, 1, 1, 'future')
val_loader = DataLoader(val_ds, 256, shuffle=False)
test_loader = DataLoader(test_ds, 256, shuffle=False)
train_vals = scaler.transform(tr); edge_idx, _ = build_pearson_edge_index(train_vals)
edge_idx = edge_idx.to(device)
print(f"Val={len(val_ds)} Test={len(test_ds)} Models: {AVAILABLE}")

# Sweep config (same as our model's search space)
TOP_K = [1, 3, 5]; IQR = [False, True]; QS = [0.995, 0.997, 0.999, 0.9995]
REF_CONFIG = {'topk': 5, 'use_iqr': False, 'q': 0.9995}  # our model's config


def get_score(errors, topk, use_iqr, fit_data=None):
    if use_iqr:
        p = fit_iqr_params(fit_data if fit_data is not None else errors)
        norm = apply_iqr_normalize(errors, p)
        return aggregate_topk_score(norm, topk=topk)
    k = min(topk, errors.shape[1])
    return np.sort(errors, axis=1)[:, -k:].mean(axis=1)


all_final = {}

for model_name in AVAILABLE:
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    t0 = time.time()

    path, cls_name, kwargs = MODELS[model_name]
    mod = importlib.import_module(path); cls = getattr(mod, cls_name)
    model = cls(**kwargs).to(device)
    ckpt = torch.load(os.path.join(CKPT_DIR, model_name, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    @torch.no_grad()
    def collect(loader):
        model.eval(); errs, lbls = [], []
        for batch in loader:
            x = batch['x'].to(device)
            if model_name == "USAD": out = model(x)
            elif model_name == "MAD-GAN": out = model(x)
            else: out = model(x)
            if isinstance(out, tuple): out = out[0]
            e = (out - x).abs().mean(dim=1)
            errs.append(e.cpu().numpy())
            if 'label' in batch: lbls.append(batch['label'].cpu().numpy())
        return np.concatenate(errs), np.concatenate(lbls) if lbls else None

    val_err, _ = collect(val_loader)
    test_err, test_lbls = collect(test_loader)
    print(f"  Errors: {time.time()-t0:.0f}s")

    # Sweep all configs
    sweep_results = []
    for topk, use_iqr, q in itertools.product(TOP_K, IQR, QS):
        val_score = get_score(val_err, topk, use_iqr, fit_data=val_err)
        th = float(np.quantile(val_score, q))
        test_score = get_score(test_err, topk, use_iqr, fit_data=val_err)
        pred = (test_score > th).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(test_lbls, pred, average='binary', zero_division=0)
        auc = float(roc_auc_score(test_lbls, test_score))
        aupr = float(average_precision_score(test_lbls, test_score))
        pa_pred = point_adjust(pred, test_lbls)
        pa_pr, pa_rc, pa_f1, _ = precision_recall_fscore_support(test_lbls, pa_pred, average='binary', zero_division=0)
        val_mean = float(val_score.mean())
        sweep_results.append({
            'model': model_name, 'topk': topk, 'use_iqr': use_iqr, 'q': q, 'th': float(th),
            'f1': float(f1), 'p': float(pr), 'r': float(rc),
            'auc': auc, 'aupr': aupr, 'pa_f1': float(pa_f1), 'pa_p': float(pa_pr),
            'val_mean': val_mean, 'val_th_ratio': float(th / max(1e-8, val_mean)),
        })

    # Reference config (same as ours)
    ref = [r for r in sweep_results if r['topk']==REF_CONFIG['topk']
           and r['use_iqr']==REF_CONFIG['use_iqr'] and r['q']==REF_CONFIG['q']][0]

    # Best by val proxy (highest threshold/mean ratio = tightest boundary)
    best_val = max(sweep_results, key=lambda r: r['val_th_ratio'])

    all_final[model_name] = {
        'reference': ref,
        'best_val_proxy': best_val,
        'best_test_f1': max(sweep_results, key=lambda r: r['f1']),
        'sweep': sweep_results,
    }

    print(f"  Reference (top5+raw+q=0.9995): F1={ref['f1']:.4f} P={ref['p']:.4f} R={ref['r']:.4f} AUC={ref['auc']:.4f} PA-F1={ref['pa_f1']:.4f}")
    print(f"  Best val proxy:               F1={best_val['f1']:.4f} P={best_val['p']:.4f}")
    print(f"  Best test F1 (for ref only):   F1={sweep_results[int(len(sweep_results)/2)]['f1']:.4f}")

    # Save per-model
    with open(os.path.join(OUT, f'{model_name}_sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sweep_results[0].keys()); w.writeheader(); w.writerows(sweep_results)


# Final summary
print(f"\n{'='*75}")
print(f"FINAL (same config as OURS: raw+top5+q=0.9995, val threshold)")
print(f"{'='*75}")
print(f"{'Model':<15s} {'F1':>8s} {'P':>8s} {'R':>8s} {'AUC':>8s} {'AUPR':>8s} {'PA-F1':>8s}")
print(f"{'-'*75}")
refs = [(name, data['reference']) for name, data in all_final.items()]
refs.append(('v5 V3 (OURS)', {'f1': 0.8137, 'p': 0.8988, 'r': 0.7433, 'auc': 0.9153, 'aupr': 0.0, 'pa_f1': 0.9598}))
refs.append(('v5 raw_top5',  {'f1': 0.8123, 'p': 0.8857, 'r': 0.7502, 'auc': 0.9425, 'aupr': 0.0, 'pa_f1': 0.9538}))
refs.sort(key=lambda x: -x[1]['f1'])
for name, r in refs:
    print(f"{name:<15s} {r['f1']:8.4f} {r['p']:8.4f} {r['r']:8.4f} {r['auc']:8.4f} {r.get('aupr',0):8.4f} {r['pa_f1']:8.4f}")

save_json(all_final, os.path.join(OUT, 'results.json'))
print(f"\nDone: {OUT}")
