"""Full-setting training for comparison models.
Usage:
  python train_full.py --model tri_branch
  python train_full.py --model baseline
  python train_full.py --model prior
"""
import sys, os, time, json, argparse, importlib.util
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
TQDM_KWARGS = {'disable': not sys.stdout.isatty(), 'file': sys.stdout, 'leave': False}
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import (prepare_data, build_pearson_edge_index, split_train_val, read_swat_csv)
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support)
from sklearn.preprocessing import StandardScaler

# ── Eval helper ──
def binary_metrics(labels, pred, score):
    pr, rc, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    return {
        'precision': float(pr), 'recall': float(rc), 'f1': float(f1),
        'roc_auc': float(roc_auc_score(labels, score)),
        'pr_auc': float(average_precision_score(labels, score)),
    }

# ── Build graphs ──
def build_graphs(cfg):
    dcfg = cfg['data']
    normal_df, _ = read_swat_csv(dcfg['train_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    merged_df, _ = read_swat_csv(dcfg['test_csv'], dcfg.get('timestamp_col'), dcfg.get('label_col'))
    common_cols = [c for c in normal_df.columns if c in merged_df.columns]

    normal_raw = normal_df[common_cols].values.astype(np.float32)
    train_raw, _, _, _ = split_train_val(normal_raw, None, 0.2)
    scaler = StandardScaler(); train_vals = scaler.fit_transform(train_raw)
    static_ei, _ = build_pearson_edge_index(train_vals)

    # Prior graph
    bgp_path = os.path.join(os.path.dirname(__file__), 'models_variants', 'prior_fusion', 'build_prior_graph.py')
    spec = importlib.util.spec_from_file_location('bpg', bgp_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    prior_ei, prior_w = mod.build_prior_graph(common_cols)

    return static_ei, prior_ei, prior_w, common_cols


# ═══════════════════════════════════════════════════
# TRAIN: tri_branch
# ═══════════════════════════════════════════════════
def train_tri_branch(cfg, device, save_dir):
    from models_variants.tri_branch.variant_model import TriBranch_USAD

    train_ds, val_ds, test_ds, _, info = prepare_data(cfg)
    bs = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    static_ei, prior_ei, prior_w, common_cols = build_graphs(cfg)
    static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

    model = TriBranch_USAD(
        nv=info['num_variables'], ws=int(cfg['data']['window_size']),
        static_edge_index=static_ei, prior_edge_index=prior_ei, prior_weights=prior_w,
        hidden_dim=int(cfg['model']['hidden_dim']), gat_heads=int(cfg['model']['gat_heads']),
        gru_hidden=int(cfg['model']['gru_hidden']), tcn_channels=int(cfg['model']['tcn_channels']),
        tcn_blocks=int(cfg['model'].get('tcn_blocks', 1)),
        dropout=float(cfg['model']['dropout']),
        encoder_mode='tri_branch_residual_gate',
        temporal_mode='per_variable_conv',
        gamma_mode='fixed', gamma_value=0.05, gate_scale=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']),
                                 weight_decay=float(cfg['train']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(cfg['train']['lr_patience']), min_lr=1e-6)
    mse = nn.MSELoss()
    epochs = int(cfg['train']['epochs'])
    early_stop = int(cfg['train']['early_stop_patience'])
    phase1_epochs = max(1, epochs - 2)

    print(f"Tri-Branch: {sum(p.numel() for p in model.parameters()):,} params")
    return _train_usad(model, train_loader, val_loader, test_loader, static_ei,
                       optimizer, scheduler, mse, epochs, early_stop, phase1_epochs,
                       device, save_dir, cfg)


# ═══════════════════════════════════════════════════
# TRAIN: parallel_usad_prior
# ═══════════════════════════════════════════════════
def train_prior(cfg, device, save_dir):
    from models_variants.parallel_usad_prior.variant_model import ParallelPrior_USAD

    train_ds, val_ds, test_ds, _, info = prepare_data(cfg)
    bs = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    static_ei, prior_ei, prior_w, common_cols = build_graphs(cfg)
    static_ei = static_ei.to(device); prior_ei = prior_ei.to(device); prior_w = prior_w.to(device)

    model = ParallelPrior_USAD(
        nv=info['num_variables'], ws=int(cfg['data']['window_size']),
        static_edge_index=static_ei, prior_edge_index=prior_ei, prior_weights=prior_w,
        hidden_dim=int(cfg['model']['hidden_dim']), gat_heads=int(cfg['model']['gat_heads']),
        gru_hidden=int(cfg['model']['gru_hidden']), tcn_channels=int(cfg['model']['tcn_channels']),
        tcn_blocks=int(cfg['model'].get('tcn_blocks', 1)),
        dropout=float(cfg['model']['dropout']),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']),
                                 weight_decay=float(cfg['train']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(cfg['train']['lr_patience']), min_lr=1e-6)
    mse = nn.MSELoss()
    epochs = int(cfg['train']['epochs'])
    early_stop = int(cfg['train']['early_stop_patience'])
    phase1_epochs = max(1, epochs - 2)

    print(f"Parallel Prior USAD: {sum(p.numel() for p in model.parameters()):,} params")
    return _train_usad(model, train_loader, val_loader, test_loader, static_ei,
                       optimizer, scheduler, mse, epochs, early_stop, phase1_epochs,
                       device, save_dir, cfg)


# ═══════════════════════════════════════════════════
# TRAIN: baseline GATv2+TCN+GRU
# ═══════════════════════════════════════════════════
def train_baseline(cfg, device, save_dir):
    from model import GATv2TCNGRUDetector

    train_ds, val_ds, test_ds, edge_index, info = prepare_data(cfg)
    bs = int(cfg['train']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    edge_index = edge_index.to(device)

    model = GATv2TCNGRUDetector(
        num_variables=info['num_variables'],
        window_size=int(cfg['data']['window_size']),
        hidden_dim=int(cfg['model']['hidden_dim']),
        gat_heads=int(cfg['model']['gat_heads']),
        gru_hidden=int(cfg['model']['gru_hidden']),
        tcn_channels=int(cfg['model']['tcn_channels']),
        tcn_blocks=int(cfg['model'].get('tcn_blocks', 2)),
        dropout=float(cfg['model']['dropout']),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['train']['lr']),
                                 weight_decay=float(cfg['train']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(cfg['train']['lr_patience']), min_lr=1e-6)
    mse = nn.MSELoss()
    epochs = int(cfg['train']['epochs'])
    early_stop = int(cfg['train']['early_stop_patience'])
    lambda_pred = float(cfg['train']['lambda_pred'])

    print(f"Baseline GATv2+TCN+GRU: {sum(p.numel() for p in model.parameters()):,} params")
    return _train_baseline(model, train_loader, val_loader, test_loader, edge_index,
                           optimizer, scheduler, mse, epochs, early_stop, lambda_pred,
                           device, save_dir, cfg)


# ═══════════════════════════════════════════════════
# USAD training loop
# ═══════════════════════════════════════════════════
def _train_usad(model, train_loader, val_loader, test_loader, static_ei,
                optimizer, scheduler, mse, epochs, early_stop, phase1_epochs,
                device, save_dir, cfg):
    best_val = float('inf'); no_improve = 0; history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        phase = 1 if epoch <= phase1_epochs else 2
        pbar = tqdm(train_loader, desc=f'E{epoch}/{epochs}',
                     **TQDM_KWARGS)
        for batch in pbar:
            x = batch['x'].to(device); optimizer.zero_grad()
            r1, r2, r12 = model(x, static_ei)
            loss_r1, loss_r2, loss_r12 = mse(r1, x), mse(r2, x), mse(r12, x)
            if phase == 1:
                loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.1*mse(r1, r2.detach())
            else:
                loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.05*torch.abs(loss_r1 - loss_r2)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'phase': phase})

        train_loss /= len(train_loader.dataset)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                r1 = model.forward_eval(x, static_ei)
                val_loss += mse(r1, x).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f'Epoch {epoch:03d} (P{phase}) | train {train_loss:.6f} | val {val_loss:.6f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save({'model': model.state_dict()}, os.path.join(save_dir, 'best_model.pt'))
        else:
            no_improve += 1
        if no_improve >= early_stop: break

    train_time = time.time() - t_start
    save_json({'history': history, 'best_val_loss': float(best_val)},
              os.path.join(save_dir, 'train_history.json'))

    # ── Evaluate (val_th, IQR+k=1, q=0.995) ──
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    val_errors, test_errors, test_labels = _collect_usad_errors(model, val_loader, test_loader, static_ei, device)
    return _evaluate_iqr(val_errors, test_errors, test_labels, cfg, save_dir, train_time)


def _collect_usad_errors(model, val_loader, test_loader, static_ei, device):
    val_errs, test_errs, test_lbls = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            r1 = model.forward_eval(x, static_ei)
            val_errs.append((r1 - x).abs().mean(dim=1).cpu().numpy())
        for batch in test_loader:
            x = batch['x'].to(device)
            r1 = model.forward_eval(x, static_ei)
            test_errs.append((r1 - x).abs().mean(dim=1).cpu().numpy())
            if 'label' in batch: test_lbls.append(batch['label'].cpu().numpy())
    return np.concatenate(val_errs), np.concatenate(test_errs), np.concatenate(test_lbls)


def _evaluate_iqr(val_errors, test_errors, test_labels, cfg, save_dir, train_time):
    topk = int(cfg['score']['topk'])
    q = float(cfg['score']['threshold_quantile'])

    iqr_params = fit_iqr_params(val_errors)
    val_norm = apply_iqr_normalize(val_errors, iqr_params)
    val_score = aggregate_topk_score(val_norm, topk=topk)
    threshold = float(np.quantile(val_score, q))

    test_norm = apply_iqr_normalize(test_errors, iqr_params)
    test_score = aggregate_topk_score(test_norm, topk=topk)
    test_pred = (test_score > threshold).astype(int)

    raw = binary_metrics(test_labels, test_pred, test_score)
    pa_pred = point_adjust(test_pred, test_labels)
    pa = binary_metrics(test_labels, pa_pred, test_score)

    metrics = {'threshold': threshold, 'topk': topk, 'q': q, 'raw': raw, 'point_adjust': pa,
               'train_time_min': round(train_time/60, 2)}
    save_json(metrics, os.path.join(save_dir, 'metrics.json'))

    print(f"\n=== Results ===")
    print(f"Raw F1: {raw['f1']:.4f}  P={raw['precision']:.4f}  R={raw['recall']:.4f}")
    print(f"AUC: {raw['roc_auc']:.4f}  AUPR: {raw['pr_auc']:.4f}")
    print(f"PA  F1: {pa['f1']:.4f}")
    print(f"Train time: {train_time/60:.1f}min")
    return metrics


# ═══════════════════════════════════════════════════
# Baseline training loop
# ═══════════════════════════════════════════════════
def _train_baseline(model, train_loader, val_loader, test_loader, edge_index,
                    optimizer, scheduler, mse, epochs, early_stop, lambda_pred,
                    device, save_dir, cfg):
    best_val = float('inf'); no_improve = 0; history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'E{epoch}/{epochs}',
                     **TQDM_KWARGS)
        for batch in pbar:
            x = batch['x'].to(device)
            y_future = batch['y_future'].to(device)
            y_recon = batch['y_recon'].to(device)
            optimizer.zero_grad()
            pred, recon = model(x, edge_index)
            loss_pred = mse(pred, y_future)
            loss_recon = mse(recon, y_recon)
            loss = lambda_pred * loss_pred + (1.0 - lambda_pred) * loss_recon
            loss.backward(); optimizer.step()
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader.dataset)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                y_future = batch['y_future'].to(device)
                y_recon = batch['y_recon'].to(device)
                pred, recon = model(x, edge_index)
                val_loss += (lambda_pred * mse(pred, y_future) + (1.0 - lambda_pred) * mse(recon, y_recon)).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f'Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save({'model': model.state_dict()}, os.path.join(save_dir, 'best_model.pt'))
        else:
            no_improve += 1
        if no_improve >= early_stop: break

    train_time = time.time() - t_start
    save_json({'history': history, 'best_val_loss': float(best_val)},
              os.path.join(save_dir, 'train_history.json'))

    # Evaluate
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    val_err_list, test_err_list, test_lbl_list = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x'].to(device)
            pred, recon = model(x, edge_index)
            err = lambda_pred * (pred - batch['y_future'].to(device)).abs() + (1.0-lambda_pred) * (recon - batch['y_recon'].to(device)).abs().mean(dim=1)
            val_err_list.append(err.cpu().numpy())
        for batch in test_loader:
            x = batch['x'].to(device)
            pred, recon = model(x, edge_index)
            err = lambda_pred * (pred - batch['y_future'].to(device)).abs() + (1.0-lambda_pred) * (recon - batch['y_recon'].to(device)).abs().mean(dim=1)
            test_err_list.append(err.cpu().numpy())
            if 'label' in batch: test_lbl_list.append(batch['label'].cpu().numpy())

    val_errs = np.concatenate(val_err_list)
    test_errs = np.concatenate(test_err_list)
    test_lbls = np.concatenate(test_lbl_list)

    return _evaluate_iqr(val_errs, test_errs, test_lbls, cfg, save_dir, train_time)


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['tri_branch', 'baseline', 'prior'])
    parser.add_argument('--config', type=str, default='config_full.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg['train']['seed']))
    device = get_device(cfg['train'].get('device', 'cuda'))

    save_dir = os.path.join(cfg['output']['save_dir'], args.model)
    ensure_dir(save_dir)

    print(f'{"="*60}')
    print(f'Full-Setting Training: {args.model}')
    print(f'Config: {args.config}  |  stride=1  |  dim=32  |  heads=2  |  k=1')
    print(f'Output: {save_dir}')
    print(f'{"="*60}')

    if args.model == 'tri_branch':
        train_tri_branch(cfg, device, save_dir)
    elif args.model == 'baseline':
        train_baseline(cfg, device, save_dir)
    elif args.model == 'prior':
        train_prior(cfg, device, save_dir)

    print(f'\nDone! Model saved to {save_dir}')
