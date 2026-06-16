"""训练: Dynamic USAD + Weighted Fusion (alpha sweep)"""
import sys, os, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_loader import prepare_data
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from evaluate import binary_metrics
from variant_model import DynWeighted_USAD


def main(alpha=0.5):
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "cuda"))
    save_dir = os.path.join(cfg["output"]["save_dir"], f"dyn_usad_weighted_a{alpha:.2f}".replace(".",""))
    ensure_dir(save_dir)

    train_ds, val_ds, test_ds, _, info = prepare_data(cfg)
    bs = int(cfg["train"]["batch_size"]); nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    from data_loader import build_pearson_edge_index, split_train_val, read_swat_csv
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    dcfg = cfg["data"]
    normal_df = pd.read_csv(dcfg["train_csv"]); normal_df.columns=[str(c).strip() for c in normal_df.columns]
    if dcfg.get("timestamp_col") in normal_df.columns: normal_df=normal_df.drop(columns=[dcfg["timestamp_col"]])
    if dcfg.get("label_col") in normal_df.columns: normal_df=normal_df.drop(columns=[dcfg["label_col"]])
    merged_df = pd.read_csv(dcfg["test_csv"]); merged_df.columns=[str(c).strip() for c in merged_df.columns]
    common_cols = [c for c in normal_df.columns if c in merged_df.columns]
    normal_raw = normal_df[common_cols].values.astype(np.float32)
    train_raw, _, _, _ = split_train_val(normal_raw, None, 0.2)
    scaler_fit = StandardScaler(); train_vals = scaler_fit.fit_transform(train_raw)
    static_ei, _ = build_pearson_edge_index(train_vals)

    # 计算全局相关矩阵 (训练集全量)
    global_corr = torch.tensor(np.corrcoef(train_vals.T), dtype=torch.float32)
    global_corr = torch.nan_to_num(global_corr, nan=0.0).abs()
    print(f"Global corr matrix: {global_corr.shape}, alpha={alpha}")

    model = DynWeighted_USAD(
        info["num_variables"], int(cfg["data"]["window_size"]), static_ei, global_corr, alpha,
        int(cfg["model"]["hidden_dim"]), int(cfg["model"]["gat_heads"]),
        int(cfg["model"]["gru_hidden"]), int(cfg["model"]["tcn_channels"]),
        int(cfg["model"].get("tcn_blocks", 1)), float(cfg["model"]["dropout"]),
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]),
                                 weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
    use_amp = cfg["train"].get("amp", False)
    scaler = torch.GradScaler("cuda") if use_amp else None
    mse = nn.MSELoss()

    best_val = float("inf"); no_improve = 0; history = []
    early_stop = 10; epochs = int(cfg["train"]["epochs"])
    phase1_epochs = max(1, epochs - 2)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        phase = 1 if epoch <= phase1_epochs else 2
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            x = batch["x"].to(device); optimizer.zero_grad()
            if use_amp and scaler:
                with torch.autocast("cuda"):
                    r1, r2, r12 = model(x, static_ei)
                    loss_r1, loss_r2, loss_r12 = mse(r1, x), mse(r2, x), mse(r12, x)
                    if phase == 1:
                        loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.1*mse(r1, r2.detach())
                    else:
                        loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.05*torch.abs(loss_r1-loss_r2)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                r1, r2, r12 = model(x, static_ei)
                loss_r1, loss_r2, loss_r12 = mse(r1, x), mse(r2, x), mse(r12, x)
                if phase == 1:
                    loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.1*mse(r1, r2.detach())
                else:
                    loss = loss_r1 + loss_r2 + 0.5*loss_r12 + 0.05*torch.abs(loss_r1-loss_r2)
                loss.backward(); optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                x = batch["x"].to(device)
                r1 = model.forward_eval(x, static_ei)
                val_loss += mse(r1, x).item() * x.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} (P{phase}, alpha={alpha:.2f}) | train {train_loss:.6f} | val {val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save({"model": model.state_dict()}, os.path.join(save_dir, "best_model.pt"))
        else:
            no_improve += 1
        if no_improve >= early_stop: break

    train_time = time.time() - t_start
    save_json({"history": history, "best_val_loss": float(best_val), "alpha": alpha},
              os.path.join(save_dir, "train_history.json"))

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    eval_forward = lambda batch: model.forward_eval(batch["x"].to(device), static_ei)

    all_errors, all_labels = [], []
    for batch in tqdm(test_loader, desc="eval test"):
        with torch.no_grad():
            x = batch["x"].to(device); recon = eval_forward(batch)
            all_errors.append((recon - x).abs().mean(dim=1).cpu().numpy())
            if "label" in batch: all_labels.append(batch["label"].cpu().numpy())
    test_errors = np.concatenate(all_errors); test_labels = np.concatenate(all_labels)
    val_errors = []
    for batch in tqdm(val_loader, desc="eval val"):
        with torch.no_grad():
            x = batch["x"].to(device); recon = eval_forward(batch)
            val_errors.append((recon - x).abs().mean(dim=1).cpu().numpy())
    val_errors = np.concatenate(val_errors)

    iqr_params = fit_iqr_params(val_errors)
    val_norm = apply_iqr_normalize(val_errors, iqr_params)
    threshold = float(np.quantile(aggregate_topk_score(val_norm, 5), 0.995))
    test_norm = apply_iqr_normalize(test_errors, iqr_params)
    test_score = aggregate_topk_score(test_norm, topk=5)
    test_pred = (test_score > threshold).astype(int)

    raw_m = binary_metrics(test_labels, test_pred, test_score)
    pa_pred = point_adjust(test_pred, test_labels)
    pa_m = binary_metrics(test_labels, pa_pred, test_score)
    metrics = {"alpha": alpha, "threshold": threshold, "raw": raw_m, "point_adjust": pa_m,
               "train_time_min": train_time/60}
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    print(f"\nWeighted alpha={alpha:.2f}: F1={raw_m['f1']:.4f}  AUC={raw_m.get('roc_auc','?')}")
    return metrics

if __name__ == "__main__":
    # 扫描 alpha 范围
    for a in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}\n  Weighted Fusion: alpha={a}\n{'='*60}")
        main(alpha=a)
