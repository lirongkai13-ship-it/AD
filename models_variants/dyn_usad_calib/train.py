"""训练: Dynamic USAD + Calibrated Scoring (EMA-IQR normalization)
策略：不只在验证集上一次性拟合 IQR，而是用训练集正常数据的 EMA 来估计 IQR 参数，
使得异常评分在动态图变化时更鲁棒。
"""
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
from variant_model import DynCalib_USAD


def ema_fit_iqr(train_errors, alpha=0.01):
    """用 EMA 估计逐变量的 Q1, Q3, IQR (仅正常训练数据)"""
    N = train_errors.shape[1]
    q1_ema = np.percentile(train_errors, 25, axis=0)
    q3_ema = np.percentile(train_errors, 75, axis=0)

    # EMA 逐样本更新
    for i in range(len(train_errors)):
        q1_sample = np.percentile(train_errors[i:i+1], 25, axis=0)
        q3_sample = np.percentile(train_errors[i:i+1], 75, axis=0)
        q1_ema = (1 - alpha) * q1_ema + alpha * q1_sample
        q3_ema = (1 - alpha) * q3_ema + alpha * q3_sample

    iqr_ema = q3_ema - q1_ema
    iqr_ema = np.maximum(iqr_ema, 1e-8)
    return {"q1": q1_ema, "q3": q3_ema, "iqr": iqr_ema, "method": "ema"}


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "cuda"))
    save_dir = os.path.join(cfg["output"]["save_dir"], "dyn_usad_calib")
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

    model = DynCalib_USAD(
        info["num_variables"], int(cfg["data"]["window_size"]), static_ei,
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
        print(f"Epoch {epoch:03d} (P{phase}) | train {train_loss:.6f} | val {val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0
            torch.save({"model": model.state_dict()}, os.path.join(save_dir, "best_model.pt"))
        else:
            no_improve += 1
        if no_improve >= early_stop: break

    train_time = time.time() - t_start
    save_json({"history": history, "best_val_loss": float(best_val)},
              os.path.join(save_dir, "train_history.json"))

    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    eval_forward = lambda batch: model.forward_eval(batch["x"].to(device), static_ei)

    # 收集训练集正常数据的重构误差用于 EMA-IQR
    train_errors = []
    for batch in tqdm(train_loader, desc="collect train errors"):
        with torch.no_grad():
            x = batch["x"].to(device); recon = eval_forward(batch)
            train_errors.append((recon - x).abs().mean(dim=1).cpu().numpy())
    train_errors = np.concatenate(train_errors)

    # 收集验证集 + 测试集误差
    val_errors = []
    for batch in tqdm(val_loader, desc="eval val"):
        with torch.no_grad():
            x = batch["x"].to(device); recon = eval_forward(batch)
            val_errors.append((recon - x).abs().mean(dim=1).cpu().numpy())
    val_errors = np.concatenate(val_errors)

    all_errors, all_labels = [], []
    for batch in tqdm(test_loader, desc="eval test"):
        with torch.no_grad():
            x = batch["x"].to(device); recon = eval_forward(batch)
            all_errors.append((recon - x).abs().mean(dim=1).cpu().numpy())
            if "label" in batch: all_labels.append(batch["label"].cpu().numpy())
    test_errors = np.concatenate(all_errors); test_labels = np.concatenate(all_labels)

    # ── 方法1: 标准 IQR (同原版) ──
    iqr_std = fit_iqr_params(val_errors)
    val_norm_std = apply_iqr_normalize(val_errors, iqr_std)
    thresh_std = float(np.quantile(aggregate_topk_score(val_norm_std, 5), 0.995))
    test_norm_std = apply_iqr_normalize(test_errors, iqr_std)
    test_score_std = aggregate_topk_score(test_norm_std, topk=5)
    test_pred_std = (test_score_std > thresh_std).astype(int)
    raw_std = binary_metrics(test_labels, test_pred_std, test_score_std)

    # ── 方法2: EMA-IQR (校准后) ──
    iqr_ema = ema_fit_iqr(train_errors, alpha=0.01)
    val_norm_ema = apply_iqr_normalize(val_errors, iqr_ema)
    thresh_ema = float(np.quantile(aggregate_topk_score(val_norm_ema, 5), 0.995))
    test_norm_ema = apply_iqr_normalize(test_errors, iqr_ema)
    test_score_ema = aggregate_topk_score(test_norm_ema, topk=5)
    test_pred_ema = (test_score_ema > thresh_ema).astype(int)
    raw_ema = binary_metrics(test_labels, test_pred_ema, test_score_ema)

    metrics = {
        "threshold_std": thresh_std, "raw_std": raw_std,
        "threshold_ema": thresh_ema, "raw_ema": raw_ema,
        "train_time_min": train_time/60
    }
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    print(f"\nStandard IQR:  F1={raw_std['f1']:.4f}  AUC={raw_std.get('roc_auc','?')}")
    print(f"EMA-IQR (calib): F1={raw_ema['f1']:.4f}  AUC={raw_ema.get('roc_auc','?')}")
    return metrics

if __name__ == "__main__":
    main()
