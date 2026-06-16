"""变体模型统一训练器"""
import os, sys, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import prepare_data
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from evaluate import collect_var_errors, binary_metrics


def train_variant(model_class, variant_name, model_kwargs=None, save_dir=None, config_path=None):
    """训练单个变体模型并返回指标"""
    if model_kwargs is None:
        model_kwargs = {}
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config_dev.yaml")
    cfg = load_config(config_path)
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "cuda"))

    if save_dir is None:
        save_dir = os.path.join(cfg["output"]["save_dir"], variant_name)
    ensure_dir(save_dir)

    # 数据
    train_ds, val_ds, test_ds, edge_index, info = prepare_data(cfg)
    edge_index = edge_index.to(device)
    bs = int(cfg["train"]["batch_size"]); nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # 模型
    base_kwargs = dict(num_variables=info["num_variables"],
                       window_size=int(cfg["data"]["window_size"]),
                       hidden_dim=int(cfg["model"]["hidden_dim"]),
                       gat_heads=int(cfg["model"]["gat_heads"]),
                       gru_hidden=int(cfg["model"]["gru_hidden"]),
                       tcn_channels=int(cfg["model"]["tcn_channels"]),
                       tcn_blocks=int(cfg["model"].get("tcn_blocks", 1)),
                       dropout=float(cfg["model"]["dropout"]))
    base_kwargs.update(model_kwargs)
    model = model_class(**base_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]),
                                 weight_decay=float(cfg["train"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg["train"].get("lr_patience", 3), min_lr=1e-6)
    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    scaler = torch.GradScaler("cuda") if use_amp else None
    mse = nn.MSELoss()
    early_stop = cfg["train"].get("early_stop_patience", 5)

    best_val_loss = float("inf"); epochs_no_improve = 0; history = []
    print(f"\n{'='*60}")
    print(f"  {variant_name}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}  |  AMP: {use_amp}")
    print(f"  Samples: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,}")
    print(f"{'='*60}")

    t_start = time.time()
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train(); train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            x = batch["x"].to(device); yf = batch["y_future"].to(device); yr = batch["y_recon"].to(device)
            optimizer.zero_grad()
            if use_amp and scaler:
                with torch.autocast("cuda"):
                    pred, recon = model(x, edge_index)
                    loss = 0.5*mse(pred, yf) + 0.5*mse(recon, yr)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                pred, recon = model(x, edge_index)
                loss = 0.5*mse(pred, yf) + 0.5*mse(recon, yr)
                loss.backward(); optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="val", leave=False):
                x = batch["x"].to(device); yf = batch["y_future"].to(device); yr = batch["y_recon"].to(device)
                pred, recon = model(x, edge_index)
                val_loss += (0.5*mse(pred, yf) + 0.5*mse(recon, yr)).item() * x.size(0)
        val_loss /= len(val_ds)

        lr0 = optimizer.param_groups[0]["lr"]; scheduler.step(val_loss); lr1 = optimizer.param_groups[0]["lr"]
        lr_s = f"| lr {lr0:.2e}" + (f" -> {lr1:.2e}" if lr1 < lr0 else "")
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} {lr_s}")
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "best_val_loss": float(val_loss)},
                       os.path.join(save_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= early_stop:
            print(f"Early stopping at epoch {epoch}"); break

    train_time = time.time() - t_start
    save_json({"history": history, "best_val_loss": float(best_val_loss)},
              os.path.join(save_dir, "train_history.json"))
    print(f"Training finished in {train_time/60:.1f}min")

    # 评估
    print("Evaluating...")
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # 处理 model.state_dict() 被嵌套在 'model' key 中的情况
        model.load_state_dict(state["model"])
    model.eval()

    val_errors, _ = collect_var_errors(model, val_loader, edge_index, device, lambda_pred=0.5)
    iqr_params = fit_iqr_params(val_errors)
    val_norm = apply_iqr_normalize(val_errors, iqr_params)
    val_score = aggregate_topk_score(val_norm, topk=5)
    threshold = float(np.quantile(val_score, 0.995))

    test_errors, test_labels = collect_var_errors(model, test_loader, edge_index, device, lambda_pred=0.5)
    test_norm = apply_iqr_normalize(test_errors, iqr_params)
    test_score = aggregate_topk_score(test_norm, topk=5)
    test_pred = (test_score > threshold).astype(int)

    raw_m = binary_metrics(test_labels, test_pred, test_score)
    pa_pred = point_adjust(test_pred, test_labels)
    pa_m = binary_metrics(test_labels, pa_pred, test_score)

    metrics = {"threshold": float(threshold), "raw": raw_m, "point_adjust": pa_m,
               "train_time_min": train_time/60}
    save_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"Raw F1: {raw_m['f1']:.4f}  P={raw_m['precision']:.4f}  R={raw_m['recall']:.4f}  AUC={raw_m.get('roc_auc','?')}")
    print(f"PA  F1: {pa_m['f1']:.4f}")
    return metrics
