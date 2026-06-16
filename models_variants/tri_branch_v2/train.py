"""训练 tri_branch_v2 (upgraded Branch 3)"""
import sys, os, time, json
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data_loader import prepare_data
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from evaluate import binary_metrics
from models_variants.tri_branch_v2.variant_model import TriBranch_USAD_v2
from tqdm import tqdm


def main():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "cuda"))
    save_dir = os.path.join(cfg["output"]["save_dir"], "tri_branch_v2")
    ensure_dir(save_dir)

    train_ds, val_ds, test_ds, _, info = prepare_data(cfg)
    bs = int(cfg["train"]["batch_size"]); nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  bs, shuffle=False, num_workers=nw, pin_memory=True)

    # Build graphs
    from data_loader import build_pearson_edge_index, split_train_val, read_swat_csv
    from sklearn.preprocessing import StandardScaler
    import pandas as pd, importlib.util
    dcfg = cfg["data"]
    nfd = pd.read_csv(dcfg["train_csv"]); nfd.columns = [str(c).strip() for c in nfd.columns]
    if dcfg.get("timestamp_col") in nfd.columns: nfd = nfd.drop(columns=[dcfg["timestamp_col"]])
    if dcfg.get("label_col") in nfd.columns: nfd = nfd.drop(columns=[dcfg["label_col"]])
    mfd = pd.read_csv(dcfg["test_csv"]); mfd.columns = [str(c).strip() for c in mfd.columns]
    common_cols = [c for c in nfd.columns if c in mfd.columns]
    normal_raw = nfd[common_cols].values.astype(np.float32)
    train_raw, _, _, _ = split_train_val(normal_raw, None, 0.2)
    scaler_fit = StandardScaler(); train_vals = scaler_fit.fit_transform(train_raw)
    static_ei, _ = build_pearson_edge_index(train_vals)

    bgp_path = os.path.join(os.path.dirname(__file__), "..", "prior_fusion", "build_prior_graph.py")
    spec = importlib.util.spec_from_file_location("bpg", bgp_path)
    bpg_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(bpg_mod)
    prior_ei, prior_w = bpg_mod.build_prior_graph(common_cols)
    print(f"Prior edges: {prior_ei.shape[1]}")

    model = TriBranch_USAD_v2(
        info["num_variables"], int(cfg["data"]["window_size"]), static_ei,
        prior_edge_index=prior_ei, prior_weights=prior_w,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        tcn_channels=int(cfg["model"]["tcn_channels"]),
        tcn_blocks=int(cfg["model"].get("tcn_blocks", 1)),
        dropout=float(cfg["model"]["dropout"]),
        encoder_mode="tri_branch_residual_gate",
        use_temporal_attn_pooling=True,
        use_node_conditioned_fusion=True,
        gate_type="scalar",
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
                    r1, r2, r12, _ = model(x, static_ei)
                    loss_r1, loss_r2, loss_r12 = mse(r1, x), mse(r2, x), mse(r12, x)
                    loss = loss_r1 + loss_r2 + 0.5*loss_r12 + (0.1 if phase==1 else 0.05)*mse(r1, r2.detach())
                    if phase == 2: loss = loss + 0.05*torch.abs(loss_r1-loss_r2)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                r1, r2, r12, _ = model(x, static_ei)
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

    # Evaluate
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
    threshold = float(np.quantile(aggregate_topk_score(val_norm, 1), 0.995))
    test_norm = apply_iqr_normalize(test_errors, iqr_params)
    test_score = aggregate_topk_score(test_norm, topk=1)
    test_pred = (test_score > threshold).astype(int)

    raw_m = binary_metrics(test_labels, test_pred, test_score)
    pa_pred = point_adjust(test_pred, test_labels)
    pa_m = binary_metrics(test_labels, pa_pred, test_score)

    metrics = {"threshold": threshold, "raw": raw_m, "point_adjust": pa_m,
               "train_time_min": train_time/60}
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    print(f"\n=== Tri-Branch v2 (upgraded Branch 3) ===")
    print(f"Raw F1: {raw_m['f1']:.4f}  AUC={raw_m.get('roc_auc','?')}")
    print(f"PA  F1: {pa_m['f1']:.4f}")


if __name__ == "__main__":
    main()
