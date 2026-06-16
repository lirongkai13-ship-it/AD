import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import prepare_data
from model import GATv2TCNGRUDetector
from utils import ensure_dir, get_device, load_config, save_json, set_seed


def train_one_epoch(model, loader, edge_index, optimizer, device, lambda_pred=0.5, use_amp=False, scaler=None):
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(device)
        y_future = batch["y_future"].to(device)
        y_recon = batch["y_recon"].to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.autocast(device_type="cuda"):
                pred, recon = model(x, edge_index)
                loss_pred = mse(pred, y_future)
                loss_recon = mse(recon, y_recon)
                loss = lambda_pred * loss_pred + (1.0 - lambda_pred) * loss_recon
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, recon = model(x, edge_index)
            loss_pred = mse(pred, y_future)
            loss_recon = mse(recon, y_recon)
            loss = lambda_pred * loss_pred + (1.0 - lambda_pred) * loss_recon
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, edge_index, device, lambda_pred=0.5):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="valid", leave=False):
        x = batch["x"].to(device)
        y_future = batch["y_future"].to(device)
        y_recon = batch["y_recon"].to(device)

        pred, recon = model(x, edge_index)

        loss_pred = mse(pred, y_future)
        loss_recon = mse(recon, y_recon)
        loss = lambda_pred * loss_pred + (1.0 - lambda_pred) * loss_recon

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))

    device = get_device(cfg["train"].get("device", "auto"))
    save_dir = cfg["output"]["save_dir"]
    ensure_dir(save_dir)

    train_dataset, val_dataset, test_dataset, edge_index, info = prepare_data(cfg)
    edge_index = edge_index.to(device)

    num_workers = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    scaler = torch.GradScaler("cuda") if use_amp else None

    model = GATv2TCNGRUDetector(
        num_variables=info["num_variables"],
        window_size=int(cfg["data"]["window_size"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        gru_hidden=int(cfg["model"]["gru_hidden"]),
        tcn_channels=int(cfg["model"]["tcn_channels"]),
        tcn_blocks=int(cfg["model"].get("tcn_blocks", 2)),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg["train"].get("lr_patience", 5),
        min_lr=1e-6,
    )

    early_stop_patience = cfg["train"].get("early_stop_patience", 10)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    print(f"Device: {device}  |  AMP: {use_amp}")
    print(f"Variables: {info['num_variables']}  |  Edges: {edge_index.size(1)}")
    print(f"Model: hidden={cfg['model']['hidden_dim']}, heads={cfg['model']['gat_heads']}, "
          f"tcn_blocks={cfg['model'].get('tcn_blocks',2)}, gru={cfg['model']['gru_hidden']}")
    print(f"Train samples: {len(train_dataset):,}  (stride={cfg['data'].get('train_stride','?')})")
    print(f"Val samples:   {len(val_dataset):,}  (stride={cfg['data'].get('val_stride','?')})")
    print(f"Test samples:  {len(test_dataset):,}  (stride={cfg['data'].get('test_stride','?')})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, train_loader, edge_index, optimizer, device,
            lambda_pred=float(cfg["train"]["lambda_pred"]),
            use_amp=use_amp, scaler=scaler,
        )
        val_loss = evaluate_loss(
            model, val_loader, edge_index, device,
            lambda_pred=float(cfg["train"]["lambda_pred"]),
        )

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        history.append(row)

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        lr_info = f"| lr {lr_before:.2e}"
        if lr_after < lr_before:
            lr_info += f" -> {lr_after:.2e}"

        print(f"Epoch {epoch:03d} | train loss {train_loss:.6f} | val loss {val_loss:.6f} {lr_info}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg,
                "columns": info["columns"],
                "best_val_loss": float(best_val_loss),
            }
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved best model to {ckpt_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    save_json(
        {"history": history, "best_val_loss": float(best_val_loss)},
        os.path.join(save_dir, "train_history.json"),
    )
    print("Training finished.")


if __name__ == "__main__":
    main()
