import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data_loader import prepare_data
from model import GATv2TCNGRUDetector
from utils import (
    aggregate_topk_score,
    apply_iqr_normalize,
    ensure_dir,
    fit_iqr_params,
    get_device,
    load_config,
    point_adjust,
    save_json,
)


@torch.no_grad()
def collect_var_errors(model, loader, edge_index, device, lambda_pred=0.5):
    """
    返回：
      var_errors: [M, N]
      labels: [M] or None

    var_errors = lambda * prediction_abs_error + (1-lambda) * reconstruction_abs_error
    """
    model.eval()
    all_var_errors = []
    all_labels = []

    for batch in loader:
        x = batch["x"].to(device)
        y_future = batch["y_future"].to(device)
        y_recon = batch["y_recon"].to(device)

        pred, recon = model(x, edge_index)

        pred_err = (pred - y_future).abs()  # [B, N]
        recon_err = (recon - y_recon).abs().mean(dim=1)  # [B, N]
        var_err = lambda_pred * pred_err + (1.0 - lambda_pred) * recon_err

        all_var_errors.append(var_err.detach().cpu().numpy())

        if "label" in batch:
            all_labels.append(batch["label"].cpu().numpy())

    var_errors = np.concatenate(all_var_errors, axis=0)
    labels = np.concatenate(all_labels, axis=0) if len(all_labels) > 0 else None
    return var_errors, labels


def binary_metrics(labels, pred, score):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred,
        average="binary",
        zero_division=0,
    )

    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    try:
        result["roc_auc"] = float(roc_auc_score(labels, score))
    except Exception:
        result["roc_auc"] = None

    try:
        result["pr_auc"] = float(average_precision_score(labels, score))
    except Exception:
        result["pr_auc"] = None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["train"].get("device", "auto"))

    save_dir = cfg["output"]["save_dir"]
    ensure_dir(save_dir)

    if args.ckpt is None:
        args.ckpt = os.path.join(save_dir, "best_model.pt")

    train_dataset, val_dataset, test_dataset, edge_index, info = prepare_data(cfg)
    edge_index = edge_index.to(device)

    num_workers = cfg["train"].get("num_workers", 0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

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

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    lambda_pred = float(cfg["train"]["lambda_pred"])
    topk = int(cfg["score"]["topk"])
    threshold_quantile = float(cfg["score"]["threshold_quantile"])

    # 1) validation normal 上拟合误差归一化参数与阈值
    val_var_errors, val_labels = collect_var_errors(
        model,
        val_loader,
        edge_index,
        device,
        lambda_pred=lambda_pred,
    )
    iqr_params = fit_iqr_params(val_var_errors)
    val_norm_errors = apply_iqr_normalize(val_var_errors, iqr_params)
    val_score = aggregate_topk_score(val_norm_errors, topk=topk)
    threshold = float(np.quantile(val_score, threshold_quantile))

    # 2) Attack test 只用于最终评估
    test_var_errors, test_labels = collect_var_errors(
        model,
        test_loader,
        edge_index,
        device,
        lambda_pred=lambda_pred,
    )
    test_norm_errors = apply_iqr_normalize(test_var_errors, iqr_params)
    test_score = aggregate_topk_score(test_norm_errors, topk=topk)
    test_pred = (test_score > threshold).astype(int)

    np.save(os.path.join(save_dir, "val_score.npy"), val_score)
    np.save(os.path.join(save_dir, "val_var_errors.npy"), val_var_errors)
    np.save(os.path.join(save_dir, "test_score.npy"), test_score)
    np.save(os.path.join(save_dir, "test_var_errors.npy"), test_var_errors)
    np.save(os.path.join(save_dir, "test_pred.npy"), test_pred)

    print(f"Checkpoint: {args.ckpt}")
    print(f"Threshold from validation normal score q={threshold_quantile}: {threshold:.6f}")
    print(f"Scores saved to {save_dir}")

    metrics = {
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "topk": min(topk, test_var_errors.shape[1]),
        "lambda_pred": lambda_pred,
    }

    if test_labels is not None:
        np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)

        raw_metrics = binary_metrics(test_labels, test_pred, test_score)
        print("Raw metrics:")
        print(f"Precision: {raw_metrics['precision']:.4f}")
        print(f"Recall:    {raw_metrics['recall']:.4f}")
        print(f"F1:        {raw_metrics['f1']:.4f}")
        print(f"ROC-AUC:   {raw_metrics['roc_auc']}")
        print(f"PR-AUC:    {raw_metrics['pr_auc']}")
        metrics["raw"] = raw_metrics

        pred_pa = point_adjust(test_pred, test_labels)
        np.save(os.path.join(save_dir, "test_pred_point_adjust.npy"), pred_pa)

        pa_metrics = binary_metrics(test_labels, pred_pa, test_score)
        print("Point-adjust metrics:")
        print(f"Precision: {pa_metrics['precision']:.4f}")
        print(f"Recall:    {pa_metrics['recall']:.4f}")
        print(f"F1:        {pa_metrics['f1']:.4f}")
        metrics["point_adjust"] = pa_metrics

        # 变量贡献 Top-k，仅作为解释辅助，不等同于严格根因定位
        mean_var_error = test_var_errors.mean(axis=0)
        top_idx = np.argsort(mean_var_error)[::-1][: min(topk, len(mean_var_error))]
        top_variables = []

        print("Top variable contribution:")
        for idx in top_idx:
            item = {
                "index": int(idx),
                "name": info["columns"][idx],
                "mean_error": float(mean_var_error[idx]),
            }
            top_variables.append(item)
            print(f"{item['name']}: {item['mean_error']:.6f}")

        metrics["top_variables"] = top_variables

    save_json(metrics, os.path.join(save_dir, "metrics.json"))


if __name__ == "__main__":
    main()