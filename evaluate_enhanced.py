"""增强异常分数评估：加入图结构差异项"""
import argparse, os, json, numpy as np, torch
from torch.utils.data import DataLoader
from data_loader import prepare_data
from model import GATv2TCNGRUDetector
from utils import (load_config, set_seed, get_device, ensure_dir,
                   fit_iqr_params, apply_iqr_normalize, aggregate_topk_score,
                   point_adjust, save_json)
from evaluate import collect_var_errors, binary_metrics


def compute_graph_diff(model, loader, edge_index, device, C_baseline):
    """计算每个测试样本的图结构差异"""
    model.eval()
    all_graph_diff = []
    for batch in loader:
        x = batch["x"].to(device)
        b, w, n = x.shape
        # 计算当前样本的 Pearson 相关
        x_c = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_c.transpose(1, 2), x_c) / (w - 1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
        C_current = cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8)
        C_current = torch.nan_to_num(C_current, nan=0.0, posinf=0.0, neginf=0.0)

        # 图差异: ||C_current - C_baseline||  每样本 [B, N, N] → mean over (N,N) → [B]
        diff = (C_current - C_baseline.to(device)).abs().mean(dim=(1, 2))  # [B]
        all_graph_diff.append(diff.cpu().numpy())

    return np.concatenate(all_graph_diff, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "dynamic_graph"],
                        help="模型类型")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="图差异在异常分数中的权重")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"]["seed"]))
    device = get_device(cfg["train"].get("device", "cuda"))
    save_dir = cfg["output"]["save_dir"]
    ensure_dir(save_dir)

    if args.ckpt is None:
        args.ckpt = os.path.join(save_dir, "best_model.pt")

    train_ds, val_ds, test_ds, edge_index, info = prepare_data(cfg)
    edge_index = edge_index.to(device)
    bs = int(cfg["train"]["batch_size"])

    val_loader   = DataLoader(val_ds,  batch_size=bs, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # 构建基线 Pearson 相关矩阵（从训练集）
    train_values = info["scaler"].inverse_transform(
        np.array([train_ds[i]["x"].numpy() for i in range(0, min(10000, len(train_ds)), 100)]).reshape(-1, info["num_variables"]))
    # 简化：用 dynamic_graph 中已有的 corr 矩阵
    C_baseline = torch.from_numpy(info["corr"]).float().abs()  # [N, N] 基线相关矩阵

    if args.model_type == "dynamic_graph":
        from models_variants.dynamic_graph.variant_model import GATv2_DG_TCN_GRU
        from data_loader import build_pearson_edge_index, split_train_val
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        dcfg = cfg["data"]
        normal_df = pd.read_csv(dcfg["train_csv"])
        normal_df.columns = [str(c).strip() for c in normal_df.columns]
        if dcfg.get("timestamp_col") and dcfg["timestamp_col"] in normal_df.columns:
            normal_df = normal_df.drop(columns=[dcfg["timestamp_col"]])
        if dcfg.get("label_col") and dcfg["label_col"] in normal_df.columns:
            normal_df = normal_df.drop(columns=[dcfg["label_col"]])
        normal_raw = normal_df[info["columns"]].values.astype(np.float32)
        train_raw, _, _, _ = split_train_val(normal_raw, None, float(dcfg.get("val_ratio", 0.2)))
        scaler = StandardScaler(); train_vals = scaler.fit_transform(train_raw)
        static_ei, _ = build_pearson_edge_index(train_vals)
        model = GATv2_DG_TCN_GRU(
            info["num_variables"], int(cfg["data"]["window_size"]), static_ei,
            int(cfg["model"]["hidden_dim"]), int(cfg["model"]["gat_heads"]),
            int(cfg["model"]["gru_hidden"]), int(cfg["model"]["tcn_channels"]),
            int(cfg["model"].get("tcn_blocks", 1)), float(cfg["model"]["dropout"]),
        ).to(device)
    else:
        model = GATv2TCNGRUDetector(
            num_variables=info["num_variables"],
            window_size=int(cfg["data"]["window_size"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            gat_heads=int(cfg["model"]["gat_heads"]),
            gru_hidden=int(cfg["model"]["gru_hidden"]),
            tcn_channels=int(cfg["model"]["tcn_channels"]),
            tcn_blocks=int(cfg["model"].get("tcn_blocks", 1)),
            dropout=float(cfg["model"]["dropout"]),
        ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    lambda_pred = float(cfg["train"]["lambda_pred"])
    gamma = args.gamma

    # 1) validation normal 上拟合 IQR
    val_errors, _ = collect_var_errors(model, val_loader, edge_index, device, lambda_pred=lambda_pred)
    iqr_params = fit_iqr_params(val_errors)
    val_norm = apply_iqr_normalize(val_errors, iqr_params)
    val_score = aggregate_topk_score(val_norm, topk=5)
    threshold = float(np.quantile(val_score, 0.995))

    # 2) 测试集评估
    test_errors, test_labels = collect_var_errors(model, test_loader, edge_index, device, lambda_pred=lambda_pred)
    test_norm = apply_iqr_normalize(test_errors, iqr_params)
    base_score = aggregate_topk_score(test_norm, topk=5)

    # ── 新增：图结构差异 ──
    print("Computing graph structure difference...")
    graph_diff = compute_graph_diff(model, test_loader, edge_index, device, C_baseline)
    # 归一化图差异
    gd_median = np.median(graph_diff)
    gd_iqr = np.percentile(graph_diff, 75) - np.percentile(graph_diff, 25)
    graph_diff_norm = (graph_diff - gd_median) / (gd_iqr + 1e-8)

    # ── 融合异常分数 ──
    enhanced_score = base_score + gamma * graph_diff_norm
    test_pred = (enhanced_score > threshold).astype(int)

    # 基线分数（不加图差异）
    base_pred = (base_score > threshold).astype(int)

    # 结果
    base_m = binary_metrics(test_labels, base_pred, base_score)
    enhanced_m = binary_metrics(test_labels, test_pred, enhanced_score)

    print(f"\n=== Baseline Score ===")
    print(f"F1={base_m['f1']:.4f}  P={base_m['precision']:.4f}  R={base_m['recall']:.4f}  AUC={base_m['roc_auc']}")

    print(f"\n=== Enhanced Score (γ={gamma}) ===")
    print(f"F1={enhanced_m['f1']:.4f}  P={enhanced_m['precision']:.4f}  R={enhanced_m['recall']:.4f}  AUC={enhanced_m['roc_auc']}")

    delta = enhanced_m['f1'] - base_m['f1']
    print(f"\nΔ F1 = {delta:+.4f}")

    # 保存
    np.save(os.path.join(save_dir, "enhanced_score.npy"), enhanced_score)
    np.save(os.path.join(save_dir, "graph_diff.npy"), graph_diff)
    result = {"gamma": gamma, "base_f1": base_m["f1"], "enhanced_f1": enhanced_m["f1"],
              "delta_f1": delta, "base": base_m, "enhanced": enhanced_m}
    save_json(result, os.path.join(save_dir, "enhanced_metrics.json"))
    print(f"Saved to {save_dir}/enhanced_metrics.json")


if __name__ == "__main__":
    main()
