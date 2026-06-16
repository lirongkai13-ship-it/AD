"""训练: Baseline + 动态 Pearson 图（保存 test_score.npy）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_DG_TCN_GRU
import torch, numpy as np
from torch.utils.data import DataLoader
from utils import load_config, set_seed, get_device, fit_iqr_params, apply_iqr_normalize, aggregate_topk_score
from data_loader import prepare_data, build_pearson_edge_index, split_train_val
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 构建静态边
cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
dcfg = cfg["data"]
normal_df = pd.read_csv(dcfg["train_csv"])
normal_df.columns = [str(c).strip() for c in normal_df.columns]
if dcfg.get("timestamp_col") in normal_df.columns: normal_df = normal_df.drop(columns=[dcfg["timestamp_col"]])
if dcfg.get("label_col") in normal_df.columns: normal_df = normal_df.drop(columns=[dcfg["label_col"]])
merged_df = pd.read_csv(dcfg["test_csv"])
merged_df.columns = [str(c).strip() for c in merged_df.columns]
common_cols = [c for c in normal_df.columns if c in merged_df.columns]
normal_raw = normal_df[common_cols].values.astype(np.float32)
train_raw, _, _, _ = split_train_val(normal_raw, None, 0.2)
scaler = StandardScaler(); train_vals = scaler.fit_transform(train_raw)
static_ei, _ = build_pearson_edge_index(train_vals)

save_dir = os.path.join(cfg["output"]["save_dir"], "dynamic_graph_diff")

if __name__ == "__main__":
    metrics = train_variant(
        GATv2_DG_TCN_GRU, "dynamic_graph_diff",
        model_kwargs={"static_edge_index": static_ei},
        save_dir=save_dir)

    # ── 额外保存 test_score.npy ──
    print("Saving test_score.npy and test_labels.npy...")
    set_seed(42); device = get_device("cuda")
    train_ds, val_ds, test_ds, _, _ = prepare_data(cfg)
    bs = 256
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = GATv2_DG_TCN_GRU(
        len(common_cols), int(cfg["data"]["window_size"]), static_ei,
        int(cfg["model"]["hidden_dim"]), int(cfg["model"]["gat_heads"]),
        int(cfg["model"]["gru_hidden"]), int(cfg["model"]["tcn_channels"]),
        int(cfg["model"].get("tcn_blocks", 1)), float(cfg["model"]["dropout"]),
    ).to(device)
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def collect(loader):
        errs, labs = [], []
        for batch in loader:
            x = batch["x"].to(device)
            with torch.no_grad():
                recon = model(x, static_ei.to(device))
                if isinstance(recon, tuple): recon = recon[0]
            errs.append((recon - x).abs().mean(dim=1).cpu().numpy())
            if "label" in batch: labs.append(batch["label"].cpu().numpy())
        return np.concatenate(errs), np.concatenate(labs) if labs else None

    val_err, _ = collect(val_loader)
    test_err, test_labels = collect(test_loader)
    iqr_p = fit_iqr_params(val_err)
    test_norm = apply_iqr_normalize(test_err, iqr_p)
    test_score = aggregate_topk_score(test_norm, topk=5)

    np.save(os.path.join(save_dir, "test_score.npy"), test_score)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_labels)
    print("Saved test_score.npy and test_labels.npy")
