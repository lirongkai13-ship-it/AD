"""评估动态Pearson模型 + 图差异增强评分"""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from data_loader import prepare_data, build_pearson_edge_index
from utils import load_config, set_seed, get_device
from variant_model import GATv2_DG_TCN_GRU

# 用 dev config 改 stride=1 匹配 checkpoint 的图大小
cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
cfg["data"]["stride"] = 1  # 覆盖为 stride=1 使 Pearson 图为 399 边
cfg["data"]["train_stride"] = 1
cfg["data"]["val_stride"] = 1
cfg["data"]["test_stride"] = 1
set_seed(42); device = get_device("cuda")

train_ds, val_ds, test_ds, edge_index, info = prepare_data(cfg)
edge_index = edge_index.to(device)
bs = 256
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

# 构建模型（与训练时一致）
from sklearn.preprocessing import StandardScaler
import pandas as pd
dcfg = cfg["data"]
normal_df = pd.read_csv(dcfg["train_csv"])
normal_df.columns = [str(c).strip() for c in normal_df.columns]
if dcfg.get("timestamp_col") in normal_df.columns: normal_df = normal_df.drop(columns=[dcfg["timestamp_col"]])
if dcfg.get("label_col") in normal_df.columns:
    label_col = dcfg["label_col"]
    normal_df = normal_df.drop(columns=[label_col])
from data_loader import split_train_val
normal_raw = normal_df[info["columns"]].values.astype(np.float32)
train_raw, _, _, _ = split_train_val(normal_raw, None, 0.2)
scaler_fit = StandardScaler(); train_vals = scaler_fit.fit_transform(train_raw)
static_ei, _ = build_pearson_edge_index(train_vals)

model = GATv2_DG_TCN_GRU(
    info["num_variables"], int(cfg["data"]["window_size"]), static_ei,
    int(cfg["model"]["hidden_dim"]), int(cfg["model"]["gat_heads"]),
    int(cfg["model"]["gru_hidden"]), int(cfg["model"]["tcn_channels"]),
    int(cfg["model"].get("tcn_blocks", 1)), float(cfg["model"]["dropout"]),
).to(device)

ckpt_path = os.path.join(cfg["output"]["save_dir"], "dynamic_graph", "best_model.pt")
model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# ── 评估：收集误差 ──
def collect_errors(loader):
    errors, labels = [], []
    for batch in loader:
        x = batch["x"].to(device)
        with torch.no_grad():
            recon = model(x, static_ei.to(device))
            if isinstance(recon, tuple): recon = recon[0]
        err = (recon - x).abs().mean(dim=1).cpu().numpy()
        errors.append(err)
        if "label" in batch: labels.append(batch["label"].cpu().numpy())
    return np.concatenate(errors), np.concatenate(labels) if labels else None

print("Evaluating...")
val_err, _ = collect_errors(val_loader)
test_err, test_labels = collect_errors(test_loader)

# IQR 评分
from utils import fit_iqr_params, apply_iqr_normalize, aggregate_topk_score
iqr_p = fit_iqr_params(val_err)
val_norm = apply_iqr_normalize(val_err, iqr_p)
val_score = aggregate_topk_score(val_norm, 5)
threshold = float(np.quantile(val_score, 0.995))

test_norm = apply_iqr_normalize(test_err, iqr_p)
base_score = aggregate_topk_score(test_norm, 5)
base_pred = (base_score > threshold).astype(int)

# 基础指标
p, r, f1, _ = precision_recall_fscore_support(test_labels, base_pred, average="binary", zero_division=0)
auc = roc_auc_score(test_labels, base_score)
print(f"Base F1: {f1:.4f}  P={p:.4f}  R={r:.4f}  AUC={auc:.4f}")

# ── 图结构差异 ──
C_baseline = torch.from_numpy(info["corr"]).abs().float()
all_diff = []
for batch in test_loader:
    x = batch["x"].to(device); b,w,n = x.shape
    x_c = x - x.mean(dim=1, keepdim=True)
    cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
    std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
    C = cov / (std.unsqueeze(1)*std.unsqueeze(2) + 1e-8)
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    all_diff.append((C.abs() - C_baseline.to(device)).abs().mean(dim=(1,2)).cpu().numpy())
graph_diff = np.concatenate(all_diff)

# 归一化
gd_median = np.median(graph_diff)
gd_iqr = np.percentile(graph_diff, 75) - np.percentile(graph_diff, 25)
gd_norm = np.clip((graph_diff - gd_median) / (gd_iqr + 1e-8), -10, 10)

print(f"\n--- Enhanced Scoring ---")
for gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    enhanced = base_score + gamma * gd_norm
    threshold_e = float(np.quantile(enhanced, 0.995))
    pred_e = (enhanced > threshold_e).astype(int)
    p_e, r_e, f1_e, _ = precision_recall_fscore_support(test_labels, pred_e, average="binary", zero_division=0)
    print(f"  gamma={gamma:.2f}: F1={f1_e:.4f} (Δ={f1_e-f1:+.4f})")
