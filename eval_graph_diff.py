"""对任意模型的已保存输出做图差异增强评分"""
import sys, os, json, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from data_loader import prepare_data
from utils import load_config, set_seed, get_device

# 用法: python eval_graph_diff.py [score_dir]
score_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/swat_normal_train_merged_test"

cfg = load_config("config.yaml"); set_seed(42); device = get_device("cuda")
_, _, test_ds, _, info = prepare_data(cfg)
C_baseline = torch.from_numpy(info["corr"]).abs().float()

# 逐批计算图差异
loader = DataLoader(test_ds, batch_size=256, shuffle=False)
all_diff = []
for batch in loader:
    x = batch["x"].to(device); b,w,n = x.shape
    x_c = x - x.mean(dim=1, keepdim=True)
    cov = torch.bmm(x_c.transpose(1,2), x_c) / (w-1)
    std = torch.sqrt(torch.var(x, dim=1, unbiased=True) + 1e-8)
    C = cov / (std.unsqueeze(1) * std.unsqueeze(2) + 1e-8)
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    all_diff.append((C.abs() - C_baseline.to(device)).abs().mean(dim=(1,2)).cpu().numpy())
graph_diff = np.concatenate(all_diff)

# IQR 归一化
gd_median = np.median(graph_diff)
gd_iqr = np.percentile(graph_diff, 75) - np.percentile(graph_diff, 25)
graph_diff_norm = np.clip((graph_diff - gd_median) / (gd_iqr + 1e-8), -10, 10)

# 加载已有分数
base_score = np.load(os.path.join(score_dir, "test_score.npy"))
test_labels = np.load(os.path.join(score_dir, "test_labels.npy"))
base_f1 = float(json.load(open(os.path.join(score_dir, "metrics.json")))["raw"]["f1"])

print(f"Model: {score_dir}")
print(f"Base F1: {base_f1:.4f}")

for gamma in [0.01, 0.05, 0.1, 0.2, 0.5]:
    enhanced = base_score + gamma * graph_diff_norm
    threshold = float(np.quantile(enhanced, 0.995))
    pred = (enhanced > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(test_labels, pred, average="binary", zero_division=0)
    auc = roc_auc_score(test_labels, enhanced)
    print(f"  gamma={gamma:.2f}: F1={f1:.4f} (Δ={f1-base_f1:+.4f})  AUC={auc:.4f}")
