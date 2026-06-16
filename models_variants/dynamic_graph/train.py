"""训练: Baseline + 动态 Pearson 图"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_DG_TCN_GRU
from utils import load_config
from data_loader import build_pearson_edge_index

# 预处理：获取静态 edge_index
cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
# 加载数据
from data_loader import read_swat_csv, build_labels, split_train_val
from sklearn.preprocessing import StandardScaler
import numpy as np

dcfg = cfg["data"]
normal_df, _ = read_swat_csv(dcfg["train_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
merged_df, _ = read_swat_csv(dcfg["test_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
common_cols = [c for c in normal_df.columns if c in merged_df.columns]
normal_df = normal_df[common_cols]
normal_raw = normal_df.values.astype(np.float32)
train_raw, _, _, _ = split_train_val(normal_raw, None, float(dcfg.get("val_ratio", 0.2)))
scaler = StandardScaler()
train_vals = scaler.fit_transform(train_raw)

static_edge_index, _ = build_pearson_edge_index(train_vals, float(dcfg.get("corr_threshold", 0.3)))

if __name__ == "__main__":
    train_variant(GATv2_DG_TCN_GRU, "dynamic_graph",
                  model_kwargs={"static_edge_index": static_edge_index},
                  save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                                        "outputs", "swat_normal_train_merged_test", "dynamic_graph"))
