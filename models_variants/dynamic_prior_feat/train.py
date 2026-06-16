"""训练: 动态Pearson + 特征级先验融合"""
import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_DynPrior_TCN_GRU

# 直接加载 build_prior_graph（避免 sys.path 污染）
bgp_path = os.path.join(os.path.dirname(__file__), "..", "prior_fusion", "build_prior_graph.py")
spec = importlib.util.spec_from_file_location("build_prior_graph", bgp_path)
bgp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bgp)
build_prior_graph = bgp.build_prior_graph

from utils import load_config
from data_loader import read_swat_csv, build_pearson_edge_index, split_train_val
from sklearn.preprocessing import StandardScaler
import numpy as np

cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
dcfg = cfg["data"]
normal_df, _ = read_swat_csv(dcfg["train_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
merged_df, _ = read_swat_csv(dcfg["test_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
common_cols = [c for c in normal_df.columns if c in merged_df.columns]

# 构建先验图
prior_ei, prior_w = build_prior_graph(common_cols)

# 构建静态 Pearson 边（和 baseline 一致）
normal_raw = normal_df[common_cols].values.astype(np.float32)
train_raw, _, _, _ = split_train_val(normal_raw, None, float(dcfg.get("val_ratio", 0.2)))
scaler = StandardScaler(); train_vals = scaler.fit_transform(train_raw)
static_ei, _ = build_pearson_edge_index(train_vals, float(dcfg.get("corr_threshold", 0.3)))

print(f"Prior edges: {prior_ei.shape[1]}  |  Static edges: {static_ei.shape[1]}")

if __name__ == "__main__":
    train_variant(
        GATv2_DynPrior_TCN_GRU, "dynamic_prior_feat",
        model_kwargs={"prior_edge_index": prior_ei, "prior_weights": prior_w,
                       "static_edge_index": static_ei},
        save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                              "outputs", "swat_normal_train_merged_test", "dynamic_prior_feat"))
