"""训练: Baseline + 自适应先验知识图融合"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_PriorFusion_TCN_GRU
from build_prior_graph import build_prior_graph
from utils import load_config
from data_loader import read_swat_csv

# ── 构建先验图 ──
cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
dcfg = cfg["data"]
normal_df, _ = read_swat_csv(dcfg["train_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
merged_df, _ = read_swat_csv(dcfg["test_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
common_cols = [c for c in normal_df.columns if c in merged_df.columns]

print("Building prior knowledge graph...")
prior_ei, prior_w = build_prior_graph(common_cols)
print(f"Prior edges: {prior_ei.shape[1]}")

if __name__ == "__main__":
    train_variant(
        GATv2_PriorFusion_TCN_GRU, "prior_fusion",
        model_kwargs={"prior_edge_index": prior_ei, "prior_weights": prior_w},
        save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                              "outputs", "swat_normal_train_merged_test", "prior_fusion"))
