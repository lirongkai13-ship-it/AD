"""训练: 先验图与动态Pearson自适应加权融合"""
import sys, os, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_PD_TCN_GRU

bgp_path = os.path.join(os.path.dirname(__file__), "..", "prior_fusion", "build_prior_graph.py")
spec = importlib.util.spec_from_file_location("build_prior_graph", bgp_path)
bgp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bgp)
build_prior_graph = bgp.build_prior_graph

from utils import load_config
from data_loader import read_swat_csv

cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config_dev.yaml"))
dcfg = cfg["data"]
normal_df, _ = read_swat_csv(dcfg["train_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
merged_df, _ = read_swat_csv(dcfg["test_csv"], dcfg.get("timestamp_col"), dcfg.get("label_col"))
common_cols = [c for c in normal_df.columns if c in merged_df.columns]
prior_ei, prior_w = build_prior_graph(common_cols)

if __name__ == "__main__":
    train_variant(
        GATv2_PD_TCN_GRU, "prior_dynamic",
        model_kwargs={"prior_edge_index": prior_ei, "prior_weights": prior_w},
        save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                              "outputs", "swat_normal_train_merged_test", "prior_dynamic"))
