"""训练: Baseline + 时间注意力"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2_TA_TCN_GRU

if __name__ == "__main__":
    train_variant(GATv2_TA_TCN_GRU, "temporal_attn",
                  save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                                        "outputs", "swat_normal_train_merged_test", "temporal_attn"))
