"""训练: Baseline + 多尺度 TCN"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from base_variant_trainer import train_variant
from variant_model import GATv2TCNGRUDetector_MS

if __name__ == "__main__":
    train_variant(GATv2TCNGRUDetector_MS, "ms_tcn",
                  save_dir=os.path.join(os.path.dirname(__file__), "..", "..",
                                        "outputs", "swat_normal_train_merged_test", "ms_tcn"))
