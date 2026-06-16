#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
rm -f outputs/swat_normal_train_merged_test/abc_b_v2/metrics.json
rm -f outputs/swat_normal_train_merged_test/abc_c_v2/metrics.json
echo "=== B: $(date) ==="
python -u models_variants/abc/train_b.py > outputs/swat_normal_train_merged_test/train_abc_b_v6.log 2>&1
echo "B exit: $?"
echo "=== C: $(date) ==="
python -u models_variants/abc/train_c.py > outputs/swat_normal_train_merged_test/train_abc_c_v6.log 2>&1
echo "C exit: $?"
echo "ALL DONE: $(date)"
