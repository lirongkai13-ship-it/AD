#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== gamma=0.0: $(date) ==="
python -u models_variants/tri_branch/train_gamma_0_0.py > outputs/swat_normal_train_merged_test/train_gamma_0_0.log 2>&1
echo "gamma=0.0 exit: $?"
echo "=== gamma=0.02: $(date) ==="
python -u models_variants/tri_branch/train_gamma_0_02.py > outputs/swat_normal_train_merged_test/train_gamma_0_02.log 2>&1
echo "gamma=0.02 exit: $?"
echo "=== ALL DONE: $(date) ==="
