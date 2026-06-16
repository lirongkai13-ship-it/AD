#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
rm -f outputs/swat_normal_train_merged_test/tri_branch_gamma_0_02/metrics.json
python -u models_variants/tri_branch/train_gamma_0_02.py > outputs/swat_normal_train_merged_test/train_gamma_0_02_v2.log 2>&1
echo "exit: $?"
