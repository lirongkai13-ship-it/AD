#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
rm -f outputs/swat_normal_train_merged_test/tri_branch_gate_0_5/metrics.json
python -u models_variants/tri_branch/train_gate_05.py > outputs/swat_normal_train_merged_test/train_gate_05.log 2>&1
echo "exit: $?"
