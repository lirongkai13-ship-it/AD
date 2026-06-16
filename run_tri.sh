#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
rm -f outputs/swat_normal_train_merged_test/tri_branch/best_model.pt
python -u models_variants/tri_branch/train.py > outputs/swat_normal_train_merged_test/train_tri_branch.log 2>&1
echo "tri_branch done: $?"
