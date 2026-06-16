#!/bin/bash
rm -f outputs/swat_normal_train_merged_test/parallel_usad_d/best_model.pt outputs/swat_normal_train_merged_test/parallel_usad_d/metrics.json
rm -f outputs/swat_normal_train_merged_test/parallel_usad_e/best_model.pt outputs/swat_normal_train_merged_test/parallel_usad_e/metrics.json
echo "===== D: $(date) ====="
python -u models_variants/parallel_usad_d/train.py > outputs/swat_normal_train_merged_test/train_parallel_usad_d_v4.log 2>&1
echo "D exit: $?"
echo "===== E: $(date) ====="
python -u models_variants/parallel_usad_e/train.py > outputs/swat_normal_train_merged_test/train_parallel_usad_e_v4.log 2>&1
echo "E exit: $?"
echo "===== ALL DONE: $(date) ====="
