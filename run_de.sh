#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
LOG="outputs/swat_normal_train_merged_test"

echo "===== $(date) 开始 D/E 串行重跑 ====="

echo "===== D: MultiScale TCN ====="
rm -f $LOG/parallel_usad_d/best_model.pt $LOG/parallel_usad_d/metrics.json
python -u models_variants/parallel_usad_d/train.py > $LOG/train_parallel_usad_d_v2.log 2>&1
echo "D exit: $?"

echo "===== E: MultiScale TCN+GRU ====="
rm -f $LOG/parallel_usad_e/best_model.pt $LOG/parallel_usad_e/metrics.json
python -u models_variants/parallel_usad_e/train.py > $LOG/train_parallel_usad_e_v2.log 2>&1
echo "E exit: $?"

echo "===== $(date) 全部完成 ====="
