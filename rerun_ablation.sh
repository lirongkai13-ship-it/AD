#!/bin/bash
# 消融实验批量重跑 (非USAD变体)
# 使用 config_dev.yaml (stride=10, epoch=5)
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
LOG_DIR="outputs/swat_normal_train_merged_test"

echo "===== $(date) 开始消融实验重跑 ====="

models=("temporal_attn" "prior_fusion" "prior_dynamic" "ms_tcn" "dynamic_graph" "dynamic_graph_diff" "dynamic_prior_feat")

for m in "${models[@]}"; do
    echo "===== 开始训练: $m ====="
    SECONDS=0
    python models_variants/$m/train.py > $LOG_DIR/train_${m}.log 2>&1
    rc=$?
    echo "===== $m 完成 (exit=$rc, elapsed=${SECONDS}s) ====="

    # Sleep briefly to let GPU cool
    sleep 2
done

echo "===== $(date) 全部完成 ====="
