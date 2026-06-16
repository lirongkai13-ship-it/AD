#!/bin/bash
set -e
cd D:/1LRK/swat_gatv2_tcn_gru_baseline

echo "============================================"
echo "FULL-SETTING TRAINING: all 3 models"
echo "Start: $(date)"
echo "============================================"

echo ""
echo ">>> [1/3] tri_branch <<<"
python train_full.py --model tri_branch --config config_full.yaml
echo ">>> tri_branch DONE: $(date)"

echo ""
echo ">>> [2/3] baseline <<<"
python train_full.py --model baseline --config config_full.yaml
echo ">>> baseline DONE: $(date)"

echo ""
echo ">>> [3/3] prior <<<"
python train_full.py --model prior --config config_full.yaml
echo ">>> prior DONE: $(date)"

echo ""
echo "============================================"
echo "ALL DONE: $(date)"
echo "============================================"
