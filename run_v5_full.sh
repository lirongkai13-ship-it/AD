#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== v5 full-setting training ==="
echo "Start: $(date)"
python -u models_variants/tri_branch_v5/train_full.py
echo "End: $(date)"
