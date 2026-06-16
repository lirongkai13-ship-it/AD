#!/bin/bash
# Full-setting training: tri_branch
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== tri_branch full-setting training ==="
echo "Start: $(date)"
python train_full.py --model tri_branch --config config_full.yaml
echo "End: $(date)"
