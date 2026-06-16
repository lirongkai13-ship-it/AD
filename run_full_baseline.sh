#!/bin/bash
# Full-setting training: baseline
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== baseline full-setting training ==="
echo "Start: $(date)"
python train_full.py --model baseline --config config_full.yaml
echo "End: $(date)"
