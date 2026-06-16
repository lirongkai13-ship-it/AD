#!/bin/bash
# Full-setting training: prior
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== prior full-setting training ==="
echo "Start: $(date)"
python train_full.py --model prior --config config_full.yaml
echo "End: $(date)"
