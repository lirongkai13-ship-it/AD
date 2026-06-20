#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== Remaining external models (full stride=1) ==="
echo "Start: $(date)"

CONFIG="models/base_config_full.yaml"
RUNNER="models/run_single.py"
OUTDIR="results/models_full"
mkdir -p "$OUTDIR"

MODELS=("AnoTrans" "CAN" "GDN" "TimesNet")

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> [$model] starting at $(date) <<<"
    python -u "$RUNNER" "$model" "$CONFIG"
    echo ">>> [$model] done at $(date) <<<"
done

echo ""
echo "=== ALL DONE: $(date) ==="
