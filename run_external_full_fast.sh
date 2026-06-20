#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== External models full-parameter (fast only, skip MTAD-GAT/GCN) ==="
echo "Start: $(date)"

CONFIG="models/base_config_full.yaml"
RUNNER="models/run_single.py"
OUTDIR="results/models_full"
mkdir -p "$OUTDIR"

# Skip MTAD-GAT (2s/it) and GCN (for-loop GAT)
MODELS=("CAN" "AnoTrans" "GDN" "TimesNet")

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> [$model] starting at $(date) <<<"
    python -u "$RUNNER" "$model" "$CONFIG" 2>&1 | tee "$OUTDIR/${model}_train.log"
    echo ">>> [$model] done at $(date) <<<"
done

echo ""
echo "=== ALL DONE: $(date) ==="
