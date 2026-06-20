#!/bin/bash
cd D:/1LRK/swat_gatv2_tcn_gru_baseline
echo "=== External models full-parameter training ==="
echo "Start: $(date)"

CONFIG="models/base_config_full.yaml"
RUNNER="models/run_single.py"
OUTDIR="results/models_full"
mkdir -p "$OUTDIR"

MODELS=("USAD" "DAGMM" "LSTM-AE" "MAD-GAN" "DCdetector" "TranAD" "MTAD-GAT" "CAN" "AnoTrans" "GDN" "TimesNet")

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> [$model] starting at $(date) <<<"
    python -u "$RUNNER" "$model" "$CONFIG" 2>&1 | tee "$OUTDIR/${model}_train.log"
    echo ">>> [$model] done at $(date) <<<"
done

echo ""
echo "=== ALL DONE: $(date) ==="
