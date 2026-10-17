#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[1/2] Fold 03 B' calibrated GB loss: starting from scratch"
python train_diffusion.py \
  --config configs/4080super_32gb_fold03_structure_gb.yaml \
  --exclude-alloy-id 3

test -f "08_训练与生成结果/03_结构loss消融/Fold03_B2_gb_20000/final/run_metadata.json"
echo "[1/2] Fold 03 B' completed; starting C'"

echo "[2/2] Fold 03 C' calibrated GB + edge loss: starting from scratch"
python train_diffusion.py \
  --config configs/4080super_32gb_fold03_structure_ablation.yaml \
  --exclude-alloy-id 3

test -f "08_训练与生成结果/03_结构loss消融/Fold03_C2_gb_edge_20000/final/run_metadata.json"
echo "Both Fold 03 ablations completed successfully."
