#!/usr/bin/env bash
set -euo pipefail

# Run from the existing project root using the same Python environment as evaluation.
test -f train_diffusion.py
python -c 'import torch, matplotlib; assert torch.cuda.is_available(), "CUDA required"'
FOLLOWUP_PARENT='08_训练与生成结果/14_Fold03_MSE_结构统计_5k_运行'
test -f "$FOLLOWUP_PARENT/experiment_manifest.json"
FOLLOWUP_ROOT="$(mktemp -d "$FOLLOWUP_PARENT/followup_20261101_XXXXXX")"
printf 'New output: %s\n' "$FOLLOWUP_ROOT"

python -u -m scripts.diagnose_structure_trajectory \
  --experiment "$FOLLOWUP_PARENT" --mode heldout --output "$FOLLOWUP_ROOT/heldout"

python -u -m scripts.diagnose_structure_trajectory \
  --experiment "$FOLLOWUP_PARENT" --mode trace --output "$FOLLOWUP_ROOT/trajectory"

tar -czf "${FOLLOWUP_ROOT}.tar.gz" "$FOLLOWUP_ROOT"
printf 'ALL COMPLETED. Download: %s.tar.gz\n' "$FOLLOWUP_ROOT"
