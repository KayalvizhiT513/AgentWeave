#!/usr/bin/env bash
# LoRA fine-tune of the local base model on the plain-text SFT data. Usage: train/train_lora.sh [ITERS]
# Run from the repo root. The adapter and a copy of these settings are written next to each other.
set -euo pipefail

MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit}"
DATA="${DATA:-evals/training/full/sft}"
ITERS="${1:-1100}"           # about two passes over ~560 rows at batch size 1
OUT="${OUT:-train/adapters/critic-v1}"

mkdir -p "$OUT"
cat > "$OUT/settings.txt" <<SETTINGS
model=$MODEL
data=$DATA
iters=$ITERS
batch_size=1
num_layers=16
learning_rate=1e-4
max_seq_length=3072
mask_prompt=true
SETTINGS

"$HOME/.venvs/agentweave-mlx/bin/python" -m mlx_lm lora \
  --model "$MODEL" --train --data "$DATA" \
  --fine-tune-type lora --mask-prompt --num-layers 16 \
  --batch-size 1 --iters "$ITERS" --learning-rate 1e-4 \
  --max-seq-length 3072 --grad-checkpoint \
  --steps-per-eval 100 --val-batches 25 --save-every 200 \
  --adapter-path "$OUT" 2>&1 | tee "$OUT/train.log"
