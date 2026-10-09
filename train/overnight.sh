#!/usr/bin/env bash
# Waits for data generation, builds the SFT files, trains the LoRA, then generates held-out replies from
# the untuned and tuned models. Spends no API money. Run from the repo root: caffeinate -i train/overnight.sh
set -euo pipefail
export PYTHONPATH=src
OUT=evals/training/full
PY_MLX="$HOME/.venvs/agentweave-mlx/bin/python"
MODEL="$HOME/.lmstudio/models/lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
ADAPTER=train/adapters/critic-v1

echo "[$(date +%H:%M:%S)] waiting for data generation"
while pgrep -f "agentweave.training gen" >/dev/null; do sleep 10; done

errors=$(python3 -c "import json;print(sum(json.loads(l)['status']!='ok' for l in open('$OUT/runs.jsonl') if l.strip()))")
done_topics=$(python3 -c "print(sum(1 for l in open('$OUT/runs.jsonl') if l.strip()))")
echo "[$(date +%H:%M:%S)] generation ended: $done_topics debates, $errors failed"
if [ "$errors" -ne 0 ] || [ "$done_topics" -lt 40 ]; then echo "STOP: generation incomplete; not training"; exit 1; fi

python3 -m agentweave.training build --out "$OUT"
train_rows=$(wc -l < "$OUT/sft/train.jsonl")
echo "[$(date +%H:%M:%S)] training rows: $train_rows"
if [ "$train_rows" -lt 300 ]; then echo "STOP: fewer than 300 training rows; not training"; exit 1; fi

echo "[$(date +%H:%M:%S)] training"
DATA="$OUT/sft" OUT="$ADAPTER" train/train_lora.sh 1100

echo "[$(date +%H:%M:%S)] generating held-out replies"
"$PY_MLX" train/generate_local.py --model "$MODEL" --heldout "$OUT/sft/heldout.jsonl" --out "$OUT/replies_base.jsonl"
"$PY_MLX" train/generate_local.py --model "$MODEL" --adapter "$ADAPTER" --heldout "$OUT/sft/heldout.jsonl" --out "$OUT/replies_tuned.jsonl"
echo "[$(date +%H:%M:%S)] DONE"
