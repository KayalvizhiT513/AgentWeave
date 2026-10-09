"""
Generate replies from a local MLX model (optionally with a LoRA adapter) for held-out prompts.
Run inside the MLX environment:
  ~/.venvs/agentweave-mlx/bin/python train/generate_local.py --model <dir> [--adapter <dir>] \
      --heldout evals/training/full/sft/heldout.jsonl --out replies_base.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--adapter")
parser.add_argument("--heldout", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--max-tokens", type=int, default=200)
parser.add_argument("--temp", type=float, default=0.7)
args = parser.parse_args()

model, tokenizer = load(args.model, adapter_path=args.adapter)
sampler = make_sampler(temp=args.temp, top_p=0.95)
rows = [json.loads(line) for line in Path(args.heldout).read_text().splitlines() if line.strip()]
with open(args.out, "w") as out:
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        started = time.time()
        reply = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler).strip()
        out.write(json.dumps({"index": index, "reply": reply, "seconds": round(time.time() - started, 2)}) + "\n")
        out.flush()
        print(index, len(reply.split()), "words", flush=True)
