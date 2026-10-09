# Eval journal

Human-written. `EVALS.md` is generated from `ledger.jsonl` and shows the numbers; this file records what we
concluded from them and what we changed. Add an entry after every experiment, including the ones that found nothing.

## How the record works

- `experiments/NNN-*.json`: one pre-registered question each. Hypothesis and primary metrics are written
  before the run and are not edited afterwards. A new question is a new file.
- `ledger.jsonl`: append-only, one row per run, with full config, config hash, git sha, model names, metrics
  and a pointer to the transcript. Failed runs are rows too. A rerun adds a row; it never overwrites.
- `transcripts/`: the full conversation for every ledger row.
- `EVALS.md`: `python -m agentweave.evals report` regenerates it from the ledger.
- Defaults live in `RuntimeConfig` and `TuningConfig`; change a default only after an entry below justifies it.

## Entry template

```
### NNN-name: <date>
Predicted: <copy the hypothesis>
Observed: <numbers from EVALS.md, with n>
Noise: <spread from 001 for the same metric>
Decision: <changed a default / kept / need more runs / need a better metric>
```

## Entries

### cal-20261007-194428: judge calibration, 2026-10-07
Predicted: a blunt 1-5 judge would barely separate better from worse; pairwise judging might do better.
Observed (100 human-labeled pairs, UKPConvArg1Strict, 16 debates, gpt-5.4-mini judge v1, 409 calls, about $0.18):
- Pairwise, both orders: 0.76 accuracy (95% interval 0.69-0.83; chance 0.50). Same pick in both orders on 84% of pairs. Picked the first-shown argument 51% of the time, so no position bias.
- 1-5 per-argument scale: tied the two arguments on 39% of pairs. Of the other 61, it ranked 53 right and 8 wrong (87% right when it separated). Scores sat at 1-3; a 4 was given 4 times and a 5 never.
- Degraded transcripts (3 real transcripts, one topic): all speakers echoing one line scored 1.3 / 1.3 / 1.0 against 4 / 4 / 5 for the originals. Cutting to the first 4 turns changed only "disagreement kept" (5 to 4).
Noise: not measured on transcripts yet beyond the 3-rep smoke test (identical 4 / 4 / 5).
Decision: the judge does separate quality when the gap is large, so the earlier "it cannot find differences" reading does not hold. The weakness is resolution at the top of the scale and between similar good transcripts. Add a pairwise A/B transcript comparison (both orders) as the primary judge metric for experiments 002-004. Keep the 1-5 scores as secondary.
Limits: short single arguments, not full transcripts; 3 transcripts from one topic; the degraded versions are extreme. The 95% interval overlaps the published 0.76-0.78, so this says "comparable", not "better".
For tuning later: this 100-pair sample is a judge test set only. A training experiment must split by debate, because the same arguments recur in many pairs and would leak.

### data-pilot-2026-10-08: training-data pilot for the model that drives the agents
Predicted: two hosted candidates per Critic turn plus a pairwise turn judge (t1) would give clean chosen/rejected pairs.
Observed:
- Pilot v0 (Critic only, 3 topics, `pilot-v0-critic-only/`): 17 turns; 20 of 34 candidates were cut by the 50-word cap; the judge's two orders disagreed on 8 of 17 pairs; 4 usable rows.
- Pilot v1 (all speaking roles, 3 topics, `pilot/`, about $0.60): 81 turns, 48 usable rows (59%). Selected by compliance 34, judge 7, tie 7. 62% of all candidates exceeded 50 words (Explorer 90%, Critic 59%, Moderator 50%, Chatter 58%).
- Local base model, untuned (Qwen3-4B-Instruct-2507, 4-bit, 22 real Critic prompts): valid schema JSON 17 of 22; 16 of 17 replies over 50 words (mean 69); about 10 s per turn.
- LoRA smoke test on 40 rows, 30 iterations: runs, about 4 minutes; validation rows shared debates with training rows, so the loss drop (0.81 to 0.38) says only that training works.
Decision: collect from every speaking role, not just the Critic. Most training rows are the hosted model's replies filtered to the word limit, with the judge deciding only a minority, so this is closer to distillation than to preference tuning. Keep the 50-word cap for this run so results stay comparable with `001`. Separate finding: the cap cuts most of the hosted model's own replies mid-sentence, so the current system already shows chopped lines in most turns. Worth its own pre-registered experiment (50 vs 80 words); changing it after training would mean regenerating the data.
Limits: 3 topics; yields will vary by topic; the turn judge (t1) has only been sanity-checked against echoes, not calibrated on human labels.

### critic-v1-2026-10-08: first tuned model (LoRA on Qwen3-4B-Instruct-2507, 4-bit), turn-level
Setup: 712 training rows from 36 debates (hosted gpt-5.4-mini teacher; selection by word-limit compliance 468, judge 175, tie 140; all speaking roles), plain-text replies, LoRA rank 8 on 16 layers, 1,100 iterations at batch 1 (about 1.5 epochs), about 2 hours on an M4 Pro. Generation cost $7.78. Held out: 4 whole debates, 71 prompts. Judge: turn judge t1, both orders, replies cut at 50 words as the debate would show them.
Predicted: tuning closes part of the gap to the hosted model, mainly on the word limit and role style.
Observed:
- Within 50 words: untuned 7%, tuned 63% (iteration 400 checkpoint 49%). Mean words 69.6 to 46.9. No JSON or labels in either.
- Pairwise, decisive verdicts only: tuned beats untuned 36 to 11 (77%, 95% interval about 62-91%; 24 of 71 had no preference). Hosted beats tuned 47 to 2 (96%; 22 no preference) and beats untuned 59 to 2.
- Validation loss fell from 5.36 to 2.23 by iteration 500-700, then rose to 2.70 by 1100 (overfitting by loss), while word-limit compliance kept improving.
Decision: the evidence supports "tuned beats untuned", not "tuned matches hosted". The gap to hosted is large. Treat the hosted comparison as an upper bound on the gap, for three reasons: the hosted replies are the selected winners (compliant and judge-preferred), the judge is the same model family as the teacher (self-preference), and the turn judge t1 is not calibrated on human labels. Open question: how much of the tuned-vs-untuned gain is just not being cut mid-sentence by the 50-word cap, rather than better content. Control to run: judge the two uncut.
Cost of this comparison: about 430 judge calls.
Limits: one run, one seed, 4 held-out debates; no repeat to measure noise; the tuned model has not yet driven a full debate, so debate-level quality is untested.

_Next: uncut control comparison, then wire the tuned model into the provider and run experiment `005` (tuned vs untuned local vs hosted driving whole debates)._
