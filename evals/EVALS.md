# AgentWeave eval record

_Generated 2026-10-07 19:45 UTC from `evals/ledger.jsonl` (3 rows, 3 runs). Do not edit by hand: rerun `python -m agentweave.evals report`. Conclusions and decisions go in `JOURNAL.md`._

## Reading this record

- **The judge is not calibrated against humans yet.** Judge scores are comparable across variants within one
  judge version and model, and say nothing about absolute quality.
- **Small n.** Fewer than 10 topics per variant is directional only. No significance tests are run.
- **The in-loop evaluator and agent self-scores are logged (`loop_scores`) but never reported here.**
- **Embedding metrics measure meaning only with a real embedder.** Rows with `lexical-hash-256` measure word
  overlap; do not compare them to rows with a different embedder.

Metric glossary:

- `judge_novelty`: judge 1-5: ideas not predictable from the topic
- `judge_depth`: judge 1-5: mechanisms and tradeoffs worked through
- `judge_disagreement_preserved`: judge 1-5: distinct positions kept explicit
- `novelty_vs_history`: embedding: how far a turn is from everything said before
- `sim_late`: embedding: agreement among agents in the last third of rounds
- `convergence_slope`: embedding: trend of within-round agreement (positive = converging)
- `distinct_2`: lexical: share of unique bigrams
- `rounds`: rounds run before the conversation ended
- `tokens`: total input + output tokens across all calls
- `replacements`: agents swapped out (quality + exploration)

## Judge calibration (human labels)

Chance is 0.50. Dataset: UKPConvArg1Strict (CC-BY-4.0); published feature/LSTM models reach 0.76-0.78.

| run | judge | pairs | pairwise acc (±95%) | consistent | picks first | 1-5 acc (±95%) | 1-5 ties | commit |
|---|---|---|---|---|---|---|---|---|
| cal-20261007-194428 | gpt-5.4-mini v1 | 100 | 0.76 ± 0.07 | 0.84 | 0.51 | 0.72 ± 0.06 | 0.39 | ed3c915+dirty |

Degraded-transcript check, latest run (judge scores, mean over 3 transcripts):

| version | novelty | depth | disagreement kept |
|---|---|---|---|
| original | 4.00 | 4.00 | 5.00 |
| truncated | 4.00 | 4.00 | 4.00 |
| echo | 1.33 | 1.33 | 1.00 |

## 001-baseline-reference

**Hypothesis:** Establishes the reference numbers and run-to-run noise for the current default config. No effect predicted; the spread across reps sets how large a later difference must be to mean anything.

**Primary metrics:** judge_novelty, judge_depth, judge_disagreement_preserved, sim_late  ·  **Baseline:** `default`

Runs from commit(s) ed3c915+dirty · gpt-5.4-mini judge vv1, text-embedding-3-small

### Means (± sd across runs)

| variant | ok / err | judge_novelty | judge_depth | judge_disagreement_preserved | novelty_vs_history | sim_late | convergence_slope | distinct_2 | rounds | tokens | replacements |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `default` | 3 / 0 | 4.00 ± 0.00 | 4.00 ± 0.00 | 5.00 ± 0.00 | 0.37 ± 0.01 | 0.56 ± 0.01 | 0.00 ± 0.00 | 0.81 ± 0.01 | 10.00 ± 0.00 | 59,909 ± 2,051 | 2.00 ± 0.00 |

### Paired change vs `default` (mean of per-topic differences; n = topics in both)

| variant | judge_novelty | judge_depth | judge_disagreement_preserved | novelty_vs_history | sim_late | convergence_slope | distinct_2 | rounds | tokens | replacements |
|---|---|---|---|---|---|---|---|---|---|---|

## 002-thresholds

**Hypothesis:** A lower redundancy cutoff (0.45) replaces agents sooner and raises novelty_vs_history at the cost of tokens and judge_depth; a higher cutoff (0.75) does the opposite. Prediction: monotone in the cutoff for novelty_vs_history, flat or noisy for judge_depth.

**Primary metrics:** novelty_vs_history, judge_novelty, judge_depth, replacements  ·  **Baseline:** `cutoff_0.62`

_No runs recorded yet._

## 003-exploration-ablation

**Hypothesis:** Perspective seeding and diversity pressure each preserve disagreement. Prediction: removing both gives the lowest judge_disagreement_preserved and the highest sim_late; seeding alone recovers most of the gap.

**Primary metrics:** judge_disagreement_preserved, sim_late, judge_novelty  ·  **Baseline:** `full`

_No runs recorded yet._

## 004-role-sampling

**Hypothesis:** Giving the Critic and Chatter higher reasoning effort improves judge_depth, while lower effort on all roles cuts tokens with a small depth loss. Prediction: depth rises with effort; novelty_vs_history is unaffected. Uses reasoning_effort because temperature support depends on the model: check the first run for API errors.

**Primary metrics:** judge_depth, judge_novelty, tokens  ·  **Baseline:** `default`

_No runs recorded yet._
