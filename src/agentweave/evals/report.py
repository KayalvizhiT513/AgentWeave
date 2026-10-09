"""Renders evals/EVALS.md from the ledger. The ledger is the record; this file is a view of it."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentweave.evals.experiment import Experiment
from agentweave.evals.ledger import Ledger

# metric -> what a higher number means. No metric here is "better" on its own.
METRICS: dict[str, str] = {
    "judge_novelty": "judge 1-5: ideas not predictable from the topic",
    "judge_depth": "judge 1-5: mechanisms and tradeoffs worked through",
    "judge_disagreement_preserved": "judge 1-5: distinct positions kept explicit",
    "novelty_vs_history": "embedding: how far a turn is from everything said before",
    "sim_late": "embedding: agreement among agents in the last third of rounds",
    "convergence_slope": "embedding: trend of within-round agreement (positive = converging)",
    "distinct_2": "lexical: share of unique bigrams",
    "cut_rate": "share of turns cut mid-sentence by the 50-word cap",
    "rounds": "rounds run before the conversation ended",
    "tokens": "total input + output tokens across all calls",
    "replacements": "agents swapped out (quality + exploration)",
    "speaker_retries": "extra speaker calls because a model returned nothing",
}

LIMITATIONS = """\
- **The judge is not calibrated against humans yet.** Judge scores are comparable across variants within one
  judge version and model, and say nothing about absolute quality.
- **Small n.** Fewer than 10 topics per variant is directional only. No significance tests are run.
- **The in-loop evaluator and agent self-scores are logged (`loop_scores`) but never reported here.**
- **Embedding metrics measure meaning only with a real embedder.** Rows with `lexical-hash-256` measure word
  overlap; do not compare them to rows with a different embedder.
"""


def _value(row: dict[str, Any], key: str) -> float | None:
    if key == "rounds":
        return (row.get("outcome") or {}).get("rounds")
    if key == "tokens":
        total = (row.get("usage") or {}).get("total") or {}
        return (total.get("input_tokens", 0) + total.get("output_tokens", 0)) or None
    if key == "speaker_retries":
        return (row.get("outcome") or {}).get("speaker_retries")
    if key == "replacements":
        outcome = row.get("outcome") or {}
        if "replacements_quality" not in outcome:
            return None
        return outcome["replacements_quality"] + outcome["replacements_exploration"]
    return (row.get("metrics") or {}).get(key)


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "–"
    if abs(value) >= 1000:
        return f"{value:+,.0f}" if signed else f"{value:,.0f}"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "–"
    if len(values) == 1:
        return _fmt(values[0])
    return f"{_fmt(statistics.mean(values))} ± {_fmt(statistics.stdev(values))}"


def _per_topic(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Mean over reps, per topic, so every topic counts once."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _value(row, key)
        if value is not None:
            grouped[row["topic"]].append(value)
    return {topic: statistics.mean(values) for topic, values in grouped.items()}


def render_experiment(experiment: Experiment, rows: list[dict[str, Any]]) -> str:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    ok = {name: [r for r in rs if r["status"] == "ok"] for name, rs in by_variant.items()}

    out = [f"## {experiment.id}", "", f"**Hypothesis:** {experiment.hypothesis}", ""]
    out.append(f"**Primary metrics:** {', '.join(experiment.primary_metrics)}  ·  **Baseline:** `{experiment.baseline}`")
    out.append("")
    if not rows:
        return "\n".join([*out, "_No runs recorded yet._", ""])

    models = sorted({f"{r['models']['judge']} judge v{r['models']['judge_version']}, {r['models']['embedder']}" for r in rows if "models" in r})
    shas = sorted({f"{r['git']['sha']}{'+dirty' if r['git'].get('dirty') else ''}" for r in rows})
    out.append(f"Runs from commit(s) {', '.join(shas)} · {'; '.join(models)}")
    out.append("")

    names = list(experiment.variants)
    out += ["### Means (± sd across runs)", "", "| variant | ok / err | " + " | ".join(METRICS) + " |"]
    out.append("|" + "---|" * (len(METRICS) + 2))
    for name in names:
        errors = sum(1 for r in by_variant.get(name, []) if r["status"] != "ok")
        cells = [_mean_sd([v for r in ok.get(name, []) if (v := _value(r, key)) is not None]) for key in METRICS]
        out.append(f"| `{name}` | {len(ok.get(name, []))} / {errors} | " + " | ".join(cells) + " |")
    out.append("")

    out += [f"### Paired change vs `{experiment.baseline}` (mean of per-topic differences; n = topics in both)", ""]
    out += ["| variant | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    smallest = None
    for name in names:
        if name == experiment.baseline:
            continue
        cells = []
        for key in METRICS:
            mine, base = _per_topic(ok.get(name, []), key), _per_topic(ok.get(experiment.baseline, []), key)
            shared = sorted(set(mine) & set(base))
            if not shared:
                cells.append("–")
                continue
            smallest = len(shared) if smallest is None else min(smallest, len(shared))
            delta = statistics.mean(mine[t] - base[t] for t in shared)
            cells.append(f"{_fmt(delta, signed=True)} (n={len(shared)})")
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    out.append("")
    if smallest is not None and smallest < 10:
        out += [f"> n={smallest} topics in the smallest comparison: directional only.", ""]

    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        out += ["### Failed runs", ""]
        out += [f"- `{r['run_id']}` (attempt {r['attempt']}): {r.get('error', 'unknown')}" for r in failures]
        out.append("")
    return "\n".join(out)


def render_calibration(ledger: Ledger) -> str:
    path = ledger.directory / "calibration" / "results.jsonl"
    if not path.exists():
        return "## Judge calibration\n\n_Not run yet: `python -m agentweave.evals calibrate --yes`._\n"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = ["## Judge calibration (human labels)", ""]
    out.append("Chance is 0.50. Dataset: UKPConvArg1Strict (CC-BY-4.0); published feature/LSTM models reach 0.76-0.78.")
    out += ["", "| run | judge | pairs | pairwise acc (±95%) | consistent | picks first | 1-5 acc (±95%) | 1-5 ties | commit |", "|" + "---|" * 9]
    for row in rows:
        s, p, a = row["summary"], row["summary"]["pairwise"], row["summary"]["absolute_1to5"]
        ci = lambda block: f"{_fmt(block['mean'])} ± {_fmt(block['ci95'])}" if block.get("mean") is not None else "–"
        out.append(
            f"| {row['calibration_id']} | {row['judge']['model']} {row['judge']['prompt_version']} | {s['pairs']} "
            f"| {ci(p)} | {_fmt(p['consistency'])} | {_fmt(p['position_bias_first'])} | {ci(a)} "
            f"| {_fmt(a['tie_rate'])} | {row['git']['sha']}{'+dirty' if row['git'].get('dirty') else ''} |"
        )
    latest = rows[-1].get("degraded")
    if latest:
        out += ["", f"Degraded-transcript check, latest run (judge scores, mean over {len(latest['rows'])} transcripts):", ""]
        out += ["| version | novelty | depth | disagreement kept |", "|---|---|---|---|"]
        for kind in latest["kinds"]:
            cells = []
            for key in ("judge_novelty", "judge_depth", "judge_disagreement_preserved"):
                values = [r["scores"][kind][key] for r in latest["rows"] if r["scores"].get(kind)]
                cells.append(_fmt(statistics.mean(values)) if values else "–")
            out.append(f"| {kind} | " + " | ".join(cells) + " |")
    return "\n".join([*out, ""])


def render_pairwise(experiment_id: str, ledger: Ledger) -> str:
    path = ledger.directory / "pairwise.jsonl"
    if not path.exists():
        return ""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["experiment"] == experiment_id]
    if not rows:
        return ""
    row = rows[-1]
    out = [
        f"### Whole-debate comparison, judged pairwise (judge {row['judge']['model']} {row['judge']['version']}, both orders)",
        "",
        "Each comparison is one topic and repetition, with the variants' transcripts shown blind and in both orders. "
        "Rates count clear verdicts only; the interval is the 95% margin on that rate.",
        "",
        "| pair (A vs B) | A wins | B wins | no preference | A win rate (±95%) |",
        "|---|---|---|---|---|",
    ]
    for name, p in row["pairs"].items():
        rate = "–" if p["a_win_rate_when_decisive"] is None else f"{p['a_win_rate_when_decisive']:.2f} ± {p['ci95']:.2f}"
        out.append(f"| {name} | {p['wins_a']} | {p['wins_b']} | {p['no_preference']} | {rate} |")
    return "\n".join([*out, ""])


def render_report(experiments: list[Experiment], ledger: Ledger) -> str:
    latest = ledger.latest_by_run()
    head = [
        "# AgentWeave eval record",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from `evals/ledger.jsonl` "
        f"({len(ledger.rows())} rows, {len(latest)} runs). Do not edit by hand: rerun "
        "`python -m agentweave.evals report`. Conclusions and decisions go in `JOURNAL.md`._",
        "",
        "## Reading this record",
        "",
        LIMITATIONS,
        "Metric glossary:",
        "",
        *[f"- `{name}`: {meaning}" for name, meaning in METRICS.items()],
        "",
    ]
    body = [render_calibration(ledger)]
    for experiment in experiments:
        rows = [row for row in latest.values() if row["experiment"] == experiment.id]
        body.append(render_experiment(experiment, rows))
        body.append(render_pairwise(experiment.id, ledger))
    return "\n".join(head + body)


def load_experiments(directory: Path) -> list[Experiment]:
    return [Experiment.model_validate_json(path.read_text()) for path in sorted(directory.glob("*.json"))]
