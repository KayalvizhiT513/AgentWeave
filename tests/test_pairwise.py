import asyncio
import json

from agentweave.config import Settings
from agentweave.core.enums import AgentRole
from agentweave.core.models import Conversation, ConversationUsage, Exchange, SharedContext
from agentweave.evals.experiment import Experiment
from agentweave.evals.ledger import Ledger
from agentweave.evals.metrics import LexicalEmbedder, transcript_metrics
from agentweave.evals.pairwise import TranscriptPairJudge, compare_experiment, record
from agentweave.evals.report import render_report


def run(coro):
    return asyncio.run(coro)


def conversation(texts: list[str], topic: str = "t") -> Conversation:
    c = Conversation(topic=topic, shared_context=SharedContext(goal=topic))
    for i, text in enumerate(texts):
        c.exchanges.append(Exchange(round_number=i // 2 + 1, agent_id=f"a{i % 2}", role=AgentRole.CRITIC, content=text,
                                    contribution_score=0.5, novelty_score=0.5, repetition_score=0.1))
    return c


class FakeProvider:
    """structured_output picks the longer transcript (by character count) as the winner."""

    def __init__(self, rule=None) -> None:
        self.settings = Settings(openai_api_key="k")
        self.rule = rule

    async def structured_output(self, *, user_input, **kwargs):
        first, second = user_input.split("=====")
        if self.rule:
            return {"rationale": "r", "winner": self.rule(first, second)}
        return {"rationale": "r", "winner": "first" if len(first) > len(second) else "second"}


def test_transcript_judge_is_order_robust() -> None:
    judge = TranscriptPairJudge(FakeProvider(), ConversationUsage())
    long, short = conversation(["a long and detailed point " * 3] * 4), conversation(["brief"] * 4)
    assert run(judge.prefer(long, short)) == 0 and run(judge.prefer(short, long)) == 1
    positional = TranscriptPairJudge(FakeProvider(lambda a, b: "first"), ConversationUsage())
    assert run(positional.prefer(long, short)) is None  # always-first is position noise, not a preference


def test_cut_rate_counts_turns_ended_by_the_word_cap() -> None:
    c = conversation(["a whole sentence.", "cut off mid…", "another whole one.", "cut again…"])
    metrics = run(transcript_metrics(c, LexicalEmbedder()))
    assert metrics["cut_rate"] == 0.5 and metrics["mean_turn_words"] > 0


def test_compare_experiment_groups_by_topic_and_rep(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    experiment = Experiment.model_validate({
        "id": "exp", "hypothesis": "h", "primary_metrics": ["x"], "baseline": "a",
        "variants": {"a": {}, "b": {}, "c": {}}, "runtime": {"max_rounds": 3},
    })
    sizes = {"a": 3, "b": 2, "c": 1}  # transcript length grows from c to a, so a > b > c under FakeProvider
    for topic in ("t1", "t2"):
        for variant, n in sizes.items():
            run_id = f"exp__{variant}__{topic}__r1"
            ledger.append(
                {"run_id": run_id, "attempt": 1, "experiment": "exp", "variant": variant, "topic": topic, "rep": 1,
                 "status": "ok", "config_hash": "h", "git": {"sha": "x", "dirty": False},
                 "models": {"judge": "m", "judge_version": "v", "embedder": "e"}},
                transcript=conversation([f"point {i} " * 5 for i in range(n * 2)], topic=topic).model_dump(mode="json"),
            )
    usage = ConversationUsage()
    judge = TranscriptPairJudge(FakeProvider(), usage)
    outcome = run(compare_experiment(experiment, ledger, judge))
    assert set(outcome["pairs"]) == {"a vs b", "a vs c", "b vs c"}
    assert all(p["comparisons"] == 2 and p["wins_a"] == 2 for p in outcome["pairs"].values())
    assert outcome["pairs"]["a vs b"]["a_win_rate_when_decisive"] == 1.0

    record(ledger, experiment, judge, outcome, usage)
    text = render_report([experiment], ledger)
    assert "Whole-debate comparison" in text and "a vs b" in text
    assert json.loads((tmp_path / "pairwise.jsonl").read_text().splitlines()[-1])["judge"]["version"] == "p1"
