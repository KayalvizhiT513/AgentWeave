import asyncio

import pytest

from agentweave.core.enums import AgentRole
from agentweave.core.models import Conversation, ConversationUsage, Exchange, SharedContext
from agentweave.evals.calibration import (
    Pair,
    degrade,
    load_pairs,
    run_degraded_check,
    run_pair_calibration,
    sample_pairs,
)


def run(coro):
    return asyncio.run(coro)


def make_pairs(count: int) -> list[Pair]:
    return [
        Pair(f"p{i}", f"debate {i % 4}", "side", f"strong argument with evidence {i}", f"weak {i}")
        for i in range(count)
    ]


class Oracle:
    """Always picks the human winner (recognized by the word 'strong')."""

    model = "oracle"

    async def compare(self, debate, stance, first, second):
        return "first" if "strong" in first else "second"

    async def rate(self, debate, stance, argument):
        return 4 if "strong" in argument else 2


class AlwaysFirst:
    model = "positional"

    async def compare(self, debate, stance, first, second):
        return "first"

    async def rate(self, debate, stance, argument):
        return 3  # no resolution: everything ties


class Flaky(Oracle):
    def __init__(self) -> None:
        self.calls = 0

    async def compare(self, debate, stance, first, second):
        self.calls += 1
        if self.calls % 2:
            raise RuntimeError("boom")
        return await super().compare(debate, stance, first, second)


def test_oracle_judge_scores_perfectly() -> None:
    summary, records = run(run_pair_calibration(make_pairs(8), Oracle()))
    assert summary["pairwise"]["mean"] == 1.0
    assert summary["pairwise"]["consistency"] == 1.0
    assert summary["pairwise"]["position_bias_first"] == pytest.approx(0.5)
    assert summary["absolute_1to5"]["mean"] == 1.0
    assert summary["absolute_1to5"]["tie_rate"] == 0.0
    assert len(records) == 8


def test_positional_judge_is_chance_and_flagged() -> None:
    summary, _ = run(run_pair_calibration(make_pairs(8), AlwaysFirst()))
    pairwise = summary["pairwise"]
    assert pairwise["mean"] == pytest.approx(0.5)  # right in one order, wrong in the other
    assert pairwise["consistency"] == 0.0
    assert pairwise["position_bias_first"] == 1.0
    assert summary["absolute_1to5"]["tie_rate"] == 1.0
    assert summary["absolute_1to5"]["mean"] == pytest.approx(0.5)  # ties count half
    assert summary["absolute_1to5"]["score_histogram"]["3"] == 16


def test_failed_calls_are_counted_not_dropped(monkeypatch) -> None:
    async def no_sleep(_):  # skip the backoff delay
        return None

    monkeypatch.setattr("agentweave.evals.calibration.asyncio.sleep", no_sleep)
    # Every first attempt fails and the retry succeeds, so nothing is lost.
    summary, _ = run(run_pair_calibration(make_pairs(6), Flaky(), concurrency=1))
    assert summary["pairwise"]["errors"] == 0
    assert summary["pairwise"]["n"] == 6

    class Dead(Oracle):
        async def compare(self, debate, stance, first, second):
            raise RuntimeError("down")

    summary, records = run(run_pair_calibration(make_pairs(6), Dead(), concurrency=1))
    assert summary["pairwise"]["errors"] == 12  # 6 pairs x 2 orders, all counted
    assert summary["pairwise"]["mean"] is None
    assert summary["absolute_1to5"]["mean"] == 1.0  # the other check is unaffected
    assert all(r["forward"] is None for r in records)


def test_loader_reads_tab_separated_files_with_quotes(tmp_path) -> None:
    (tmp_path / "ban-plastic-bottles_no-bad-for-the-economy.csv").write_text(
        '#id\tlabel\ta1\ta2\n'
        'x_y\ta2\tshort "quoted" one\tthe stronger one\n'
        'bad_row\ta9\tonly\tthree\n'
        'e_f\ta1\t\tempty first\n',
        encoding="utf-8",
    )
    pairs = load_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].winner == "the stronger one"
    assert pairs[0].loser == 'short "quoted" one'
    assert pairs[0].debate == "ban plastic bottles"
    assert pairs[0].stance == "no bad for the economy"


def test_sample_is_deterministic_and_spread_across_debates() -> None:
    pairs = make_pairs(40)
    first = sample_pairs(pairs, 8, seed=3)
    assert [p.id for p in first] == [p.id for p in sample_pairs(pairs, 8, seed=3)]
    assert [p.id for p in first] != [p.id for p in sample_pairs(pairs, 8, seed=4)]
    assert {p.debate for p in first} == {"debate 0", "debate 1", "debate 2", "debate 3"}
    assert len({p.id for p in first}) == 8


def make_conversation(turns: int = 6) -> Conversation:
    conversation = Conversation(topic="t", shared_context=SharedContext(goal="t"))
    for index in range(turns):
        conversation.exchanges.append(
            Exchange(
                round_number=index // 3 + 1,
                agent_id=f"a{index % 3}",
                role=AgentRole.CRITIC,
                content=f"distinct point number {index}",
                contribution_score=0.5,
                novelty_score=0.5,
                repetition_score=0.1,
            )
        )
    return conversation


def test_degradations() -> None:
    conversation = make_conversation()
    assert len(degrade(conversation, "truncated").exchanges) == 4
    echo = degrade(conversation, "echo")
    assert {turn.content for turn in echo.exchanges} == {"distinct point number 0"}
    assert [turn.agent_id for turn in echo.exchanges] == [turn.agent_id for turn in conversation.exchanges]
    assert degrade(conversation, "original").exchanges[3].content == "distinct point number 3"
    assert conversation.exchanges[3].content == "distinct point number 3"  # source untouched
    with pytest.raises(ValueError):
        degrade(conversation, "unknown")


def test_degraded_check_scores_each_version() -> None:
    class Judge:
        version, model = "t", "t"

        async def score(self, conversation, usage):
            distinct = len({turn.content for turn in conversation.exchanges})
            return {
                "judge_novelty": distinct,
                "judge_depth": len(conversation.exchanges),
                "judge_disagreement_preserved": distinct,
                "judge_rationale": "r",
            }

    result = run(run_degraded_check([make_conversation()], Judge(), ConversationUsage()))
    scores = result["rows"][0]["scores"]
    assert scores["original"]["judge_novelty"] == 6
    assert scores["truncated"]["judge_depth"] == 4
    assert scores["echo"]["judge_novelty"] == 1
    assert "judge_rationale" not in scores["echo"]
