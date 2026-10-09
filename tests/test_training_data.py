import asyncio
import json

from agentweave.config import Settings
from agentweave.core.enums import AgentRole
from agentweave.core.models import AgentProfile, Conversation, ConversationUsage, SharedContext
from agentweave.services.provider import AgentResponse, OpenAIAgentProvider
from agentweave.training.datagen import TurnDataProvider, build_sft
from agentweave.training.turn_judge import TurnContext, TurnJudge

CONTEXT = TurnContext(topic="t", scene=None, role="critic", role_brief="b", history=["x"])


def run(coro):
    return asyncio.run(coro)


class FakeStructured:
    """Stands in for the provider's structured_output; `rule(user_input)` returns 'first' or 'second'."""

    def __init__(self, rule) -> None:
        self.settings = Settings(openai_api_key="k")
        self.rule = rule

    async def structured_output(self, *, user_input, **kwargs):
        return {"rationale": "r", "winner": self.rule(user_input)}


def judge_with(rule) -> TurnJudge:
    return TurnJudge(FakeStructured(rule), ConversationUsage())


def test_judge_returns_none_when_orders_disagree() -> None:
    assert run(judge_with(lambda text: "first").prefer(CONTEXT, "A", "B")) is None


def test_judge_follows_a_consistent_preference() -> None:
    def prefers_good(text: str) -> str:
        first = text.split("First candidate reply:\n")[1].split("\n\nSecond")[0]
        return "first" if first == "good" else "second"

    judge = judge_with(prefers_good)
    assert run(judge.prefer(CONTEXT, "good", "bad")) == 0
    assert run(judge.prefer(CONTEXT, "bad", "good")) == 1


def sample(topic_id: str, winner, *, truncated=False, words=10, how="judge", role="critic") -> dict:
    def response(content: str) -> dict:
        return {"content": content, "contribution_score": 0.5, "novelty_score": 0.5, "repetition_score": 0.1,
                "state_update": {"core_thesis": "c", "assumptions": [], "causal_model": [], "claims": [], "concessions": [], "unresolved_attacks": []}}
    return {
        "topic_id": topic_id, "topic": f"Topic {topic_id}", "scene": None, "history_tail": ["earlier line"], "system": "SYS", "user": f"USR {topic_id}", "system_plain": "SYSP", "user_plain": f"USRP {topic_id}", "winner": winner, "selected_by": how if winner is not None else None, "role": role,
        "candidates": [
            {"response": response(" ".join(["w"] * words)), "truncated": truncated},
            {"response": response("loser reply"), "truncated": False},
        ],
        "judge": {"version": "t1", "model": "m"},
    }


def test_build_sft_filters_and_splits_by_topic(tmp_path) -> None:
    rows = []
    for index in range(10):
        rows += [sample(f"t{index}", 0), sample(f"t{index}", 1, how="tie", role="chatter")]
    rows += [sample("t0", None), sample("t1", 0, truncated=True), sample("t2", 0, words=60)]
    rows += [sample("t3", 0, how="compliance")]
    source = tmp_path / "samples.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in rows))

    stats = build_sft(source, tmp_path / "sft", valid_fraction=0.2)
    assert stats["no_usable_reply"] == 1 and stats["skipped_by_filter"] == 2
    assert stats["train"] + stats["valid"] == 21
    assert stats["by_selection"] == {"judge": 10, "tie": 10, "compliance": 1}
    assert stats["by_role"] == {"critic": 11, "chatter": 10}
    assert stats["pairs"] == 10  # only consistent judge decisions become preference pairs

    train = [json.loads(l) for l in (tmp_path / "sft" / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(l) for l in (tmp_path / "sft" / "valid.jsonl").read_text().splitlines()]
    first = train[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "assistant"]
    assert first[0]["content"] == "SYSP"  # the plain-text prompt, not the JSON one
    assert first[2]["content"].startswith("w w")  # the target is the spoken reply itself, not JSON
    users = lambda rows: {r["messages"][1]["content"] for r in rows}  # noqa: E731
    assert users(valid) and not users(valid) & users(train)  # no topic appears on both sides
    pairs = [json.loads(l) for l in (tmp_path / "sft" / "pairs.jsonl").read_text().splitlines()]
    assert all(p["rejected"] != p["chosen"] and isinstance(p["chosen"], str) for p in pairs)

    heldout = [json.loads(l) for l in (tmp_path / "sft" / "heldout.jsonl").read_text().splitlines()]
    assert len(heldout) == len(valid) and all(h["reference"] and h["system"] == "SYSP" for h in heldout)

    only_judge = build_sft(source, tmp_path / "judge_only", valid_fraction=0.2, judge_only=True)
    assert only_judge["train"] + only_judge["valid"] == 10


class FakeTurnJudge:
    version, model = "t1", "m"

    def __init__(self, result) -> None:
        self.result = result

    async def prefer(self, context, a, b):
        return self.result


def make_conversation(round_number: int) -> tuple[Conversation, AgentProfile, AgentProfile]:
    critic = AgentProfile(role=AgentRole.CRITIC, personality="p")
    chatter = AgentProfile(role=AgentRole.CHATTER, personality="p")
    conversation = Conversation(topic="t", shared_context=SharedContext(goal="t"), agents=[critic, chatter], current_round=round_number)
    return conversation, critic, chatter


def scripted_provider(monkeypatch, contents: list[str], verdict):
    queue = list(contents)
    calls: list[AgentRole] = []

    async def fake_respond(self, conversation, agent):
        calls.append(agent.role)
        return AgentResponse(content=queue.pop(0), contribution_score=0.5, novelty_score=0.5, repetition_score=0.1)

    monkeypatch.setattr(OpenAIAgentProvider, "respond", fake_respond)
    sink: list[dict] = []
    provider = TurnDataProvider(Settings(openai_api_key="k"), FakeTurnJudge(verdict), sink.append, k=2)
    return provider, sink, calls


def test_two_fitting_replies_go_to_the_judge(monkeypatch) -> None:
    provider, sink, calls = scripted_provider(monkeypatch, ["first fits", "second fits"], verdict=1)
    conversation, critic, _ = make_conversation(round_number=3)
    reply = run(provider.respond(conversation, critic))
    assert len(calls) == 2
    assert (sink[0]["winner"], sink[0]["selected_by"]) == (1, "judge")
    assert reply.content == "second fits"


def test_judge_disagreement_is_recorded_as_a_tie_and_uses_the_first(monkeypatch) -> None:
    provider, sink, _ = scripted_provider(monkeypatch, ["alpha", "beta"], verdict=None)
    conversation, critic, _ = make_conversation(round_number=2)
    assert run(provider.respond(conversation, critic)).content == "alpha"
    assert (sink[0]["winner"], sink[0]["selected_by"]) == (0, "tie")


def test_a_cut_reply_loses_to_one_that_fits_without_calling_the_judge(monkeypatch) -> None:
    class Boom(FakeTurnJudge):
        async def prefer(self, context, a, b):
            raise AssertionError("judge should not run with only one fitting reply")

    long_reply = " ".join(["w"] * 60)
    provider, sink, _ = scripted_provider(monkeypatch, [long_reply[:20] + "…", "short and whole"], verdict=0)
    provider.judge = Boom(0)
    conversation, critic, _ = make_conversation(round_number=2)
    assert run(provider.respond(conversation, critic)).content == "short and whole"
    assert (sink[0]["winner"], sink[0]["selected_by"]) == (1, "compliance")
    assert [c["truncated"] for c in sink[0]["candidates"]] == [True, False]


def test_over_limit_text_counts_as_not_fitting(monkeypatch) -> None:
    provider, sink, _ = scripted_provider(monkeypatch, [" ".join(["w"] * 51), " ".join(["w"] * 60)], verdict=0)
    conversation, critic, _ = make_conversation(round_number=2)
    run(provider.respond(conversation, critic))
    assert (sink[0]["winner"], sink[0]["selected_by"]) == (None, None)  # no training row, debate continues


def test_every_speaking_role_is_sampled_but_not_round_one_or_the_evaluator(monkeypatch) -> None:
    provider, sink, calls = scripted_provider(monkeypatch, ["a1", "a2", "b1", "b2", "c1", "d1", "e1"], verdict=0)
    conversation, critic, chatter = make_conversation(round_number=3)
    run(provider.respond(conversation, critic))
    run(provider.respond(conversation, chatter))
    assert len(sink) == 2  # both speaking roles sampled twice

    evaluator = AgentProfile(role=AgentRole.EVALUATOR, personality="p")
    run(provider.respond(conversation, evaluator))
    assert len(sink) == 2 and len(calls) == 5  # one plain call, no sampling

    early, early_critic, _ = make_conversation(round_number=1)
    run(provider.respond(early, early_critic))
    assert len(sink) == 2 and len(calls) == 6  # round 1 has no dialogue to respond to


def test_build_refuses_samples_without_plain_prompts(tmp_path) -> None:
    import pytest

    old = sample("t0", 0)
    del old["system_plain"], old["user_plain"]
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="plain-text prompts"):
        build_sft(source, tmp_path / "sft")


def test_plain_prompt_drops_json_and_state_update_instructions() -> None:
    from agentweave.core.models import AgentState

    provider = OpenAIAgentProvider(Settings(openai_api_key="k"))
    agent = AgentProfile(role=AgentRole.CRITIC, personality="p", state=AgentState(core_thesis="T", claims=["c"]))
    conversation = Conversation(topic="t", shared_context=SharedContext(goal="t"), agents=[agent], current_round=2)
    structured = provider._agent_instructions(agent), provider._agent_input(conversation, agent)[0]["content"]
    plain = provider._agent_instructions(agent, plain=True), provider._agent_input(conversation, agent, plain=True)[0]["content"]
    assert "structured JSON" in structured[0] and "state_update" in structured[1]
    assert "structured JSON" not in plain[0] and "plain text" in plain[0]
    assert "state_update" not in plain[1] and "Core thesis: T" in plain[1]  # state is still shown, just not rewritten
