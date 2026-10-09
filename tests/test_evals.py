import asyncio
import json

import pytest

from agentweave.config import Settings
from agentweave.core.enums import AgentRole, EvaluationRecommendation
from agentweave.core.models import AgentProfile, Conversation, EvaluationSnapshot, RuntimeConfig, SharedContext
from agentweave.evals.experiment import Experiment, Topic
from agentweave.evals.ledger import Ledger
from agentweave.evals.metrics import LexicalEmbedder, transcript_metrics
from agentweave.evals.report import render_experiment, render_report
from agentweave.evals.runner import plan_runs, run_experiment
from agentweave.services.event_stream import EventBus
from agentweave.services.orchestrator import ConversationOrchestrator
from agentweave.services.provider import AgentResponse, BaseAgentProvider, OpenAIAgentProvider
from agentweave.services.store import ConversationStore
from agentweave.tuning import SamplingProfile, Thresholds, TuningConfig


def asyncio_run(coro):
    return asyncio.run(coro)


# ---- tuning config -------------------------------------------------------------------------------


def test_default_tuning_sends_no_sampling_fields() -> None:
    payload = {"model": "base"}
    OpenAIAgentProvider._apply_sampling(payload, TuningConfig().for_role(AgentRole.CRITIC))
    assert payload == {"model": "base"}


def test_role_profile_overrides_default_field_by_field() -> None:
    tuning = TuningConfig(
        default=SamplingProfile(temperature=0.5, reasoning_effort="low"),
        roles={AgentRole.CRITIC: SamplingProfile(temperature=1.0, model="big")},
    )
    critic = tuning.for_role(AgentRole.CRITIC)
    assert (critic.model, critic.temperature, critic.reasoning_effort) == ("big", 1.0, "low")
    assert tuning.for_role(AgentRole.CHATTER).temperature == 0.5


def test_sampling_applied_to_payload() -> None:
    payload = {"model": "base"}
    profile = SamplingProfile(model="m2", temperature=0.2, top_p=0.9, reasoning_effort="high")
    OpenAIAgentProvider._apply_sampling(payload, profile)
    assert payload == {"model": "m2", "temperature": 0.2, "top_p": 0.9, "reasoning": {"effort": "high"}}


def test_tuning_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError):
        TuningConfig(thresholds={"redundancy_cutof": 0.5})


def test_fingerprint_changes_with_config() -> None:
    assert TuningConfig().fingerprint() == TuningConfig().fingerprint()
    assert TuningConfig().fingerprint() != TuningConfig(thresholds=Thresholds(redundancy_cutoff=0.5)).fingerprint()


def test_provider_uses_role_sampling_in_request(monkeypatch) -> None:
    sent: list[dict] = []

    class Client:
        def __init__(self, *a, **k) -> None: ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None

        async def post(self, url, headers, json):
            sent.append(json)

            class R:
                status_code = 200
                text = ""

                def json(self_inner):
                    return {
                        "output_text": '{"content":"x","contribution_score":0.5,"novelty_score":0.5,'
                        '"repetition_score":0.1,"state_update":null}'
                    }

            return R()

    monkeypatch.setattr("agentweave.services.provider.httpx.AsyncClient", Client)
    provider = OpenAIAgentProvider(Settings(openai_api_key="k"))
    critic = AgentProfile(role=AgentRole.CRITIC, personality="p")
    chatter = AgentProfile(role=AgentRole.CHATTER, personality="p")
    conversation = Conversation(
        topic="t",
        shared_context=SharedContext(goal="t"),
        agents=[critic, chatter],
        tuning=TuningConfig(roles={AgentRole.CRITIC: SamplingProfile(reasoning_effort="high")}),
    )
    asyncio_run(provider.respond(conversation, critic))
    asyncio_run(provider.respond(conversation, chatter))
    assert sent[0]["reasoning"] == {"effort": "high"}
    assert "reasoning" not in sent[1]


# ---- orchestration thresholds --------------------------------------------------------------------


class Scripted(BaseAgentProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def respond(self, conversation, agent) -> AgentResponse:
        self.calls += 1
        word = f"{agent.role.value}{conversation.current_round}{self.calls}"
        return AgentResponse(
            content=f"{word} unique argument number {self.calls} about {conversation.topic}",
            contribution_score=0.6,
            novelty_score=0.5,
            repetition_score=0.2,
        )

    async def evaluate(self, conversation) -> EvaluationSnapshot:
        return EvaluationSnapshot(
            round_number=conversation.current_round,
            progress_score=0.5,
            novelty_score=0.5,
            coherence_score=0.5,
            redundancy_score=0.55,
            goal_alignment_score=0.5,
            depth_score=0.5,
            conflict_utility_score=0.5,
            recommendation=EvaluationRecommendation.CONTINUE,
            rationale="ok",
        )


def test_failure_mode_reads_thresholds_from_conversation() -> None:
    orchestrator = ConversationOrchestrator(ConversationStore(), EventBus(), Scripted())
    evaluation = asyncio_run(Scripted().evaluate(Conversation(topic="t", shared_context=SharedContext(goal="t"), current_round=1)))
    loose = Conversation(topic="t", shared_context=SharedContext(goal="t"), current_round=1)
    strict = Conversation(
        topic="t",
        shared_context=SharedContext(goal="t"),
        current_round=1,
        tuning=TuningConfig(thresholds=Thresholds(redundancy_cutoff=0.5)),
    )
    assert orchestrator._determine_failure_mode(loose, evaluation) != "too_repetitive"
    assert orchestrator._determine_failure_mode(strict, evaluation) == "too_repetitive"


# ---- experiment definition -----------------------------------------------------------------------


def make_experiment(**overrides) -> Experiment:
    spec = {
        "id": "exp",
        "hypothesis": "h",
        "primary_metrics": ["judge_depth"],
        "runtime": {"max_rounds": 3, "agent_turn_delay_seconds": 0, "perspective_dimensions": 0},
        "baseline": "a",
        "variants": {"a": {}, "b": {"tuning": {"thresholds": {"redundancy_cutoff": 0.4}}}},
    }
    spec.update(overrides)
    return Experiment.model_validate(spec)


def test_experiment_resolves_overrides() -> None:
    experiment = make_experiment()
    runtime, tuning = experiment.resolve("b")
    assert runtime.max_rounds == 3
    assert tuning.thresholds.redundancy_cutoff == 0.4
    assert experiment.resolve("a")[1].thresholds.redundancy_cutoff == 0.62


@pytest.mark.parametrize(
    "bad",
    [
        {"baseline": "missing"},
        {"runtime": {"max_round": 3}},
        {"variants": {"a": {"tuning": {"nope": 1}}}},
        {"variants": {"a": {"runtime": {"evaluation_intervall": 2}}}},
    ],
)
def test_experiment_rejects_typos(bad) -> None:
    with pytest.raises(ValueError):
        make_experiment(**bad)


def test_shipped_experiments_are_valid() -> None:
    from agentweave.evals.ledger import EVALS_DIR
    from agentweave.evals.report import load_experiments

    experiments = load_experiments(EVALS_DIR / "experiments")
    assert len(experiments) >= 4
    topics = json.loads((EVALS_DIR / "topics.json").read_text())
    assert all(Topic.model_validate(item) for item in topics)


# ---- metrics -------------------------------------------------------------------------------------


def _conversation_with(texts_by_round: list[list[str]]) -> Conversation:
    from agentweave.core.models import Exchange

    conversation = Conversation(topic="t", shared_context=SharedContext(goal="t"))
    for round_number, texts in enumerate(texts_by_round, start=1):
        for index, text in enumerate(texts):
            conversation.exchanges.append(
                Exchange(
                    round_number=round_number,
                    agent_id=f"a{index}",
                    role=AgentRole.CRITIC if index else AgentRole.CHATTER,
                    content=text,
                    contribution_score=0.5,
                    novelty_score=0.5,
                    repetition_score=0.1,
                )
            )
    return conversation


def test_identical_turns_look_converged_and_distinct_turns_do_not() -> None:
    same = _conversation_with([["we should tax carbon now"] * 2] * 4)
    different = _conversation_with(
        [["tax carbon heavily", "subsidize nuclear reactors"], ["ban gas boilers", "plant forests widely"]] * 2
    )
    embedder = LexicalEmbedder()
    same_metrics = asyncio_run(transcript_metrics(same, embedder))
    different_metrics = asyncio_run(transcript_metrics(different, embedder))
    assert same_metrics["sim_late"] == pytest.approx(1.0)
    assert different_metrics["sim_late"] < 0.3
    assert same_metrics["novelty_vs_history"] < different_metrics["novelty_vs_history"]
    assert set(different_metrics["novelty_vs_history_by_role"]) == {"chatter", "critic"}


def test_metrics_handle_empty_transcript() -> None:
    empty = Conversation(topic="t", shared_context=SharedContext(goal="t"))
    assert asyncio_run(transcript_metrics(empty, LexicalEmbedder()))["turns"] == 0


# ---- runner, ledger, report ----------------------------------------------------------------------


class FakeJudge:
    version = "test"
    model = "fake-judge"

    async def score(self, conversation, usage) -> dict:
        usage.record("judge_transcript", 10, 5)
        return {"judge_novelty": 3, "judge_depth": 4, "judge_disagreement_preserved": 2, "judge_rationale": "r"}


class Exploding(Scripted):
    async def respond(self, conversation, agent):
        raise RuntimeError("model call failed")


TOPICS = [Topic(id="t1", topic="Topic one"), Topic(id="t2", topic="Topic two")]


def _run(experiment, ledger, provider=None, **kwargs):
    return asyncio_run(
        run_experiment(
            experiment,
            TOPICS,
            provider=provider or Scripted(),
            judge=FakeJudge(),
            embedder=LexicalEmbedder(),
            ledger=ledger,
            models={"agent_default": "m", "evaluator": "m"},
            **kwargs,
        )
    )


def test_run_records_rows_and_resumes(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    experiment = make_experiment(reps=2)
    assert len(plan_runs(experiment, TOPICS)) == 8

    first = _run(experiment, ledger, concurrency=3)
    assert first == {"ok": 8, "error": 0, "skipped": 0}
    rows = ledger.rows()
    assert len(rows) == 8
    row = rows[0]
    assert row["metrics"]["judge_depth"] == 4
    assert row["config"]["tuning"]["thresholds"]["redundancy_cutoff"] in (0.62, 0.4)
    assert row["usage"]["total"]["calls"] == 0  # scripted provider records no usage
    assert row["judge_usage"]["total"]["calls"] == 1
    assert (tmp_path / row["transcript"]).exists()

    assert _run(experiment, ledger) == {"ok": 0, "error": 0, "skipped": 8}
    assert len(ledger.rows()) == 8


def test_changed_config_or_rerun_appends_new_attempts(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    _run(make_experiment(), ledger)
    same_b = {"tuning": {"thresholds": {"redundancy_cutoff": 0.4}}}
    changed = make_experiment(variants={"a": {"tuning": {"thresholds": {"novelty_floor": 0.2}}}, "b": same_b})
    result = _run(changed, ledger)
    assert result == {"ok": 2, "error": 0, "skipped": 2}  # a reran (new hash), b skipped
    assert max(row["attempt"] for row in ledger.rows()) == 2
    assert _run(make_experiment(), ledger, rerun=True)["ok"] == 4


def test_failed_run_is_recorded_not_dropped(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    counts = _run(make_experiment(), ledger, provider=Exploding())
    assert counts["error"] == 4 and counts["ok"] == 0
    assert all("model call failed" in row["error"] for row in ledger.rows())
    # An errored run is retried next time instead of being skipped.
    assert _run(make_experiment(), ledger)["ok"] == 4


def test_report_summarizes_variants_and_flags_small_n(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    experiment = make_experiment()
    _run(experiment, ledger)
    text = render_report([experiment], ledger)
    assert "## exp" in text and "`b`" in text
    assert "directional only" in text
    assert "judge_depth" in text
    empty = render_experiment(make_experiment(id="other"), [])
    assert "No runs recorded yet" in empty
