import asyncio

from agentweave.training.compare import clean, compare_systems, normalize, reply_stats


class PreferLonger:
    version, model = "t1", "m"

    async def prefer(self, context, a, b):
        return 0 if len(a) > len(b) else 1 if len(b) > len(a) else None


def test_normalize_cuts_like_the_orchestrator() -> None:
    text = " ".join(f"w{i}" for i in range(60))
    out = normalize(text)
    assert out.endswith("…") and len(out.split()) == 50
    assert normalize("  short   reply ") == "short reply"


def test_reply_stats_flags_limit_and_json() -> None:
    stats = reply_stats(["fine reply", " ".join(["w"] * 60), '{"content": "x"}', "critic: label"])
    assert stats["within_limit"] == 0.75
    assert stats["json_or_label_like"] == 0.5


def test_pairwise_win_rates() -> None:
    heldout = [
        {"topic": "t", "scene": None, "role": "critic", "history_tail": ["x"]} for _ in range(4)
    ]
    replies = {
        "hosted": ["aaaa"] * 4,
        "tuned": ["aaaaaa", "aaaaaa", "aaaaaa", "aaaa"],  # longer on 3, equal on 1
    }
    result = asyncio.run(compare_systems(heldout, replies, PreferLonger(), lambda role: "brief"))
    pair = result["pairwise"]["hosted vs tuned"]
    assert (pair["wins_a"], pair["wins_b"], pair["no_preference"]) == (0, 3, 1)
    assert pair["a_win_rate_when_decisive"] == 0.0
    assert result["systems"]["tuned"]["n"] == 4


def test_empty_think_block_is_not_part_of_the_reply() -> None:
    assert clean("<think>\n\n</think>\n\nA sharp reply.") == "A sharp reply."
    assert clean("Keep <think> in the middle.") == "Keep <think> in the middle."
    assert normalize("<think>\n\n</think>\n\n" + " ".join(["w"] * 50)).endswith("w")  # not cut: 50 words
    assert reply_stats(["<think></think> one two"])["mean_words"] == 2
