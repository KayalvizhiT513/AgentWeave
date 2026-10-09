import re

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning_block(text: str) -> str:
    """Drop the empty <think></think> block Qwen's chat template leaves at the start of a reply."""
    return _THINK_BLOCK.sub("", text, count=1).strip()
