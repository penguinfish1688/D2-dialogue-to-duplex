from __future__ import annotations

from collections.abc import Sequence
from typing import Any


CONVERSATION_SYSTEM_PROMPT = (
    "You are a helpful spoken conversational assistant. Respond naturally "
    "when the user finishes speaking."
)
AVQA_SYSTEM_PROMPT = (
    "You are answering a multiple-choice audio question. Respond with the "
    "correct choice letter followed by the corresponding option wording."
)


def format_system_prompt(content: str) -> str:
    content = str(content).strip()
    if not content:
        raise ValueError("system prompt content must be non-empty")
    return f"<|im_start|>system\n{content}<|im_end|>\n"


def encode_system_prompt(tokenizer: Any, content: str) -> tuple[int, ...]:
    token_ids = tokenizer.encode(
        format_system_prompt(content),
        add_special_tokens=False,
    )
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise TypeError("tokenizer.encode must return a token-ID sequence")
    encoded = tuple(int(token_id) for token_id in token_ids)
    if not encoded or any(token_id < 0 for token_id in encoded):
        raise ValueError("system prompt encoded to invalid token IDs")
    return encoded


def task_system_prompt_token_ids(tokenizer: Any) -> dict[str, tuple[int, ...]]:
    return {
        "conversation": encode_system_prompt(
            tokenizer,
            CONVERSATION_SYSTEM_PROMPT,
        ),
        "avqa": encode_system_prompt(tokenizer, AVQA_SYSTEM_PROMPT),
    }


__all__ = [
    "AVQA_SYSTEM_PROMPT",
    "CONVERSATION_SYSTEM_PROMPT",
    "encode_system_prompt",
    "format_system_prompt",
    "task_system_prompt_token_ids",
]
