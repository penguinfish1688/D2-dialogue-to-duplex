"""Qwen D2 task instructions encoded by the released Omni2 tokenizer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from d2.prompts import (
    AVQA_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    encode_system_prompt,
    format_system_prompt,
)

from d2_llama.core.constants import THINKER_BASE_VOCAB_SIZE


TASK_PROMPT_CONTRACT = "d2.llama_omni2.qwen_task_system_prompts.v1"
TASK_PROMPT_PLACEMENT = "once_at_conversation_start_serial_prefix_no_targets_v1"
TASK_PROMPT_TEXTS = {
    "conversation": CONVERSATION_SYSTEM_PROMPT,
    "avqa": AVQA_SYSTEM_PROMPT,
}


def normalize_task_kind(task_kind: str | None) -> str:
    if task_kind in ("instructs2s_dynamic", "soda_dynamic"):
        return "conversation"
    if task_kind not in TASK_PROMPT_TEXTS:
        raise ValueError("task_kind must be conversation or avqa")
    return task_kind


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_task_prompt_contract(tokenizer: Any) -> dict[str, Any]:
    payload = {
        "format": TASK_PROMPT_CONTRACT,
        "source": "d2.prompts",
        "placement": TASK_PROMPT_PLACEMENT,
        "tasks": {
            kind: {
                "text": text,
                "formatted_text": format_system_prompt(text),
                "token_ids": list(encode_system_prompt(tokenizer, text)),
            }
            for kind, text in TASK_PROMPT_TEXTS.items()
        },
    }
    payload["sha256"] = _digest(payload)
    return validate_task_prompt_contract(payload)


def validate_task_prompt_contract(
    contract: Mapping[str, Any], tokenizer: Any = None
) -> dict[str, Any]:
    """Reject altered instructions, token IDs, or tokenizer/checkpoint drift."""
    if not isinstance(contract, Mapping):
        raise ValueError("task system prompt contract must be a mapping")
    value = deepcopy(dict(contract))
    if (
        set(value) != {"format", "source", "placement", "tasks", "sha256"}
        or value["format"] != TASK_PROMPT_CONTRACT
        or value["source"] != "d2.prompts"
        or value["placement"] != TASK_PROMPT_PLACEMENT
        or not isinstance(value["tasks"], dict)
        or set(value["tasks"]) != set(TASK_PROMPT_TEXTS)
    ):
        raise ValueError("task system prompt schema changed")
    for kind, text in TASK_PROMPT_TEXTS.items():
        task = value["tasks"][kind]
        if (
            not isinstance(task, dict)
            or set(task) != {"text", "formatted_text", "token_ids"}
            or task["text"] != text
            or task["formatted_text"] != format_system_prompt(text)
        ):
            raise ValueError(f"{kind} system prompt differs from Qwen D2")
        ids = task["token_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                type(token) is not int or not 0 <= token < THINKER_BASE_VOCAB_SIZE for token in ids
            )
        ):
            raise ValueError(f"{kind} system prompt token IDs are invalid")
        if tokenizer is not None and ids != list(encode_system_prompt(tokenizer, text)):
            raise ValueError(f"{kind} system prompt token IDs differ from the Omni2 tokenizer")
    if value["sha256"] != _digest({key: item for key, item in value.items() if key != "sha256"}):
        raise ValueError("task system prompt contract SHA256 changed")
    return value


def configure_task_prompts(model: Any, tokenizer: Any = None, *, contract=None) -> dict[str, Any]:
    if not getattr(model, "thinker_alignment", False):
        raise ValueError("task system prompts require the aligned serial Thinker")
    if contract is None:
        if tokenizer is None:
            raise ValueError("new task system prompts require the Omni2 tokenizer")
        contract = build_task_prompt_contract(tokenizer)
    model.task_prompt_contract = validate_task_prompt_contract(contract, tokenizer)
    return deepcopy(model.task_prompt_contract)


def disable_task_prompts(model: Any) -> None:
    model.task_prompt_contract = None


def task_prompt_ids(model: Any, task_kind: str | None) -> tuple[int, ...]:
    contract = getattr(model, "task_prompt_contract", None)
    if contract is None:
        return ()
    kind = normalize_task_kind(task_kind)
    return tuple(contract["tasks"][kind]["token_ids"])


def task_prompt_metadata(model: Any, task_kind: str) -> dict[str, Any]:
    kind = normalize_task_kind(task_kind)
    ids = task_prompt_ids(model, kind)
    result = {"enabled": bool(ids), "task_kind": kind, "token_count": len(ids)}
    if ids:
        result.update(deepcopy(model.task_prompt_contract["tasks"][kind]))
        result["contract_sha256"] = model.task_prompt_contract["sha256"]
    return result
