from __future__ import annotations

from typing import Any, Mapping


def clone_token_usage(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not usage:
        return None
    cloned = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "call_count": int(usage.get("call_count", 0)),
        "by_call_type": {},
    }
    for key, value in dict(usage.get("by_call_type", {})).items():
        cloned["by_call_type"][str(key)] = {
            "prompt_tokens": int(value.get("prompt_tokens", 0)),
            "completion_tokens": int(value.get("completion_tokens", 0)),
            "total_tokens": int(value.get("total_tokens", 0)),
            "call_count": int(value.get("call_count", 0)),
        }
    return cloned


def merge_token_usage_buckets(target: dict[str, Any], source: Mapping[str, Any] | None) -> None:
    if not source:
        return
    target["prompt_tokens"] += int(source.get("prompt_tokens", 0))
    target["completion_tokens"] += int(source.get("completion_tokens", 0))
    target["total_tokens"] += int(source.get("total_tokens", 0))
    target["call_count"] += int(source.get("call_count", 0))
    for key, value in dict(source.get("by_call_type", {})).items():
        bucket = target["by_call_type"].setdefault(
            str(key),
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            },
        )
        bucket["prompt_tokens"] += int(value.get("prompt_tokens", 0))
        bucket["completion_tokens"] += int(value.get("completion_tokens", 0))
        bucket["total_tokens"] += int(value.get("total_tokens", 0))
        bucket["call_count"] += int(value.get("call_count", 0))
