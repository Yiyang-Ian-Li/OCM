from __future__ import annotations

from typing import Any

from benchmarks import BENCHMARKS

from .usage import merge_token_usage_buckets


def summarize_records(benchmark_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [record["result"]["meta"] for record in records]
    rewards = [_score_reward(benchmark_name, meta["reward"]) for meta in metas if meta["reward"] is not None]
    successes = [meta["success"] for meta in metas]
    step_counts = [meta["steps"] for meta in metas]
    invalid_actions = [meta.get("invalid_actions", 0) for meta in metas]
    total_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
        "by_call_type": {},
    }
    terminate_reasons: dict[str, int] = {}
    for meta in metas:
        reason = meta["terminate_reason"] or "unknown"
        terminate_reasons[reason] = terminate_reasons.get(reason, 0) + 1
        merge_token_usage_buckets(total_token_usage, meta.get("token_usage"))
    return {
        "num_tasks": len(records),
        "average_reward": sum(rewards) / len(rewards) if rewards else None,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "average_steps": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "invalid_actions": sum(invalid_actions),
        "terminate_reasons": terminate_reasons,
        "token_usage": {
            **total_token_usage,
            "average_total_tokens_per_task": (total_token_usage["total_tokens"] / len(records)) if records else 0.0,
        },
        "benchmark_metrics": BENCHMARKS[benchmark_name]["summarize"](records),
    }


def _score_reward(benchmark_name: str, reward: float) -> float:
    if benchmark_name == "sciworld":
        return max(0.0, float(reward))
    return reward
