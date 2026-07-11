from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_sciworld(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_invalid_actions = 0
    total_steps = 0
    for record in records:
        sub_task_name = record["task"].get("sub_task_name", "unknown")
        by_subtask[sub_task_name].append(record)
        result = record["result"]["meta"]
        total_invalid_actions += result.get("invalid_actions", 0)
        total_steps += result["steps"]

    subtask_metrics = {}
    for subtask, items in sorted(by_subtask.items()):
        metas = [item["result"]["meta"] for item in items]
        rewards = [_score_reward(meta["reward"]) for meta in metas if meta["reward"] is not None]
        successes = [meta["success"] for meta in metas]
        subtask_metrics[subtask] = {
            "count": len(items),
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "average_reward": sum(rewards) / len(rewards) if rewards else None,
            "average_steps": sum(meta["steps"] for meta in metas) / len(metas) if metas else 0.0,
            "variation_indices": sorted({item["task"].get("variation_idx", -1) for item in items}),
        }

    rewards = [_score_reward(record["result"]["meta"]["reward"]) for record in records if record["result"]["meta"]["reward"] is not None]
    return {
        "average_steps": total_steps / len(records) if records else 0.0,
        "invalid_action_rate": total_invalid_actions / total_steps if total_steps else 0.0,
        "max_reward": max(rewards) if rewards else None,
        "min_reward": min(rewards) if rewards else None,
        "subtask_metrics": subtask_metrics,
    }


def _score_reward(reward: float) -> float:
    return max(0.0, float(reward))
