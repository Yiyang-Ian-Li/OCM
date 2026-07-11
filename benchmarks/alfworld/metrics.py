from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_alfworld(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_invalid_actions = 0
    total_steps = 0
    for record in records:
        task_type = record["task"].get("task_type", "unknown")
        by_task_type[task_type].append(record)
        result = record["result"]["meta"]
        total_invalid_actions += result.get("invalid_actions", 0)
        total_steps += result["steps"]

    task_type_metrics = {}
    for task_type, items in sorted(by_task_type.items()):
        successes = [item["result"]["meta"]["success"] for item in items]
        rewards = [item["result"]["meta"]["reward"] for item in items if item["result"]["meta"]["reward"] is not None]
        task_type_metrics[task_type] = {
            "count": len(items),
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "average_reward": sum(rewards) / len(rewards) if rewards else None,
        }

    return {
        "average_steps": total_steps / len(records) if records else 0.0,
        "invalid_action_rate": total_invalid_actions / total_steps if total_steps else 0.0,
        "task_type_metrics": task_type_metrics,
    }
