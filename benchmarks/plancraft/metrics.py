from __future__ import annotations

from collections import defaultdict
from typing import Any


def _group_success_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["task"].get(key, "unknown"))].append(record)

    metrics = {}
    for group, items in sorted(groups.items()):
        metas = [item["result"]["meta"] for item in items]
        successes = [meta["success"] for meta in metas]
        rewards = [meta["reward"] for meta in metas if meta["reward"] is not None]
        metrics[group] = {
            "count": len(items),
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "average_reward": sum(rewards) / len(rewards) if rewards else None,
            "average_steps": sum(meta["steps"] for meta in metas) / len(metas) if metas else 0.0,
        }
    return metrics


def summarize_plancraft(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = 0
    total_invalid_actions = 0
    impossible_tasks = 0
    impossible_successes = 0

    for record in records:
        result = record["result"]["meta"]
        total_steps += result["steps"]
        total_invalid_actions += result.get("invalid_actions", 0)
        if record["task"].get("impossible"):
            impossible_tasks += 1
            impossible_successes += int(bool(result["success"]))

    return {
        "average_steps": total_steps / len(records) if records else 0.0,
        "invalid_action_rate": total_invalid_actions / total_steps if total_steps else 0.0,
        "impossible_success_rate": impossible_successes / impossible_tasks if impossible_tasks else None,
        "complexity_split_metrics": _group_success_metrics(records, "complexity_split"),
        "recipe_type_metrics": _group_success_metrics(records, "recipe_type"),
    }
