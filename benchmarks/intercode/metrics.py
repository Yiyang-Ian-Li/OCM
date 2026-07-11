from __future__ import annotations

from collections import defaultdict
from typing import Any


def summarize_intercode(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_steps = 0
    invalid_actions = 0
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record["task"].get("split", "unknown"))].append(record)
        meta = record["result"]["meta"]
        total_steps += int(meta.get("steps", 0))
        invalid_actions += int(meta.get("invalid_actions", 0))
    return {
        "average_steps": total_steps / len(records) if records else 0.0,
        "invalid_action_rate": invalid_actions / total_steps if total_steps else 0.0,
        "split_metrics": {name: _group_metrics(items) for name, items in sorted(by_split.items())},
    }


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["result"]["meta"].get("reward") or 0.0) for record in records]
    successes = [bool(record["result"]["meta"].get("success")) for record in records]
    return {
        "count": len(records),
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
    }
