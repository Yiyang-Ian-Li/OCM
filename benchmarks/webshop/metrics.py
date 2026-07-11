from __future__ import annotations

from typing import Any


def summarize_webshop(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["result"]["meta"].get("reward") or 0.0) for record in records]
    successes = [bool(record["result"]["meta"].get("success")) for record in records]
    total_steps = sum(int(record["result"]["meta"].get("steps", 0)) for record in records)
    invalid_actions = sum(int(record["result"]["meta"].get("invalid_actions", 0)) for record in records)
    return {
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "average_steps": total_steps / len(records) if records else 0.0,
        "invalid_action_rate": invalid_actions / total_steps if total_steps else 0.0,
    }
