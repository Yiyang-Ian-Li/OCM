from __future__ import annotations

from collections import OrderedDict
from typing import Any

from benchmarks import BENCHMARKS

from .usage import merge_token_usage_buckets


def summarize_records(
    benchmark_name: str,
    records: list[dict[str, Any]],
    *,
    num_trials: int = 1,
) -> dict[str, Any]:
    if int(num_trials) <= 1:
        summary = _summarize_flat(benchmark_name, records)
        summary["agent_metrics"] = summarize_agent_mechanism(records)
        return summary

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        metadata = record["task"]
        base_task_id = str(metadata.get("base_task_id") or metadata["task_id"])
        groups.setdefault(base_task_id, []).append(record)

    best_records = [
        max(
            group,
            key=lambda record: (
                bool(record["result"]["meta"].get("success")),
                _score_reward(
                    benchmark_name,
                    record["result"]["meta"].get("reward"),
                ),
            ),
        )
        for group in groups.values()
    ]
    summary = _summarize_flat(benchmark_name, best_records)
    summary["num_trial_records"] = len(records)
    summary["num_trials_per_task"] = int(num_trials)
    summary["success_at_1"] = (
        sum(bool(group[0]["result"]["meta"].get("success")) for group in groups.values())
        / len(groups)
        if groups
        else 0.0
    )
    summary["cumulative_by_trial"] = _cumulative_metrics(
        benchmark_name,
        groups,
        int(num_trials),
    )
    summary["invalid_actions"] = sum(
        int(record["result"]["meta"].get("invalid_actions", 0)) for record in records
    )
    summary["token_usage"] = _token_summary(records, divisor=max(1, len(groups)))
    summary["agent_metrics"] = summarize_agent_mechanism(records)
    return summary


def summarize_agent_mechanism(records: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    updates = []
    for record in records:
        artifact = record.get("result", {}).get("agent_artifact") or {}
        decisions.extend(artifact.get("decision_log") or [])
        if artifact.get("online_update"):
            updates.append(artifact["online_update"])
    predicted = [decision for decision in decisions if decision.get("prediction")]
    memory_available = [
        decision for decision in predicted if decision.get("knowledge_available")
    ]
    memory_referenced = [
        decision for decision in memory_available if decision.get("knowledge_references")
    ]
    validity_checked = [
        decision for decision in predicted if decision.get("validity_correct") is not None
    ]
    revised = [
        decision for decision in predicted
        if decision.get("action_changed_by_foresight")
    ]
    return {
        "decisions": len(decisions),
        "predictions": len(predicted),
        "prediction_coverage": len(predicted) / max(1, len(decisions)),
        "memory_available_decisions": len(memory_available),
        "memory_referenced_decisions": len(memory_referenced),
        "memory_reference_rate": len(memory_referenced) / max(1, len(memory_available)),
        "foresight_revisions": len(revised),
        "foresight_revision_rate": len(revised) / max(1, len(predicted)),
        "post_revision_invalid_actions": sum(
            bool((decision.get("actual") or {}).get("invalid_action"))
            for decision in revised
        ),
        "validity_predictions_checked": len(validity_checked),
        "validity_prediction_accuracy": (
            sum(bool(decision.get("validity_correct")) for decision in validity_checked)
            / max(1, len(validity_checked))
        ),
        "planned_knowledge_updates": sum(
            int(update.get("planned_updates", 0)) for update in updates
        ),
        "written_knowledge_updates": sum(
            len((update.get("written_knowledge") or {}).get(kind, []))
            for update in updates
            for kind in ("object", "procedural")
        ),
    }


def _summarize_flat(
    benchmark_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metas = [record["result"]["meta"] for record in records]
    rewards = [
        _score_reward(benchmark_name, meta["reward"])
        for meta in metas
        if meta["reward"] is not None
    ]
    successes = [bool(meta["success"]) for meta in metas]
    step_counts = [int(meta["steps"]) for meta in metas]
    invalid_actions = [int(meta.get("invalid_actions", 0)) for meta in metas]
    terminate_reasons: dict[str, int] = {}
    for meta in metas:
        reason = meta["terminate_reason"] or "unknown"
        terminate_reasons[reason] = terminate_reasons.get(reason, 0) + 1
    return {
        "num_tasks": len(records),
        "average_reward": sum(rewards) / len(rewards) if rewards else None,
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "average_steps": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "invalid_actions": sum(invalid_actions),
        "terminate_reasons": terminate_reasons,
        "token_usage": _token_summary(records, divisor=max(1, len(records))),
        "benchmark_metrics": BENCHMARKS[benchmark_name]["summarize"](records),
    }


def _cumulative_metrics(
    benchmark_name: str,
    groups: OrderedDict[str, list[dict[str, Any]]],
    num_trials: int,
) -> list[dict[str, Any]]:
    metrics = []
    for limit in range(1, num_trials + 1):
        successes = []
        rewards = []
        for group in groups.values():
            prefix = group[:limit]
            successes.append(any(record["result"]["meta"].get("success") for record in prefix))
            rewards.append(
                max(
                    _score_reward(benchmark_name, record["result"]["meta"].get("reward"))
                    for record in prefix
                )
            )
        metrics.append(
            {
                "trial": limit,
                "success_rate": sum(successes) / len(successes) if successes else 0.0,
                "average_reward": sum(rewards) / len(rewards) if rewards else None,
            }
        )
    return metrics


def _token_summary(records: list[dict[str, Any]], *, divisor: int) -> dict[str, Any]:
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
        "by_call_type": {},
    }
    for record in records:
        merge_token_usage_buckets(total, record["result"]["meta"].get("token_usage"))
    return {
        **total,
        "average_total_tokens_per_task": total["total_tokens"] / max(1, divisor),
    }


def _score_reward(benchmark_name: str, reward: float | None) -> float:
    value = float(reward or 0.0)
    return max(0.0, value) if benchmark_name == "sciworld" else value
