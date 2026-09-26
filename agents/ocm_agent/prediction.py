from __future__ import annotations

from typing import Any, Mapping, Sequence

from .json_utils import load_jsonish


def parse_action_prediction(
    raw: str,
    action: str,
    *,
    available_knowledge: Sequence[str] = (),
) -> dict[str, Any]:
    payload = load_jsonish(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("world-model prediction must be a JSON object")
    outcome = str(payload.get("outcome") or "").strip()
    valid = payload.get("valid")
    effect = str(payload.get("effect") or "").strip().lower()
    if not outcome:
        raise ValueError("prediction needs a concise outcome")
    if valid not in {True, False, None}:
        raise ValueError("prediction valid must be true, false, or null")
    if effect not in {"advance", "neutral", "harm", "complete", "unknown"}:
        raise ValueError(
            "prediction effect must be advance, neutral, harm, complete, or unknown"
        )
    known = set(available_knowledge)
    references = []
    values = payload.get("knowledge")
    for value in values if isinstance(values, list) else []:
        reference = str(value).strip().strip("`")
        if reference and reference in known and reference not in references:
            references.append(reference)
    return {
        "action": action,
        "outcome": outcome,
        "valid": valid,
        "effect": effect,
        "knowledge": references,
    }


def evaluate_observed_transition(
    prediction: Mapping[str, Any],
    *,
    reward: float | None,
    previous_reward: float | None,
    done: bool,
    success: bool | None,
    invalid_action: bool,
) -> dict[str, Any]:
    predicted_valid = prediction.get("valid")
    validity_correct = (
        None
        if predicted_valid is None
        else bool(predicted_valid) is (not bool(invalid_action))
    )
    reward_delta = None
    if reward is not None and previous_reward is not None:
        reward_delta = float(reward) - float(previous_reward)
    return {
        "actual": {
            "reward": reward,
            "reward_delta": reward_delta,
            "done": bool(done),
            "success": success,
            "invalid_action": bool(invalid_action),
        },
        "validity_correct": validity_correct,
    }
