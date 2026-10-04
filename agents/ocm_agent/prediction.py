from __future__ import annotations

from typing import Any, Mapping, Sequence

from .json_utils import load_jsonish


PREDICTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                component: {
                    "type": "object",
                    "properties": {
                        "prediction": {"type": "string", "minLength": 1},
                        "knowledge": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["prediction", "knowledge"],
                    "additionalProperties": False,
                }
                for component in ("transition", "progress")
            },
            "required": ["transition", "progress"],
            "additionalProperties": False,
        },
    },
}


def parse_action_prediction(
    raw: str,
    action: str,
    *,
    available_knowledge: Sequence[str] = (),
) -> dict[str, Any]:
    payload = load_jsonish(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("world-model prediction must be a JSON object")
    known = set(available_knowledge)
    result = {"action": action}
    for component in ("transition", "progress"):
        value = payload.get(component)
        if not isinstance(value, Mapping) or not isinstance(value.get("prediction"), str) or not value["prediction"].strip():
            raise ValueError(f"{component} needs a nonempty prediction string")
        values = value.get("knowledge", [])
        if not isinstance(values, list):
            raise ValueError(f"{component}.knowledge must be a list")
        references = list(dict.fromkeys(
            str(item).strip().strip("`") for item in values
            if str(item).strip().strip("`") in known
        ))
        result[component] = {"prediction": value["prediction"].strip(), "knowledge": references}
    return result


def evaluate_observed_transition(
    *,
    reward: float | None,
    previous_reward: float | None,
    done: bool,
    success: bool | None,
    invalid_action: bool,
) -> dict[str, Any]:
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
    }
