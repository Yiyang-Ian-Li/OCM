from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from agents.common import extract_action_candidate


KNOWLEDGE_ROLES = (
    "object_knowledge.py stores reusable facts and mechanisms about environment entities, states, "
    "affordances, relations, and transitions. procedural_knowledge/*.py stores reusable action-facing "
    "lessons such as workflows, preconditions, syntax, failure modes, and recovery rules."
)


def policy_messages(
    *,
    benchmark_instruction: str,
    benchmark_examples: str,
    knowledge_context: str,
    observation: str,
    trajectory: Sequence[dict[str, str]] | None,
    foresight: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    system_parts = [
        (
            "You are an interactive environment agent. Follow benchmark guidance if provided. "
            "Choose the next primitive action that best advances the task. Use learned OCM code as a fallible prior. "
            "Prior predictions appear immediately before actions in the trajectory; treat the following environment "
            "observation as ground truth and adapt when they disagree. If foresight is supplied, it is a fallible "
            "prediction for a proposed action: decide whether to keep or revise that action. Output exactly two lines:\n"
            "Thought: <brief reasoning>\n"
            "Action: <one primitive environment action>"
        )
    ]
    if benchmark_instruction:
        system_parts.append("Benchmark instruction:\n" + benchmark_instruction)
    if benchmark_examples.strip():
        system_parts.append("Benchmark demonstration:\n" + benchmark_examples.strip())
    system_parts.append(
        "Learned OCM code:\n"
        "```python\n"
        + (knowledge_context.strip() or "# empty")
        + "\n```"
    )
    messages = _conversation_messages(system_parts, observation, trajectory)
    if foresight:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Proposed action and reasoning:\n"
                    + json.dumps(dict(draft or {}), ensure_ascii=True)
                    + "\nForesight for the proposed action:\n"
                    + json.dumps(dict(foresight), ensure_ascii=True)
                    + "\nReturn the action you now choose. It may be the same action or a revision."
                ),
            }
        )
    return messages


def prediction_messages(
    *,
    benchmark_instruction: str,
    knowledge_context: str,
    observation: str,
    trajectory: Sequence[dict[str, str]] | None,
    action: str,
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    observed_trajectory = [
        {"role": message["role"], "content": (
            "Action: " + extract_action_candidate(message["content"])
            if message["role"] == "assistant" else message["content"]
        )}
        for message in _compact_trajectory(trajectory)
    ]
    system_parts = [
        (
            "You are the agent's explicit world model. "
            "Before the proposed action is executed, predict its immediate externally observable outcome and whether it advances the "
            "current task. Use the observed trajectory as ground truth and the learned OCM code as fallible prior "
            "knowledge. The prediction is a hypothesis, not observed evidence. valid is true, false, or null when "
            "validity is genuinely unknown. effect is advance, neutral, harm, complete, or unknown. "
            "knowledge lists only exact top-level ids visibly present in the supplied OCM code that materially "
            "support the prediction; otherwise use an empty list. A cited knowledge item's code must logically entail "
            "the predicted outcome in the observed state; never cite a relevant-looking item while predicting the "
            "opposite of its rule. Keep outcome to one short sentence. Return JSON only: "
            '{"outcome":"...","valid":true|false|null,'
            '"effect":"advance|neutral|harm|complete|unknown",'
            '"knowledge":["object.Name"|"procedural.name"]}'
        )
    ]
    if benchmark_instruction:
        system_parts.append("Benchmark instruction:\n" + benchmark_instruction)
    system_parts.append(
        "Learned OCM code:\n```python\n"
        + (knowledge_context.strip() or "# empty")
        + "\n```"
    )
    if previous_error:
        system_parts.append("The previous output was invalid: " + previous_error)
    messages = _conversation_messages(system_parts, observation, observed_trajectory)
    messages.append(
        {
            "role": "user",
            "content": (
                "Predict the immediate result of this proposed action:\n"
                + json.dumps({"action": action}, ensure_ascii=True)
            ),
        }
    )
    return messages


def reflection_messages(
    *,
    benchmark_name: str,
    task_context: str,
    object_signatures: str,
    procedural_index: str,
    steps: Sequence[Mapping[str, Any]],
    result_meta: Mapping[str, Any],
    previous_error: str | None = None,
    knowledge_context: str = "",
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Reflect on one completed episode to maintain an object-centric Python world model. "
                + KNOWLEDGE_ROLES
                + " Compare each pre-action prediction with the resulting environment transition. A prediction is a "
                "hypothesis; observations, reward, success, and invalid-action feedback are evidence. Use their residual "
                "to add, revise, or leave unchanged compact reusable knowledge. New evidence may update knowledge the "
                "prediction did not cite, and uncertain or conflicting evidence may warrant no update. Prefer the smallest "
                "object property, relation, transition, or procedural precondition that explains an observed difference. "
                "The task context defines what reward and success mean but is not itself environment evidence. Return JSON "
                "only with keys "
                "object_knowledge_updates and procedural_knowledge_updates. Each value is a list with at most one "
                "item containing target, operation (create, update, or replace), lesson, and evidence. Return empty "
                "lists when the episode adds nothing reusable. An object target is a PascalCase semantic entity "
                "category or mechanism (for example Container, Animal, or Material), never a room, location, named "
                "instance, or method. Store observed relations among named types inside their shared category rather "
                "than copying the scene inventory. A procedural target is exactly a procedural file stem from the "
                "catalog, or a short snake_case stem for a new file; never append a function name to it. For an "
                "existing item, copy its top-level catalog id exactly."
                " Read the existing implementation before proposing a revision. Evidence must distinguish the "
                "old claim from the proposed one: an observed state change supports that transition, while "
                "task completion requires observed success. A failed episode alone does not establish a new "
                "precondition. Preserve the conditions under which a lesson was observed; task-specific values "
                "are parameters, not universal constants. Revise contradicted claims rather than appending exceptions."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "benchmark": benchmark_name,
                    "task_context": task_context,
                    "result": dict(result_meta),
                    "object_knowledge_catalog": object_signatures,
                    "procedural_knowledge_catalog": procedural_index,
                    "existing_knowledge": knowledge_context,
                    "executed_prediction_traces": _reflection_evidence(steps),
                    "previous_error": previous_error,
                },
                ensure_ascii=True,
                indent=2,
            ),
        },
    ]


def knowledge_update_messages(
    *,
    object_signatures: str,
    procedural_index: str,
    update_plan: Mapping[str, Any],
    editable_source: Mapping[str, str | None],
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    system = (
        "Implement exactly the validated update plan for an object-centric Python world model. "
        + KNOWLEDGE_ROLES
        + " Return only one apply_patch block, from *** Begin Patch through *** End Patch. Patch exactly the files "
        "named by the plan: object_knowledge.py for object targets and procedural_knowledge/<target>.py for procedural "
        "targets. editable_source maps paths to exact file contents; null means the file does not exist. "
        "Use minimal hunks that preserve unrelated code; never reproduce a complete existing file. For a new "
        "procedural target, add its file. Keep knowledge compact, general, and readable by a language model. Implement "
        "only the reusable claims in the plan without adding unsupported claims. Object code represents entities, "
        "states, relations, and transitions; procedural code represents action-facing workflows and preconditions and "
        "may import object_knowledge. Prefer compact constants and small query helpers. The patched code must compile."
    )
    if previous_error:
        system += " The previous implementation was rejected: " + previous_error
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "object_knowledge_catalog": object_signatures,
                    "procedural_knowledge_catalog": procedural_index,
                    "update_plan": dict(update_plan),
                    "editable_source": editable_source,
                },
                ensure_ascii=True,
                indent=2,
            ),
        },
    ]


def _conversation_messages(
    system_parts: Sequence[str],
    observation: str,
    trajectory: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
    for message in _compact_trajectory(trajectory):
        messages.append(message)
    if (
        not messages
        or messages[-1].get("role") != "user"
        or messages[-1].get("content") != observation
    ):
        messages.append({"role": "user", "content": observation})
    return messages


def _reflection_evidence(
    steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "step",
        "prediction",
        "action",
        "next_observation",
        "actual",
        "validity_correct",
    )
    traces = [{field: step[field] for field in fields if field in step} for step in steps]
    if traces and "observation" in steps[0]:
        traces[0]["observation"] = steps[0]["observation"]
    return traces


def _compact_trajectory(
    trajectory: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in list(trajectory or [])
        if isinstance(message, dict)
        and message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
