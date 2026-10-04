from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


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
) -> list[dict[str, str]]:
    system_parts = [
        (
            "You are an interactive environment agent. Follow benchmark guidance if provided. "
            "Choose the next primitive action that best advances the task. Use learned OCM code as a fallible prior. "
            "Treat environment observations in the trajectory as ground truth. If foresight is supplied, it is a fallible "
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
                    "Foresight for the proposed action:\n"
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
    system_parts = [
        (
            "You are the agent's explicit world model. "
            "Before the proposed action is executed, predict two connected aspects: transition, the immediate "
            "externally observable change or information returned; and progress, what that expected result "
            "contributes to the current goal, including prerequisites or lack of progress. Use the observed "
            "trajectory as ground truth and the learned OCM code as fallible prior knowledge. "
            "Each prediction is a hypothesis, not observed evidence. Express uncertainty when the evidence "
            "does not determine the outcome or its contribution. Each component cites only exact top-level "
            "knowledge ids that support that specific judgment, or an empty list when none do. Either component "
            "may use object or procedural knowledge; do not force a citation or a category assignment. "
            "Keep each prediction to one short sentence. Return JSON only: "
            '{"transition":{"prediction":"...","knowledge":[]},'
            '"progress":{"prediction":"...","knowledge":[]}}'
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
    messages = _conversation_messages(system_parts, observation, trajectory)
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
    knowledge_prediction_steps: Mapping[str, list[dict[str, Any]]] | None = None,
    referenced_source: Mapping[str, str | None] | None = None,
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Propose the minimal knowledge changes supported by this completed episode. "
                + KNOWLEDGE_ROLES
                + " A prediction is a hypothesis; observations and environment feedback are evidence. Compare transition "
                "predictions with observed changes, and progress predictions with the task goal and subsequent trajectory, "
                "not immediate reward alone. knowledge_prediction_steps links predictions to their supporting knowledge; "
                "referenced_source provides that knowledge. The catalog also permits new or uncited targets. "
                "For each update, lesson states only what to add, retract, or change in the existing knowledge, "
                "not a restatement of unchanged facts or workflows. evidence identifies the observations supporting "
                "that change. A contradicted claim may need revision without establishing an alternative explanation. "
                "A separate writer will apply these changes to the existing source. "
                "Return JSON only with keys "
                "object_knowledge_updates and procedural_knowledge_updates. Each value is a list with at most one "
                "item containing target, operation (create, update, or replace), lesson, and evidence. Return empty "
                "lists when no reusable change is supported. Copy existing target names exactly from the catalog. "
                "New object targets are PascalCase entity categories or mechanisms, not individual instances or locations; "
                "procedural targets are snake_case file stems, not function paths."
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
                    "knowledge_prediction_steps": dict(knowledge_prediction_steps or {}),
                    "referenced_source": dict(referenced_source or {}),
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
        "Each planned object target must exist as a top-level definition with its exact target name in the resulting "
        "object_knowledge.py; helper functions or constants do not substitute for that definition. "
        "Use minimal hunks that preserve unrelated code; never reproduce a complete existing file. For a new "
        "procedural target, add its file. Keep knowledge compact, general, and readable by a language model. Implement "
        "only the reusable claims in the plan without adding unsupported claims. Object code represents entities, "
        "states, relations, and transitions; procedural code represents action-facing workflows and preconditions and "
        "may import object_knowledge. Prefer compact constants and small query helpers. The patched code must compile.\n"
        "Patch syntax (illustrative paths and code, not requested edits):\n"
        "*** Begin Patch\n*** Update File: existing.py\n@@\n-OLD_LINE\n+NEW_LINE\n"
        "*** Add File: new.py\n+FIRST_LINE\n+\n+NEXT_LINE\n*** End Patch\n"
        "Update existing files, including empty files, with Update File and @@ hunks; copy context/removal lines "
        "exactly from editable_source. For an empty existing file use @@ followed by + lines. "
        "Add nonexistent files with Add File and + lines, without @@. Context lines start with a space; "
        "removed lines with '-'; inserted lines with '+'. Include all planned files in one patch."
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
    )
    traces = [
        {field: step[field] for field in fields if field in step}
        for step in steps
    ]
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
