from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


KNOWLEDGE_ROLES = (
    "object_knowledge.py models reusable environment entities, states, affordances, constraints, relations, "
    "and state changes.\n"
    "procedural_knowledge/*.py stores compact action-facing lessons that use object_knowledge types: workflows, "
    "rules, syntax constraints, failure patterns, decision heuristics, or recovery principles."
)

KNOWLEDGE_DESIGN_PRINCIPLES = (
    "Separate the world model from action lessons. Ground every update in observations, invalid-action feedback, "
    "reward, done, or success signals. Treat existing knowledge as provisional: preserve it when supported, refine "
    "it when incomplete, and replace it when contradicted. Successful episodes show what works; failed episodes "
    "mostly show preconditions, invalid syntax, missing observations, failure causes, and recovery. Predictions are "
    "commitments made before observing an action's outcome: use prediction-observation agreement as support and "
    "mismatch as a counterexample to the cited knowledge. Keep final knowledge compact and reusable."
)

REFLECTION_RULES = {
    "output": [
        "Return JSON only with keys: object_knowledge_updates, procedural_knowledge_updates, inspect_objects, inspect_procedural.",
        "Each planned update should be compact: target, operation, lesson, evidence, reason.",
        "operation is create, update, or replace.",
        "inspect_objects contains object top-level names or ClassName.method_name names to inspect.",
        "inspect_procedural contains procedural file stems without .py to inspect.",
    ],
    "selection": [
        "Plan only reusable knowledge supported by the episode.",
        "Use object updates for entities, states, affordances, constraints, relations, or mappings.",
        "Use procedural updates for workflows, rules, syntax constraints, failure patterns, heuristics, or recovery.",
        "Attribute an incorrect predicted environment effect to the cited object knowledge.",
        "Attribute an unsuitable decision whose predicted effect was accurate to the cited procedural knowledge.",
        "Skip one-offs, exact locations, task ids, raw trajectories, and full action scripts.",
    ],
    "content": [
        "Evidence should mention the relevant observation, invalid feedback, reward, done, or success signal.",
        "Use each step's prediction and realized next observation to identify supported or contradicted knowledge.",
        "When a prediction is contradicted, inspect and revise the object or procedural symbols cited in that step before creating unrelated knowledge.",
        "Treat semantic agreement as support even when prediction and observation use different wording.",
        "If existing knowledge is contradicted, plan replace rather than only adding new knowledge.",
        "For procedural updates, name the object_knowledge types or helpers the lesson should use.",
    ],
}

KNOWLEDGE_UPDATE_SCHEMA = (
    "Return JSON only with keys 'object_knowledge' and 'procedural_knowledge'. Each key maps to a list "
    "of objects with fields 'existing_name', 'name', and 'code'. For new knowledge, existing_name is empty. "
    "For replacement or rename, existing_name is the current object class name or procedural file stem. "
    "name is the resulting class name or file stem. code is the complete replacement code for that item."
)

KNOWLEDGE_UPDATE_RULES = {
    "object_knowledge": [
        "Write compact classes or top-level helpers for reusable environment entities, states, affordances, constraints, relations, and mappings.",
        "Model only what the episode evidence supports.",
        "Use simple defaults so deterministic checks can instantiate classes without live environment access.",
        "Never import object_knowledge from inside object_knowledge.py.",
        "Do not write task solutions, action scripts, or episode transcripts.",
    ],
    "procedural_knowledge": [
        "Each file represents one compact reusable interaction lesson.",
        "Import object_knowledge classes or helpers instead of redefining environment entities.",
        "Use short docstrings, small constants, typed records, and focused helpers.",
        "Action strings are allowed only as primitive syntax examples, not complete scripts or next-action APIs.",
        "Helpers should classify, explain, check applicability, or expose examples; they should not choose the next action.",
        "When replacing an existing file, remove claims contradicted by the episode.",
    ],
    "quality": [
        "Use retrieval-friendly names and brief docstrings.",
        "Keep detailed evidence in the reflection plan, not in long code comments or docstrings.",
        "Every procedural file must include a lightweight no-argument __verify__() with deterministic assertions.",
        "__verify__ should check imports/references and simple helper behavior.",
        "__verify__ must not call the live environment or external services.",
        "The candidate knowledge must compile and import successfully.",
    ],
}


def action_messages(
    *,
    benchmark_instruction: str,
    object_signatures: str,
    procedural_index: str,
    expanded_source: str,
    observation: str,
    trajectory: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an interactive environment agent with access to two learned Python resources. "
                "object_knowledge.py describes reusable environment object types, observable states, affordances, "
                "constraints, relations, and state changes. procedural_knowledge/ describes reusable interaction "
                "lessons: workflows, rules, failure patterns, insights, syntax constraints, decision heuristics, "
                "and recovery strategies that use those object types. These resources are for reading and reasoning, "
                "not for direct execution. Do not copy internal class, file, or function names as environment actions. "
                "Use signatures, procedural index entries, and any inspected code to choose the next real text action. "
                "Inspect relevant learned resources when you are uncertain about object affordances, action syntax, "
                "preconditions, success signals, failure causes, or recovery. Before acting, commit to one concise, "
                "falsifiable prediction of the immediate observable outcome and cite the exact learned symbols that "
                "support it. Output exactly four lines: a brief Thought line, a Prediction line, a Grounding line, "
                "and an Action line. Use Grounding: none when no learned symbol supports the prediction. Use internal "
                "inspect only to read learned resources, for example: "
                "Action: Inspect[object.Container, procedural.heating_substance]. Otherwise the Action line must "
                "be one primitive environment action. Use this format:\n"
                "Thought: <brief reasoning>\n"
                "Prediction: <immediate observable outcome if the action is executed>\n"
                "Grounding: <comma-separated object.* and procedural.* symbols, or none>\n"
                "Action: <one primitive environment action>"
            ),
        }
    ]
    if benchmark_instruction:
        messages.append({"role": "system", "content": "Benchmark instruction:\n" + benchmark_instruction})
    context = (
        "object_knowledge.py signatures:\n"
        + (object_signatures.strip() or "(empty)")
        + "\n\nprocedural_knowledge/index.py:\n```python\n"
        + (procedural_index.strip() or "PROCEDURAL_KNOWLEDGE = {}")
        + "\n```"
    )
    if expanded_source.strip():
        context += "\n\nInspected learned code available in this episode:\n```python\n" + expanded_source.strip() + "\n```"
    messages.append({"role": "system", "content": context})
    for msg in list(trajectory or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != observation:
        messages.append({"role": "user", "content": observation})
    return messages


def plain_action_messages(
    *,
    benchmark_instruction: str,
    observation: str,
    trajectory: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an interactive environment agent. "
                "Follow benchmark guidance if provided. "
                "Before acting, commit to one concise, falsifiable prediction of the immediate observable outcome. "
                "Output exactly four lines in this format:\n"
                "Thought: <brief reasoning>\n"
                "Prediction: <immediate observable outcome if the action is executed>\n"
                "Grounding: none\n"
                "Action: <one primitive environment action>\n"
                "Do not output markdown or extra text."
            ),
        }
    ]
    if benchmark_instruction:
        messages.append({"role": "system", "content": "Benchmark instruction:\n" + benchmark_instruction})
    for msg in list(trajectory or []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != observation:
        messages.append({"role": "user", "content": observation})
    return messages


def reflection_messages(
    *,
    benchmark_name: str,
    object_signatures: str,
    procedural_index: str,
    steps: Sequence[Mapping[str, Any]],
    result_meta: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Analyze one completed environment episode and write a concise update plan for two learned Python "
                "resources that will be read during future decisions.\n\n"
                + KNOWLEDGE_ROLES
                + "\n\nDesign principles:\n"
                + KNOWLEDGE_DESIGN_PRINCIPLES
                + "\n\nReturn JSON only. Do not return Markdown."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "benchmark": benchmark_name,
                    "result": dict(result_meta),
                    "object_knowledge_signatures": object_signatures,
                    "procedural_knowledge_index_py": procedural_index,
                    "episode_steps": list(steps),
                    "rules": REFLECTION_RULES,
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
    inspected_source: str,
    steps: Sequence[Mapping[str, Any]],
    result_meta: Mapping[str, Any],
    previous_error: str | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Implement updates from a completed environment episode for two learned Python resources that "
                "future decisions will read.\n\n"
                + KNOWLEDGE_ROLES
                + "\n\nDesign principles:\n"
                + KNOWLEDGE_DESIGN_PRINCIPLES
                + "\n\n"
                + KNOWLEDGE_UPDATE_SCHEMA
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "object_knowledge_signatures": object_signatures,
                    "procedural_knowledge_index_py": procedural_index,
                    "update_plan": update_plan,
                    "inspected_source": inspected_source,
                    "result": dict(result_meta),
                    "episode_steps": list(steps),
                    "previous_error": previous_error,
                    "rules": KNOWLEDGE_UPDATE_RULES,
                },
                ensure_ascii=True,
                indent=2,
            ),
        },
    ]
