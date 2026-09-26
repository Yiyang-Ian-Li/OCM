from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.base import Agent
from agents.common import (
    extract_action_candidate,
    load_benchmark_examples,
    load_benchmark_instruction,
)
from agents.llm import OpenAIChatClient

from .knowledge import (
    KnowledgeStore,
    parse_reflection_plan,
    validate_knowledge_updates,
    validate_reflection_plan,
)
from .prediction import (
    evaluate_observed_transition,
    parse_action_prediction,
)
from .prompting import (
    knowledge_update_messages,
    policy_messages,
    prediction_messages,
    reflection_messages,
)


MAX_COMPLETION_TOKENS = {
    "ocm_policy": 256,
    "ocm_predict": 256,
    "ocm_reflect": 1024,
    "ocm_knowledge_update": 2048,
}


class OCMAgent(Agent):
    """Predictive controller that validates and updates OCM code memory."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.model_name = str(self.config["model_name"])
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Missing API key for selected provider. Set it in .env or pass --api-key.")
        self.temperature = float(self.config.get("temperature", 0.0))
        self.top_p = self.config.get("top_p")
        self.seed = self.config.get("seed")
        self.timeout = int(self.config.get("timeout", 120))
        self.output_dir = Path(self.config["output_dir"])
        self.max_update_attempts = int(self.config.get("ocm_codegen_max_attempts", 3))
        self.knowledge = KnowledgeStore.create(
            self.output_dir,
            load_path=self.config.get("ocm_knowledge_load_path"),
        )
        self.llm = OpenAIChatClient(
            api_key=self.api_key,
            base_url=str(self.config.get("base_url")),
            timeout=self.timeout,
            usage_callback=self._record_token_usage,
        )
        task_dir = self.output_dir / "ocm_agent" / "tasks"
        self._episode_index = _max_index(task_dir, "episode_", ".json")
        self._prompt_counter = _max_index(task_dir / "prompts", "", ".json")
        self._benchmark_name = ""
        self._task_log_path: Path | None = None
        self._task_log: dict[str, Any] = {}
        self._decision_log: list[dict[str, Any]] = []
        self._previous_reward: float | None = 0.0
        self.last_raw_output = ""

    def _chat(self, messages: list[dict[str, str]], *, call_type: str) -> str:
        response = self.llm.create_chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop=self.stop_sequences or None,
            max_tokens=MAX_COMPLETION_TOKENS.get(call_type),
            call_type=call_type,
        )
        return response.content or ""

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _flush_task_log(self) -> None:
        if self._task_log_path is not None:
            self._write_json(self._task_log_path, self._task_log)

    def _log_prompt(
        self,
        kind: str,
        messages: list[dict[str, str]],
        raw_output: str,
        parsed: Any,
    ) -> None:
        self._prompt_counter += 1
        prompt_dir = self.output_dir / "ocm_agent" / "tasks" / "prompts"
        self._write_json(
            prompt_dir / f"{self._prompt_counter:06d}_{kind}.json",
            {
                "kind": kind,
                "messages": messages,
                "raw_output": raw_output,
                "parsed": parsed,
            },
        )

    def on_episode_start(
        self,
        benchmark_name: str | None = None,
        observation: str | None = None,
    ) -> None:
        self.reset_episode_token_usage()
        self._benchmark_name = str(benchmark_name or "")
        self._episode_index += 1
        self._decision_log = []
        self._previous_reward = 0.0
        episode_id = f"episode_{self._episode_index:06d}"
        self._task_log_path = (
            self.output_dir / "ocm_agent" / "tasks" / f"{episode_id}.json"
        )
        self._task_log = {
            "benchmark": self._benchmark_name,
            "knowledge_root": str(self.knowledge.root),
            "initial_observation": observation or "",
            "steps": [],
            "result_meta": None,
            "online_update": None,
        }
        self._flush_task_log()

    def act(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None = None,
    ) -> str:
        usage_before = self.episode_token_usage()
        instruction = load_benchmark_instruction(self._benchmark_name)
        examples = load_benchmark_examples(self._benchmark_name)
        model_context = self.knowledge.model_context()

        available_knowledge = self.knowledge.knowledge_ids()

        def decide(
            foresight: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            policy_prompt = policy_messages(
                benchmark_instruction=instruction,
                benchmark_examples=examples,
                knowledge_context=model_context,
                observation=observation,
                trajectory=trajectory,
                foresight=foresight,
            )
            policy_raw, policy, policy_attempts = self._call_with_format_retry(
                policy_prompt,
                call_type="ocm_policy",
                parser=_parse_policy_output,
            )
            self._log_prompt(
                "ocm_policy",
                policy_prompt,
                policy_raw,
                {"value": policy, "attempts": policy_attempts},
            )
            return {
                "policy": policy,
                "policy_raw_output": policy_raw,
            }

        def predict(action: str) -> dict[str, Any]:
            prediction_prompt = prediction_messages(
                benchmark_instruction=instruction,
                knowledge_context=model_context,
                observation=observation,
                trajectory=trajectory,
                action=action,
            )
            prediction_raw, prediction, prediction_attempts = self._call_with_format_retry(
                prediction_prompt,
                call_type="ocm_predict",
                parser=lambda raw: parse_action_prediction(
                    raw,
                    action,
                    available_knowledge=available_knowledge,
                ),
            )
            self._log_prompt(
                "ocm_predict",
                prediction_prompt,
                prediction_raw,
                {"value": prediction, "attempts": prediction_attempts},
            )
            return {
                "prediction": prediction,
                "prediction_raw_output": prediction_raw,
            }

        draft = decide()
        draft_action = str(draft["policy"]["action"])
        draft_prediction = predict(draft_action)
        foresight = dict(draft_prediction["prediction"])
        revised = decide(foresight=foresight)
        policy = revised["policy"]
        policy_raw = revised["policy_raw_output"]
        action = str(policy["action"])
        if _same_action(action, draft_action):
            final_prediction = draft_prediction
        else:
            final_prediction = predict(action)
        prediction = final_prediction["prediction"]
        prediction_raw = final_prediction["prediction_raw_output"]
        self.last_raw_output = (
            "Thought: "
            + str(policy["thought"]).strip()
            + "\nPrediction: "
            + str(prediction["outcome"]).strip()
            + "\nAction: "
            + action
        )
        step_usage = _usage_delta(usage_before, self.episode_token_usage())
        if step_usage is not None:
            self._last_token_usage = {
                key: int(step_usage[key])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        self._decision_log.append(
            {
                "step": len(self._decision_log),
                "observation": observation,
                "policy_thought": policy["thought"],
                "policy_raw_output": policy_raw,
                "draft_thought": draft["policy"]["thought"],
                "draft_action": draft_action,
                "draft_raw_output": draft["policy_raw_output"],
                "draft_prediction": draft_prediction["prediction"],
                "foresight": foresight,
                "prediction": prediction,
                "prediction_raw_output": prediction_raw,
                "parsed_action": action,
                "action_changed_by_foresight": not _same_action(action, draft_action),
                "knowledge_available": bool(available_knowledge),
                "knowledge_references": list(prediction.get("knowledge") or []),
                "knowledge_context_chars": len(model_context),
                "token_usage": step_usage,
            }
        )
        return action

    def _call_with_format_retry(
        self,
        messages: list[dict[str, str]],
        *,
        call_type: str,
        parser,
    ) -> tuple[str, Any, list[dict[str, str | None]]]:
        working = list(messages)
        raw = ""
        error: str | None = None
        attempts: list[dict[str, str | None]] = []
        for _ in range(3):
            raw = self._chat(working, call_type=call_type)
            try:
                parsed = parser(raw)
                attempts.append({"raw_output": raw, "error": None})
                return raw, parsed, attempts
            except Exception as exc:
                error = str(exc)
                attempts.append({"raw_output": raw, "error": error})
                working.extend(
                    [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "The output format was invalid: "
                                + error
                                + ". Return only a corrected output in the requested schema."
                            ),
                        },
                    ]
                )
        raise RuntimeError(
            f"{call_type} did not return valid output after three attempts: {error}"
        )

    def on_step(
        self,
        step_idx: int,
        observation: str,
        action: str,
        outcome: Any,
        next_observation: str,
        benchmark_name: str | None = None,
    ) -> None:
        if not self._decision_log:
            return
        decision = self._decision_log[-1]
        prediction = decision["prediction"]
        reward = getattr(outcome, "reward", None)
        evaluation = evaluate_observed_transition(
            prediction,
            reward=reward,
            previous_reward=self._previous_reward,
            done=bool(getattr(outcome, "done", False)),
            success=getattr(outcome, "success", None),
            invalid_action=bool(getattr(outcome, "invalid_action", False)),
        )
        decision.update(
            {
                "step": int(step_idx),
                "action": action,
                "next_observation": next_observation,
                **evaluation,
            }
        )
        self._task_log["steps"].append(dict(decision))
        self._flush_task_log()
        if reward is not None:
            self._previous_reward = float(reward)

    def on_episode_end(
        self,
        result_meta: Mapping[str, Any] | None = None,
        benchmark_name: str | None = None,
    ) -> dict[str, Any] | None:
        visible_meta = dict(result_meta or {})
        log_meta = dict(visible_meta)
        log_meta["token_usage"] = self.episode_token_usage()
        self._task_log["result_meta"] = log_meta
        update_meta = self._update_knowledge(visible_meta)
        self._task_log["online_update"] = update_meta
        self._flush_task_log()
        return {
            "result_meta": log_meta,
            "online_update": update_meta,
            "decision_log": self._decision_log,
            "artifact_path": str(self._task_log_path),
        }

    def _update_knowledge(self, result_meta: Mapping[str, Any]) -> dict[str, Any]:
        update_id = f"episode_{self._episode_index:06d}"
        update_dir = self.knowledge.root / "updates" / update_id
        update_dir.mkdir(parents=True, exist_ok=True)
        reflection_raw = ""
        plan: dict[str, Any] = {
            "object_knowledge_updates": [],
            "procedural_knowledge_updates": [],
        }
        plan_error: str | None = None
        reflection_prompt: list[dict[str, str]] = []
        for attempt in range(1, max(1, self.max_update_attempts) + 1):
            reflection_prompt = reflection_messages(
                benchmark_name=self._benchmark_name,
                task_context=str(self._task_log.get("initial_observation") or ""),
                object_signatures=self.knowledge.object_signatures(),
                procedural_index=self.knowledge.procedural_index(),
                steps=self._task_log.get("steps", []),
                result_meta=result_meta,
                previous_error=plan_error,
            )
            try:
                reflection_raw = self._chat(
                    reflection_prompt, call_type="ocm_reflect"
                )
                plan = parse_reflection_plan(reflection_raw)
                validate_reflection_plan(
                    plan,
                    object_names=self.knowledge.object_names(),
                    procedural_names=self.knowledge.procedural_names(),
                )
                plan_error = None
                self._log_prompt(
                    "ocm_reflect",
                    reflection_prompt,
                    reflection_raw,
                    {"attempt": attempt, "plan": plan},
                )
                break
            except Exception as exc:
                plan_error = str(exc)
                self._log_prompt(
                    "ocm_reflect",
                    reflection_prompt,
                    reflection_raw,
                    {"attempt": attempt, "error": plan_error},
                )
        self._write_json(update_dir / "reflection_plan.json", plan)

        references = [
            "object." + str(item["target"])
            for item in plan.get("object_knowledge_updates", [])
        ] + [
            "procedural." + str(item["target"])
            for item in plan.get("procedural_knowledge_updates", [])
        ]
        editable_source = self.knowledge.inspect(references)
        (update_dir / "editable_source.py").write_text(
            editable_source, encoding="utf-8"
        )

        object_updates: list[dict[str, str]] = []
        procedural_updates: list[dict[str, str]] = []
        written = {"object": [], "procedural": []}
        update_error = plan_error
        if plan_error is None and references:
            for attempt in range(1, max(1, self.max_update_attempts) + 1):
                update_prompt = knowledge_update_messages(
                    object_signatures=self.knowledge.object_signatures(),
                    procedural_index=self.knowledge.procedural_index(),
                    update_plan=plan,
                    editable_source=editable_source,
                    previous_error=update_error,
                )
                raw = ""
                try:
                    raw = self._chat(
                        update_prompt, call_type="ocm_knowledge_update"
                    )
                    object_updates, procedural_updates = self.knowledge.updates_from_patch(
                        raw, plan
                    )
                    validate_knowledge_updates(
                        plan,
                        object_updates=object_updates,
                        procedural_updates=procedural_updates,
                    )
                    written = self.knowledge.commit_knowledge_updates(
                        object_updates=object_updates,
                        procedural_updates=procedural_updates,
                    )
                    update_error = None
                    self._log_prompt(
                        "ocm_knowledge_update",
                        update_prompt,
                        raw,
                        {"attempt": attempt, "written": written},
                    )
                    break
                except Exception as exc:
                    update_error = str(exc)
                    self._log_prompt(
                        "ocm_knowledge_update",
                        update_prompt,
                        raw,
                        {"attempt": attempt, "error": update_error},
                    )

        self._write_json(
            update_dir / "knowledge_updates.json",
            {
                "object_knowledge": object_updates,
                "procedural_knowledge": procedural_updates,
            },
        )
        meta = {
            "reflection_plan_path": str(update_dir / "reflection_plan.json"),
            "editable_source_path": str(update_dir / "editable_source.py"),
            "knowledge_updates_path": str(update_dir / "knowledge_updates.json"),
            "written_knowledge": written,
            "reflection_plan_error": plan_error,
            "knowledge_update_error": update_error,
            "planned_updates": len(plan["object_knowledge_updates"])
            + len(plan["procedural_knowledge_updates"]),
            "knowledge_root": str(self.knowledge.root),
        }
        self._write_json(update_dir / "meta.json", meta)
        return meta


def _max_index(directory: Path, prefix: str, suffix: str) -> int:
    maximum = 0
    if not directory.exists():
        return maximum
    for path in directory.glob(f"{prefix}*{suffix}"):
        match = re.search(r"(\d+)", path.stem)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _parse_policy_output(
    raw: str,
) -> dict[str, str]:
    action = extract_action_candidate(raw)
    if not action:
        raise ValueError("policy output must include one primitive action")
    match = re.search(
        r"^\s*Thought\s*:\s*(.+?)(?=^\s*(?:Prediction|Action)\s*:)",
        str(raw or ""),
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    thought = match.group(1).strip() if match is not None else "Act on the current evidence."
    return {"thought": thought, "action": action}


def _same_action(left: str, right: str) -> bool:
    return " ".join(left.lower().split()) == " ".join(right.lower().split())


def _usage_delta(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not after:
        return None
    before = before or {}
    result = {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count")
    }
    result["by_call_type"] = {}
    for name, values in (after.get("by_call_type") or {}).items():
        previous = (before.get("by_call_type") or {}).get(name, {})
        delta = {
            key: int(values.get(key, 0)) - int(previous.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count")
        }
        if any(delta.values()):
            result["by_call_type"][name] = delta
    return result
