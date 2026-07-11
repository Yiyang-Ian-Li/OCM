from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.base import Agent
from agents.common import extract_action_candidate, load_benchmark_instruction
from agents.llm import OpenAIChatClient

from .knowledge_code import parse_inspect_request
from .knowledge import KnowledgeStore, parse_knowledge_updates, parse_reflection_plan
from .prompting import (
    action_messages,
    knowledge_update_messages,
    plain_action_messages,
    reflection_messages,
)


class OCMAgent(Agent):
    """Online OCM agent that maintains object and procedural knowledge."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.model_name = str(self.config["model_name"])
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Missing API key for selected provider. Set it in .env or pass --api-key.")
        self.base_url = str(self.config.get("base_url"))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.timeout = int(self.config.get("timeout", 120))
        self.output_dir = Path(self.config["output_dir"])
        self.max_update_attempts = int(self.config.get("ocm_codegen_max_attempts", 3))
        self.knowledge = KnowledgeStore.create(
            self.output_dir,
            load_path=self.config.get("ocm_knowledge_load_path"),
        )
        self.llm = OpenAIChatClient(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            usage_callback=self._record_token_usage,
        )
        self.client = self.llm.client
        self._benchmark_name = ""
        self._task_log_path: Path | None = None
        self._task_log: dict[str, Any] = {}
        self._decision_log: list[dict[str, Any]] = []
        self._episode_index = 0
        self._expanded_symbols: list[str] = []
        self._expanded_source = ""
        self._episode_inspected_sources: dict[str, str] = {}
        self._prompt_counter = 0
        self.last_raw_output = ""

    def _agent_log_dir_name(self) -> str:
        return "ocm_agent"

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _flush_task_log(self) -> None:
        if self._task_log_path is not None:
            self._write_json(self._task_log_path, self._task_log)

    def _chat(self, messages: list[dict[str, str]], *, call_type: str) -> str:
        result = self.llm.create_chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stop=self.stop_sequences or None,
            call_type=call_type,
        )
        return result.content or ""

    def _log_prompt(self, kind: str, messages: list[dict[str, str]], raw_output: str, parsed: Any = None) -> None:
        self._prompt_counter += 1
        prompt_dir = self.output_dir / "ocm_agent" / "prompts"
        if self._task_log_path is not None:
            prompt_dir = self._task_log_path.parent / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            prompt_dir / f"{self._prompt_counter:04d}_{kind}.json",
            {"kind": kind, "messages": messages, "raw_output": raw_output, "parsed": parsed},
        )

    def _build_messages(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None,
        *,
        expanded_source: str = "",
    ) -> list[dict[str, str]]:
        benchmark_instruction = load_benchmark_instruction(self._benchmark_name)
        if not self.knowledge.has_knowledge():
            return plain_action_messages(
                benchmark_instruction=benchmark_instruction,
                observation=observation,
                trajectory=trajectory,
            )
        if not expanded_source.strip():
            expanded_source = self._episode_inspected_context()
        return action_messages(
            benchmark_instruction=benchmark_instruction,
            object_signatures=self.knowledge.object_signatures(),
            procedural_index=self.knowledge.procedural_index(),
            expanded_source=expanded_source,
            observation=observation,
            trajectory=trajectory,
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
        self._expanded_symbols = []
        self._expanded_source = ""
        self._episode_inspected_sources = {}
        episode_id = f"episode_{self._episode_index:06d}"
        self._task_log_path = self.output_dir / "ocm_agent" / "tasks" / f"{episode_id}.json"
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
        messages = self._build_messages(observation, trajectory)
        raw_output = self._chat(messages, call_type="act")
        all_inspect_symbols: list[str] = []
        for _ in range(2):
            inspect_symbols = parse_inspect_request(raw_output)
            if not inspect_symbols:
                break
            all_inspect_symbols.extend(inspect_symbols)
            expanded_source = self.knowledge.inspect(inspect_symbols)
            if not expanded_source.strip():
                expanded_source = (
                    "# Inspect returned no matching learned code for: "
                    + ", ".join(inspect_symbols)
                    + "\n# Choose a real environment action from the current observation."
                )
            else:
                self._remember_inspected_source(inspect_symbols, expanded_source)
            self._expanded_source = expanded_source
            self._expanded_symbols = all_inspect_symbols
            messages = self._build_messages(
                observation,
                trajectory,
                expanded_source=expanded_source,
            )
            raw_output = self._chat(messages, call_type="act_with_inspected_code")
        if parse_inspect_request(raw_output):
            messages = self._build_messages(
                observation,
                trajectory,
                expanded_source=(
                    "# Inspect is an internal code-reading action and cannot be sent to the environment.\n"
                    "# Choose one real environment action now."
                ),
            )
            raw_output = self._chat(messages, call_type="act_after_unresolved_inspect")
        action = extract_action_candidate(raw_output)
        self.last_raw_output = raw_output
        self._decision_log.append(
            {
                "step": len(self._decision_log),
                "observation": observation,
                "raw_output": raw_output,
                "parsed_action": action,
                "inspect_symbols": all_inspect_symbols,
                "token_usage": self.last_token_usage(),
            }
        )
        return action

    def _remember_inspected_source(self, symbols: Sequence[str], source: str) -> None:
        text = source.strip()
        if not text:
            return
        key = ", ".join(str(symbol).strip() for symbol in symbols if str(symbol).strip())
        if not key:
            key = f"inspect_{len(self._episode_inspected_sources) + 1}"
        self._episode_inspected_sources[key] = text

    def _episode_inspected_context(self, *, max_chars: int = 12000) -> str:
        chunks: list[str] = []
        total = 0
        for key, source in reversed(list(self._episode_inspected_sources.items())):
            chunk = f"# Earlier Inspect result for: {key}\n{source.strip()}"
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    chunks.append(chunk[:remaining].rstrip() + "\n... [truncated]")
                break
            chunks.append(chunk)
            total += len(chunk)
        return "\n\n".join(reversed(chunks))

    def on_step(
        self,
        step_idx: int,
        observation: str,
        action: str,
        outcome: Any,
        next_observation: str,
        benchmark_name: str | None = None,
    ) -> None:
        if self._decision_log:
            self._decision_log[-1].update(
                {
                    "step": int(step_idx),
                    "action": action,
                    "reward": getattr(outcome, "reward", None),
                    "done": getattr(outcome, "done", False),
                    "success": getattr(outcome, "success", None),
                    "invalid_action": getattr(outcome, "invalid_action", False),
                    "next_observation": next_observation,
                    "token_usage": self._decision_log[-1].get("token_usage"),
                }
            )
            self._task_log["steps"].append(dict(self._decision_log[-1]))
            self._flush_task_log()

    def on_episode_end(
        self,
        result_meta: Mapping[str, Any] | None = None,
        benchmark_name: str | None = None,
    ) -> dict[str, Any] | None:
        visible_meta = dict(result_meta or {})
        log_meta = dict(visible_meta)
        log_meta["token_usage"] = self.episode_token_usage()
        self._task_log["result_meta"] = log_meta
        update_meta = self._online_update(visible_meta)
        self._task_log["online_update"] = update_meta
        self._flush_task_log()
        return {
            "result_meta": log_meta,
            "online_update": update_meta,
            "decision_log": self._decision_log,
            "artifact_path": str(self._task_log_path) if self._task_log_path is not None else None,
        }

    def _online_update(self, result_meta: Mapping[str, Any]) -> dict[str, Any]:
        update_id = f"episode_{self._episode_index:06d}"
        update_dir = self.output_dir / "ocm_agent" / "knowledge" / "updates" / update_id
        update_dir.mkdir(parents=True, exist_ok=True)

        messages = reflection_messages(
            benchmark_name=self._benchmark_name,
            object_signatures=self.knowledge.object_signatures(),
            procedural_index=self.knowledge.procedural_index(),
            steps=_dedupe_invalid_steps(self._task_log.get("steps", [])),
            result_meta=result_meta,
        )
        reflection_raw = ""
        update_plan: dict[str, Any] = {
            "object_knowledge_updates": [],
            "procedural_knowledge_updates": [],
            "inspect_objects": [],
            "inspect_procedural": [],
        }
        plan_error: str | None = None
        for attempt in range(1, max(1, self.max_update_attempts) + 1):
            try:
                reflection_raw = self._chat(messages, call_type="ocm_reflect")
                update_plan = parse_reflection_plan(reflection_raw)
                plan_error = None
                self._log_prompt("ocm_reflect", messages, reflection_raw, {"attempt": attempt, "update_plan": update_plan})
                break
            except Exception as exc:
                plan_error = str(exc)
                self._log_prompt("ocm_reflect", messages, reflection_raw, {"attempt": attempt, "error": plan_error})
        self._write_json(update_dir / "reflection_plan.json", update_plan)

        inspected_symbols = [f"object.{name}" for name in update_plan.get("inspect_objects", [])]
        inspected_symbols.extend(f"procedural.{name}" for name in update_plan.get("inspect_procedural", []))
        inspected_source = self.knowledge.inspect(inspected_symbols)
        (update_dir / "inspected_source.py").write_text(inspected_source, encoding="utf-8")

        object_updates: list[dict[str, str]] = []
        procedural_updates: list[dict[str, str]] = []
        written_knowledge: dict[str, list[str]] = {"object": [], "procedural": []}
        update_error: str | None = plan_error
        update_attempts = range(1, max(1, self.max_update_attempts) + 1) if plan_error is None else []
        for attempt in update_attempts:
            messages = knowledge_update_messages(
                object_signatures=self.knowledge.object_signatures(),
                procedural_index=self.knowledge.procedural_index(),
                update_plan=update_plan,
                inspected_source=inspected_source,
                steps=_dedupe_invalid_steps(self._task_log.get("steps", [])),
                result_meta=result_meta,
                previous_error=update_error,
            )
            raw = ""
            try:
                raw = self._chat(messages, call_type="ocm_knowledge_update")
                object_updates, procedural_updates = parse_knowledge_updates(raw)
                written_knowledge = self.knowledge.commit_knowledge_updates(
                    object_updates=object_updates,
                    procedural_updates=procedural_updates,
                )
                self._log_prompt(
                    "ocm_knowledge_update",
                    messages,
                    raw,
                    {"attempt": attempt, "written": written_knowledge, "checked": True},
                )
                update_error = None
                break
            except Exception as exc:
                update_error = str(exc)
                self._log_prompt("ocm_knowledge_update", messages, raw, {"attempt": attempt, "error": update_error})
        self._write_json(
            update_dir / "knowledge_updates.json",
            {"object_knowledge": object_updates, "procedural_knowledge": procedural_updates},
        )

        meta = {
            "reflection_plan_path": str(update_dir / "reflection_plan.json"),
            "inspected_source_path": str(update_dir / "inspected_source.py"),
            "knowledge_updates_path": str(update_dir / "knowledge_updates.json"),
            "written_knowledge": written_knowledge,
            "reflection_plan_error": plan_error,
            "knowledge_update_error": update_error,
            "knowledge_root": str(self.knowledge.root),
        }
        (update_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
        return meta


def _dedupe_invalid_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    invalid_seen: dict[str, dict[str, Any]] = {}
    for step in steps:
        item = dict(step)
        action = str(item.get("action") or item.get("parsed_action") or "").strip()
        invalid = bool(item.get("invalid_action"))
        if not invalid:
            compacted.append(item)
            continue
        key = action.lower()
        if key not in invalid_seen:
            item["repeat_count"] = 1
            invalid_seen[key] = item
            compacted.append(item)
        else:
            invalid_seen[key]["repeat_count"] = int(invalid_seen[key].get("repeat_count", 1)) + 1
    return compacted
