from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .react_agent import ReactAgent


class ReflexionAgent(ReactAgent):
    def __init__(self, config):
        super().__init__(config)
        self.reflection_limit = int(self.config.get("reflexion_reflection_limit", 3))
        self.attempt_index = 1
        self.reflections: list[dict[str, Any]] = []

    def reset_reflections(self) -> None:
        self.reflections = []

    def set_attempt_index(self, attempt_index: int) -> None:
        self.attempt_index = int(attempt_index)

    def _agent_log_dir_name(self) -> str:
        return "reflexion_agent"

    def _task_log_filename(self, task_id: str) -> str:
        return f"{task_id}_attempt_{self.attempt_index}.json"

    def _relevant_reflections(self) -> list[dict[str, Any]]:
        return self.reflections[-self.reflection_limit :]

    def on_episode_start(
        self,
        benchmark_name: str | None = None,
        observation: str | None = None,
    ) -> None:
        super().on_episode_start(benchmark_name=benchmark_name, observation=observation)
        self._task_log["attempt"] = self.attempt_index
        self._task_log["reflections_at_start"] = list(self.reflections[-self.reflection_limit :])
        self._flush_task_log()

    def _build_messages(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages = super()._build_messages(observation, trajectory)
        reflections = self._relevant_reflections()
        if not reflections:
            return messages
        lessons = "\n".join(f"- {item['reflection']}" for item in reflections if item.get("reflection"))
        if not lessons:
            return messages
        insert_at = 2 if len(messages) > 1 and messages[1].get("role") == "system" else 1
        messages.insert(
            insert_at,
            {
                "role": "system",
                "content": "Relevant lessons from previous failed attempts:\n" + lessons,
            },
        )
        return messages

    def _build_prompt_preamble(self) -> list[dict[str, str]]:
        preamble = super()._build_prompt_preamble()
        reflections = self._relevant_reflections()
        if reflections:
            lessons = "\n".join(f"- {item['reflection']}" for item in reflections if item.get("reflection"))
            if lessons:
                preamble.append({"role": "system", "content": "Relevant lessons from previous failed attempts:\n" + lessons})
        return preamble

    def _build_reflection_messages(self, result_meta: Mapping[str, Any]) -> list[dict[str, str]]:
        compact_steps = [
            {
                "step": item.get("step"),
                "observation": item.get("observation"),
                "raw_output": item.get("raw_output"),
                "action": item.get("action"),
                "reward": item.get("reward"),
                "invalid_action": item.get("invalid_action"),
                "next_observation": item.get("next_observation"),
            }
            for item in self._decision_log
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You write concise lessons for a text-environment agent after a failed episode. "
                    "Return 1-3 short bullet-style lessons. Focus on what to do differently next time. "
                    "Do not include markdown headers."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "result_meta": dict(result_meta),
                        "steps": compact_steps,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
            },
        ]

    def _reflect(self, result_meta: Mapping[str, Any]) -> str:
        messages = self._build_reflection_messages(result_meta)
        result = self.llm.create_chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stop=self.stop_sequences or None,
            call_type="reflect",
        )
        return (result.content or "").strip()

    def on_episode_end(
        self,
        result_meta: Mapping[str, Any] | None = None,
        benchmark_name: str | None = None,
    ) -> dict[str, Any] | None:
        meta = dict(result_meta or {})
        reflection = ""
        if not bool(meta.get("success", False)):
            reflection = self._reflect(meta)
            if reflection:
                self.reflections.append(
                    {
                        "reflection": reflection,
                        "result_meta": meta,
                        "attempt": self.attempt_index,
                    }
                )
                self.reflections = self.reflections[-self.reflection_limit :]
        payload = super().on_episode_end(meta, benchmark_name=benchmark_name)
        if self._task_log:
            self._task_log["reflection"] = reflection
            self._task_log["num_reflections"] = len(self.reflections)
            self._flush_task_log()
        if payload is not None:
            payload["reflection"] = reflection
            payload["num_reflections"] = len(self.reflections)
        return payload
