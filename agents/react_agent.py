from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import Agent
from .common import extract_action_candidate, load_benchmark_examples, load_benchmark_instruction
from .llm import OpenAIChatClient


class ReactAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = self.config["model_name"]
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError("Missing API key for selected provider. Set it in .env or pass --api-key.")

        base_url = self.config.get("base_url")
        self.temperature = self.config.get("temperature", 0.0)
        self.top_p = self.config.get("top_p")
        self.seed = self.config.get("seed")
        self.timeout = self.config.get("timeout", 120)
        self.llm = OpenAIChatClient(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
            usage_callback=self._record_token_usage,
        )
        self.output_dir = Path(str(self.config.get("output_dir", ""))) if self.config.get("output_dir") else None
        self._task_log_path: Path | None = None
        self._task_log: dict[str, Any] = {}
        self._decision_log: list[dict[str, Any]] = []
        self._benchmark_name = ""
        self._episode_index = 0
        self.last_raw_output = ""

    def _normalize_action(self, raw_output: str) -> str:
        return extract_action_candidate(raw_output)

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _flush_task_log(self) -> None:
        if self._task_log_path is not None:
            self._write_json(self._task_log_path, self._task_log)

    def _agent_log_dir_name(self) -> str:
        return "react_agent"

    def _task_log_filename(self, task_id: str) -> str:
        return f"{task_id}.json"

    def _build_messages(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        benchmark_instruction = load_benchmark_instruction(self._benchmark_name)
        system_parts = [
            (
                    "You are an interactive environment agent. "
                    "Follow benchmark guidance if provided. "
                    "Output exactly two lines in this format:\n"
                    "Thought: <brief reasoning>\n"
                    "Action: <one primitive environment action> "
                    "Do not output markdown or extra text."
            )
        ]
        if benchmark_instruction:
            system_parts.append("Benchmark instruction:\n" + benchmark_instruction)
        benchmark_examples = load_benchmark_examples(self._benchmark_name)
        if benchmark_examples:
            system_parts.append("Benchmark demonstration:\n" + benchmark_examples)
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]

        # Keep conversation history for multi-step planning.
        for msg in list(trajectory or []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})

        current_user = observation
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != current_user:
            messages.append({"role": "user", "content": current_user})
        return messages

    def _build_prompt_preamble(self) -> list[dict[str, str]]:
        benchmark_instruction = load_benchmark_instruction(self._benchmark_name)
        system_parts = [
            (
                    "You are an interactive environment agent. "
                    "Follow benchmark guidance if provided. "
                    "Output exactly two lines in this format:\n"
                    "Thought: <brief reasoning>\n"
                    "Action: <one primitive environment action> "
                    "Do not output markdown or extra text."
            )
        ]
        if benchmark_instruction:
            system_parts.append("Benchmark instruction:\n" + benchmark_instruction)
        benchmark_examples = load_benchmark_examples(self._benchmark_name)
        if benchmark_examples:
            system_parts.append("Benchmark demonstration:\n" + benchmark_examples)
        return [{"role": "system", "content": "\n\n".join(system_parts)}]

    def on_episode_start(
        self,
        benchmark_name: str | None = None,
        observation: str | None = None,
    ) -> None:
        self.reset_episode_token_usage()
        self._benchmark_name = str(benchmark_name or "")
        self._episode_index += 1
        self._decision_log = []
        self._task_log = {
            "benchmark": benchmark_name,
            "prompt_preamble": self._build_prompt_preamble(),
            "initial_observation": observation or "",
            "steps": [],
            "result_meta": None,
        }
        if self.output_dir is None:
            self._task_log_path = None
            return
        episode_id = f"episode_{self._episode_index:06d}"
        self._task_log_path = self.output_dir / self._agent_log_dir_name() / "tasks" / self._task_log_filename(episode_id)
        self._flush_task_log()

    def act(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None = None,
    ) -> str:
        messages = self._build_messages(observation, trajectory)
        result = self.llm.create_chat_completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stop=self.stop_sequences or None,
            call_type="act",
        )
        raw_output = result.content or ""
        self.last_raw_output = raw_output
        action = self._normalize_action(raw_output)
        self._decision_log.append(
            {
                "step": len(self._decision_log),
                "observation": observation,
                "raw_output": raw_output,
                "parsed_action": action,
                "token_usage": self.last_token_usage(),
            }
        )
        return action

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
        meta = dict(result_meta or {})
        meta["token_usage"] = self.episode_token_usage()
        self._task_log["result_meta"] = meta
        self._flush_task_log()
        payload = {
            "result_meta": meta,
            "decision_log": self._decision_log,
            "artifact_path": str(self._task_log_path) if self._task_log_path is not None else None,
        }
        return payload
