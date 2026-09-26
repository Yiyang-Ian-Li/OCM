from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .react_agent import ReactAgent


_T = TypeVar("_T")
def _load_worldevolver_classes():
    source_dir = Path(__file__).resolve().parents[1] / "baselines" / "WorldEvolver" / "src"
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.append(source_text)
    from llm.openai_gpt import OPENAI_GPT
    from world_model.episodic_semantic import WMEpisodicSemantic

    return OPENAI_GPT, WMEpisodicSemantic


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
    before_types = before.get("by_call_type", {})
    for name, values in after.get("by_call_type", {}).items():
        previous = before_types.get(name, {})
        delta = {
            key: int(values.get(key, 0)) - int(previous.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count")
        }
        if any(delta.values()):
            result["by_call_type"][name] = delta
    return result


def _thought_from_output(raw_output: str) -> str:
    match = re.search(
        r"(?:Thought|Think)\s*:\s*(.+?)(?=\s+Action\s*:|$)",
        str(raw_output or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match is not None else ""


class WorldEvolverAgent(ReactAgent):
    """ReAct with the released WorldEvolver episodic-semantic world model."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.policy_max_tokens = int(self.config.get("worldevolver_policy_max_tokens", 256))
        self.world_model_max_tokens = int(
            self.config.get("worldevolver_world_model_max_tokens", 1000)
        )
        self._world_model_state = ""
        self._world_model_goal = ""
        self._trajectory_addendum = ""
        self._world_model_llm, self.world_model = self._create_world_model()
        self._checkpoint_path = (
            self.output_dir / "worldevolver_checkpoint.json"
            if self.output_dir is not None
            else None
        )
        if self.config.get("resume") and self._checkpoint_path is not None:
            if not self._checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Missing WorldEvolver checkpoint for resume: {self._checkpoint_path}"
                )
            import json

            self.world_model.load_checkpoint_state(
                json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            )

    def _create_world_model(self):
        llm_class, world_model_class = _load_worldevolver_classes()
        llm = llm_class(
            engine=self.model_name,
            temperature=self.temperature,
            max_tokens=self.world_model_max_tokens,
            base_url=self.config.get("base_url"),
            api_key=self.api_key,
            top_p=1,
            stop=None,
            retry_delays=1,
            max_retry_iters=3,
            context_length=int(self.config.get("worldevolver_context_length", 32768)),
            timeout=self.timeout,
        )
        world_model = world_model_class(
            llm=llm,
            env_name="scienceworld",
            top_k=5,
            batch_k=1,
            sf_confidence_pct=float(
                self.config.get("worldevolver_sf_confidence_pct", 0.0)
            ),
            wm_max_tokens=self.world_model_max_tokens,
            reset_scope="none",
        )
        return llm, world_model

    def _agent_log_dir_name(self) -> str:
        return "worldevolver_agent"

    def on_episode_start(
        self,
        benchmark_name: str | None = None,
        observation: str | None = None,
    ) -> None:
        if benchmark_name != "sciworld":
            raise ValueError("WorldEvolverAgent currently supports the ScienceWorld protocol only.")
        super().on_episode_start(benchmark_name=benchmark_name, observation=observation)
        self._world_model_state = str(observation or "")
        self._world_model_goal = ""
        self._trajectory_addendum = ""
        self.world_model.reset()
        self._task_log["world_model_start"] = self.world_model.state_dict()
        self._flush_task_log()

    def set_world_model_state(self, state: str) -> None:
        self._world_model_state = str(state or "")

    def set_world_model_goal(self, goal: str) -> None:
        self._world_model_goal = str(goal or "")

    def trajectory_observation_addendum(self) -> str:
        return self._trajectory_addendum

    def save_checkpoint(self) -> None:
        if self._checkpoint_path is not None:
            self._write_json(self._checkpoint_path, self.world_model.checkpoint_state())

    def _record_world_model_usage(self, call_type: str, before: Mapping[str, Any]) -> None:
        after = dict(getattr(self._world_model_llm, "total_usage", {}) or {})
        calls = max(0, int(after.get("n_calls", 0)) - int(before.get("n_calls", 0)))
        if calls <= 0:
            return
        usage = {
            key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        for bucket in (self._episode_token_usage, self._lifetime_token_usage):
            for key, value in usage.items():
                bucket[key] += value
            bucket["call_count"] += calls
            by_type = bucket["by_call_type"].setdefault(
                call_type,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "call_count": 0,
                },
            )
            for key, value in usage.items():
                by_type[key] += value
            by_type["call_count"] += calls
        self._last_token_usage = usage

    def _world_model_call(self, call_type: str, fn: Callable[[], _T]) -> _T:
        before = dict(getattr(self._world_model_llm, "total_usage", {}) or {})
        try:
            return fn()
        finally:
            self._record_world_model_usage(call_type, before)

    def _policy_decision(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None,
        *,
        foresight: str = "",
    ) -> tuple[str, str, str]:
        messages = self._build_messages(observation, trajectory)
        if foresight:
            messages[0]["content"] += (
                "\n\nWhen Foresight is provided, it is a world-model prediction of the "
                "next observation for the proposed action. Use it to guide the next action."
            )
            messages[-1]["content"] += "\n\nForesight: " + foresight
        raw_output = ""
        for _ in range(3):
            result = self.llm.create_chat_completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.config.get("top_p"),
                seed=self.config.get("seed"),
                stop=self.stop_sequences or None,
                max_tokens=self.policy_max_tokens,
                call_type="worldevolver_policy",
            )
            raw_output = result.content or ""
            if re.search(r"^\s*Action\s*:\s*\S", raw_output, flags=re.IGNORECASE | re.MULTILINE):
                return (
                    self._normalize_action(raw_output),
                    _thought_from_output(raw_output),
                    raw_output,
                )
            messages.extend(
                [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "The response was incomplete. Return exactly two lines: "
                            "Thought: <brief reasoning> and Action: <one primitive action>."
                        ),
                    },
                ]
            )
        return self._normalize_action(raw_output), _thought_from_output(raw_output), raw_output

    @staticmethod
    def _format_foresight(action: str, prediction: str) -> str:
        text = str(prediction or "").strip()
        if text.lower().startswith("prediction:"):
            text = text.split(":", 1)[1].lstrip()
        return (
            f'If you take action "{action.strip()}", world model predicts '
            f"the next observation: {text.replace(chr(10), ' ')}"
        )

    def act(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None = None,
    ) -> str:
        usage_before = self.episode_token_usage()
        self._trajectory_addendum = ""
        draft_action, draft_thought, draft_raw = self._policy_decision(observation, trajectory)
        state = self._world_model_state or observation
        prediction_output: dict[str, Any] | None = None
        visible_foresight = ""
        final_action, final_thought, final_raw = draft_action, draft_thought, draft_raw

        prediction_output = self._world_model_call(
            "worldevolver_predict",
            lambda: self.world_model.predict(
                state=state,
                action=draft_action,
                goal=self._world_model_goal,
            ),
        )
        prediction = prediction_output.get("prediction")
        foresight = prediction_output.get("foresight", prediction)
        if foresight is not None and str(foresight).strip():
            visible_foresight = self._format_foresight(draft_action, str(foresight))
            final_action, final_thought, final_raw = self._policy_decision(
                observation,
                trajectory,
                foresight=visible_foresight,
            )
            if final_action != draft_action:
                prediction_output = self._world_model_call(
                    "worldevolver_predict",
                    lambda: self.world_model.predict(
                        state=state,
                        action=final_action,
                        goal=self._world_model_goal,
                    ),
                )

        persisted_output = final_raw.strip()
        if visible_foresight and final_action == draft_action:
            self._trajectory_addendum = "Foresight: " + visible_foresight
        self.last_raw_output = persisted_output
        step_usage = _usage_delta(usage_before, self.episode_token_usage())
        if step_usage is not None:
            self._last_token_usage = {
                key: int(step_usage[key])
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        prediction = prediction_output.get("prediction") if prediction_output else None
        self._decision_log.append(
            {
                "step": len(self._decision_log),
                "observation": observation,
                "world_model_state": state,
                "raw_output": persisted_output,
                "parsed_action": final_action,
                "draft_thought": draft_thought,
                "draft_action": draft_action,
                "draft_raw_output": draft_raw,
                "foresight": visible_foresight,
                "final_thought": final_thought,
                "final_raw_output": final_raw,
                "action_changed_by_foresight": final_action != draft_action,
                "prediction": prediction,
                "world_model_trace": {
                    key: value
                    for key, value in (prediction_output or {}).items()
                    if key != "system"
                },
                "token_usage": step_usage,
            }
        )
        return final_action

    def on_step(
        self,
        step_idx: int,
        observation: str,
        action: str,
        outcome: Any,
        next_observation: str,
        benchmark_name: str | None = None,
    ) -> None:
        decision = self._decision_log[-1]
        prediction = decision.get("prediction")
        if prediction is not None:
            self._world_model_call(
                "worldevolver_update",
                lambda: self.world_model.update(
                    state=str(decision["world_model_state"]),
                    action=action,
                    prediction=str(prediction),
                    gold_next_state=next_observation,
                    info={"task": self._world_model_goal},
                ),
            )
            decision["world_model_after_update"] = self.world_model.state_dict()
        super().on_step(
            step_idx=step_idx,
            observation=observation,
            action=action,
            outcome=outcome,
            next_observation=next_observation,
            benchmark_name=benchmark_name,
        )

    def on_episode_end(
        self,
        result_meta: Mapping[str, Any] | None = None,
        benchmark_name: str | None = None,
    ) -> dict[str, Any] | None:
        self._task_log["world_model_end"] = self.world_model.state_dict()
        return super().on_episode_end(result_meta=result_meta, benchmark_name=benchmark_name)
