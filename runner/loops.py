from __future__ import annotations

from typing import Any, Mapping

from benchmarks.base import StepResult

from .state import logger
from .usage import clone_token_usage


def empty_result() -> dict[str, Any]:
    return {
        "meta": {
            "steps": 0,
            "actions": 0,
            "env_steps": 0,
            "reward": None,
            "finished": False,
            "success": False,
            "terminate_reason": None,
            "invalid_actions": 0,
            "repeated_invalid_action_count": 0,
            "repeated_invalid_action_limit": 10,
        },
        "trajectory": [],
        "step_records": [],
    }


def update_reward(current_reward: float | None, outcome: StepResult, reward_mode: str) -> float | None:
    if outcome.reward is None:
        return current_reward
    if reward_mode == "max":
        return outcome.reward if current_reward is None else max(current_reward, outcome.reward)
    return outcome.reward


def agent_visible_result_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(meta.get("success", False)),
        "reward": meta.get("reward"),
        "terminate_reason": meta.get("terminate_reason"),
        "steps": meta.get("steps"),
        "invalid_actions": meta.get("invalid_actions"),
    }


def interactive_loop(
    task,
    agent,
    env_class,
    env_config: dict[str, Any],
    benchmark_name: str,
    reward_mode: str,
) -> dict[str, Any]:
    environment = env_class(task, **env_config)
    try:
        observation = environment.reset()
        agent_observation = observation
        result = empty_result()
        result["trajectory"] = [{"role": "user", "content": agent_observation}]
        logger.info("%s\nInitial observation:\n%s", _task_label(task), observation)
        agent.on_episode_start(
            benchmark_name=benchmark_name,
            observation=observation,
        )
        _publish_world_model_goal(agent, environment)
        previous_env_steps = 0
        last_invalid_action_key: str | None = None
        repeated_invalid_action_count = 0
        repeated_invalid_action_limit = int(result["meta"]["repeated_invalid_action_limit"])
        while not result["meta"]["finished"]:
            _publish_world_model_state(agent, environment, observation)
            action = agent.act(
                observation=agent_observation,
                trajectory=result["trajectory"],
            )
            action_token_usage = clone_token_usage(agent.last_token_usage())
            _attach_observation_addendum(agent, result["trajectory"])
            assistant_content = agent.last_raw_output or action
            result["trajectory"].append({"role": "assistant", "content": assistant_content})
            outcome = environment.step(action)
            next_observation = outcome.observation
            next_agent_observation = _agent_observation(next_observation, outcome)
            result["trajectory"].append({"role": "user", "content": next_agent_observation})
            result["meta"]["actions"] += 1
            result["meta"]["env_steps"] = _effective_step_count(outcome, fallback=int(result["meta"]["actions"]))
            env_step_delta = max(0, int(result["meta"]["env_steps"]) - previous_env_steps)
            previous_env_steps = int(result["meta"]["env_steps"])
            result["meta"]["steps"] += max(1, env_step_delta)
            if outcome.invalid_action:
                result["meta"]["invalid_actions"] += 1
                invalid_action_key = action.strip().lower()
                if invalid_action_key == last_invalid_action_key:
                    repeated_invalid_action_count += 1
                else:
                    last_invalid_action_key = invalid_action_key
                    repeated_invalid_action_count = 1
                result["meta"]["repeated_invalid_action_count"] = repeated_invalid_action_count
            else:
                last_invalid_action_key = None
                repeated_invalid_action_count = 0
                result["meta"]["repeated_invalid_action_count"] = 0
            result["meta"]["reward"] = update_reward(result["meta"]["reward"], outcome, reward_mode)
            if outcome.done:
                result["meta"]["finished"] = True
                result["meta"]["success"] = bool(outcome.success)
                result["meta"]["terminate_reason"] = outcome.terminate_reason or ("success" if outcome.success else "done")
            elif repeated_invalid_action_count >= repeated_invalid_action_limit:
                result["meta"]["finished"] = True
                result["meta"]["success"] = False
                result["meta"]["terminate_reason"] = "repeated_invalid_action"
            elif environment.max_steps is not None and result["meta"]["steps"] >= environment.max_steps:
                result["meta"]["finished"] = True
                result["meta"]["success"] = False
                result["meta"]["terminate_reason"] = "max_steps"
            result["step_records"].append(
                {
                    "step": result["meta"]["steps"],
                    "action_index": result["meta"]["actions"],
                    "env_step": result["meta"]["env_steps"],
                    "observation": observation,
                    "action": action,
                    "next_observation": next_observation,
                    "reward": outcome.reward,
                    "done": outcome.done,
                    "success": outcome.success,
                    "invalid_action": outcome.invalid_action,
                    "token_usage": action_token_usage,
                }
            )
            logger.info(
                "%s",
                _format_step_log(
                    step=result["meta"]["steps"],
                    max_steps=environment.max_steps,
                    action=action,
                    next_observation=next_observation,
                    outcome=outcome,
                ),
            )
            agent.on_step(
                step_idx=result["meta"]["steps"],
                observation=observation,
                action=action,
                outcome=outcome,
                next_observation=next_observation,
                benchmark_name=benchmark_name,
            )
            observation = next_observation
            agent_observation = next_agent_observation

        agent_artifact = agent.on_episode_end(
            result_meta=agent_visible_result_meta(result["meta"]),
            benchmark_name=benchmark_name,
        )
        result["meta"]["token_usage"] = clone_token_usage(agent.episode_token_usage())
        if agent_artifact is not None:
            result["agent_artifact"] = agent_artifact
        return result
    finally:
        environment.close()


def _task_label(task) -> str:
    metadata = task.metadata()
    task_id = metadata.get("task_id", "unknown")
    name = metadata.get("sub_task_name") or metadata.get("task_name") or metadata.get("name") or metadata.get("id") or task_id
    variation = metadata.get("variation_idx", metadata.get("variation", None))
    if variation is None:
        return f"Task {task_id} | {name}"
    return f"Task {task_id} | {name} | variation={variation}"


def _format_step_log(*, step: int, max_steps: int | None, action: str, next_observation: str, outcome: StepResult) -> str:
    status = "INVALID" if outcome.invalid_action else "ok"
    step_label = f"{step}/{max_steps}" if max_steps is not None else str(step)
    return "\n".join(
        [
            f"> Step {step_label}",
            f"Action: {action}",
            "Observation:",
            next_observation,
            f"Result: {status} reward={outcome.reward} done={outcome.done} success={outcome.success}",
        ]
    )


def _effective_step_count(outcome: StepResult, *, fallback: int) -> int:
    moves = outcome.info.get("moves") if isinstance(outcome.info, dict) else None
    if isinstance(moves, int):
        return moves
    try:
        return int(moves)
    except (TypeError, ValueError):
        return fallback


def _publish_world_model_state(agent, environment, observation: str) -> None:
    setter = getattr(agent, "set_world_model_state", None)
    if not callable(setter):
        return
    state_builder = getattr(environment, "world_model_state", None)
    state = state_builder(observation) if callable(state_builder) else observation
    setter(state)


def _publish_world_model_goal(agent, environment) -> None:
    setter = getattr(agent, "set_world_model_goal", None)
    if not callable(setter):
        return
    goal_builder = getattr(environment, "world_model_goal", None)
    setter(goal_builder() if callable(goal_builder) else "")


def _attach_observation_addendum(agent, trajectory: list[dict[str, str]]) -> None:
    getter = getattr(agent, "trajectory_observation_addendum", None)
    addendum = str(getter() if callable(getter) else "").strip()
    if not addendum:
        return
    if not trajectory or trajectory[-1].get("role") != "user":
        raise ValueError("World-model foresight must follow the current observation.")
    trajectory[-1]["content"] = trajectory[-1]["content"].rstrip() + "\n\n" + addendum


def _agent_observation(observation: str, outcome: StepResult) -> str:
    feedback = (
        f"reward={outcome.reward}, done={outcome.done}, "
        f"success={outcome.success}, invalid_action={outcome.invalid_action}"
    )
    return f"{observation}\n\nEnvironment feedback: {feedback}"
