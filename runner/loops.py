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
    progress_callback=None,
    attempt_idx: int | None = None,
) -> dict[str, Any]:
    environment = env_class(task, **env_config)
    try:
        observation = environment.reset()
        result = empty_result()
        result["trajectory"] = [{"role": "user", "content": observation}]
        logger.info("%s\nInitial observation:\n%s", _task_label(task), observation)
        agent.on_episode_start(
            benchmark_name=benchmark_name,
            observation=observation,
        )
        _publish_task_info(agent, task)
        _emit_progress(
            progress_callback,
            event="start",
            task=task,
            step=0,
            max_steps=environment.max_steps,
            reward=result["meta"]["reward"],
            attempt=attempt_idx,
        )

        previous_env_steps = 0
        last_invalid_action_key: str | None = None
        repeated_invalid_action_count = 0
        repeated_invalid_action_limit = int(result["meta"]["repeated_invalid_action_limit"])
        while not result["meta"]["finished"]:
            action = agent.act(
                observation=observation,
                trajectory=result["trajectory"],
            )
            action_token_usage = clone_token_usage(getattr(agent, "last_token_usage", lambda: None)())
            assistant_content = getattr(agent, "last_raw_output", action) or action
            result["trajectory"].append({"role": "assistant", "content": assistant_content})
            outcome = environment.step(action)
            next_observation = outcome.observation
            result["trajectory"].append({"role": "user", "content": next_observation})
            result["meta"]["actions"] += 1
            result["meta"]["env_steps"] = _effective_step_count(outcome, fallback=int(result["meta"]["actions"]))
            env_step_delta = max(0, int(result["meta"]["env_steps"]) - previous_env_steps)
            previous_env_steps = int(result["meta"]["env_steps"])
            result["meta"]["steps"] += max(1, env_step_delta)
            if hasattr(agent, "on_step_start"):
                agent.on_step_start(
                    step_idx=result["meta"]["steps"],
                    observation=observation,
                    action=action,
                    next_observation=next_observation,
                    benchmark_name=benchmark_name,
                )
            if outcome.invalid_action:
                result["meta"]["invalid_actions"] += 1
                invalid_action_key = action.strip().lower()
                if invalid_action_key == last_invalid_action_key:
                    repeated_invalid_action_count += 1
                else:
                    last_invalid_action_key = invalid_action_key
                    repeated_invalid_action_count = 1
                result["meta"]["repeated_invalid_action_count"] = repeated_invalid_action_count
                result["meta"]["repeated_invalid_action_limit"] = repeated_invalid_action_limit
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
            if not result["meta"]["finished"] and hasattr(agent, "should_stop_episode"):
                should_stop, stop_reason = agent.should_stop_episode()
                if should_stop:
                    result["meta"]["finished"] = True
                    result["meta"]["success"] = False
                    result["meta"]["terminate_reason"] = stop_reason or "agent_stop"
            if _should_emit_step_progress(result["meta"]["steps"], done=bool(result["meta"]["finished"])):
                _emit_progress(
                    progress_callback,
                    event="step",
                    task=task,
                    step=result["meta"]["steps"],
                    max_steps=environment.max_steps,
                    reward=result["meta"]["reward"],
                    action=action,
                    attempt=attempt_idx,
                    done=bool(result["meta"]["finished"]),
                    success=result["meta"]["success"] if result["meta"]["finished"] else None,
                    terminate_reason=result["meta"]["terminate_reason"] if result["meta"]["finished"] else None,
                )
            observation = next_observation

        agent_artifact = agent.on_episode_end(
            result_meta=agent_visible_result_meta(result["meta"]),
            benchmark_name=benchmark_name,
        )
        result["meta"]["token_usage"] = clone_token_usage(getattr(agent, "episode_token_usage", lambda: None)())
        if agent_artifact is not None:
            result["agent_artifact"] = agent_artifact
        return result
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def reflexion_loop(
    task,
    agent,
    env_class,
    env_config: dict[str, Any],
    benchmark_name: str,
    reward_mode: str,
    max_attempts: int,
    progress_callback=None,
) -> dict[str, Any]:
    if hasattr(agent, "reset_reflections"):
        agent.reset_reflections()
    attempts: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    for attempt_idx in range(1, max(1, int(max_attempts)) + 1):
        if hasattr(agent, "set_attempt_index"):
            agent.set_attempt_index(attempt_idx)
        logger.info("Reflexion attempt %s/%s for task %s", attempt_idx, max_attempts, task.task_id)
        result = interactive_loop(
            task,
            agent,
            env_class,
            env_config,
            benchmark_name=benchmark_name,
            reward_mode=reward_mode,
            progress_callback=progress_callback,
            attempt_idx=attempt_idx,
        )
        attempts.append(
            {
                "attempt": attempt_idx,
                "meta": dict(result["meta"]),
                "agent_artifact": result.get("agent_artifact"),
            }
        )
        final_result = result
        if result["meta"]["success"]:
            break
    assert final_result is not None
    final_result["attempts"] = attempts
    final_result["meta"]["attempts"] = len(attempts)
    return final_result


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


def _publish_task_info(agent, task) -> None:
    setter = getattr(agent, "set_task_info", None)
    if not callable(setter):
        return
    metadata = task.metadata() if hasattr(task, "metadata") else {}
    setter(metadata)


def _emit_progress(
    progress_callback,
    *,
    event: str,
    task,
    step: int,
    max_steps: int | None,
    reward: float | None,
    action: str | None = None,
    attempt: int | None = None,
    done: bool = False,
    success: bool | None = None,
    terminate_reason: str | None = None,
) -> None:
    if progress_callback is None:
        return
    payload = {
        "event": event,
        "task_id": str(getattr(task, "task_id", "unknown")),
        "step": int(step),
        "max_steps": int(max_steps) if max_steps is not None else None,
        "reward": reward,
        "done": bool(done),
        "success": success,
        "terminate_reason": terminate_reason,
    }
    if action is not None:
        payload["action"] = str(action)
    if attempt is not None:
        payload["attempt"] = int(attempt)
    progress_callback(payload)


def _should_emit_step_progress(step: int, *, done: bool) -> bool:
    return step == 1 or step % 5 == 0 or done
