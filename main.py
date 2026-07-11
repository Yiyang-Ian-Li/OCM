from __future__ import annotations

import json
import logging
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from benchmarks import BENCHMARKS
from config import resolve_args
from runner import (
    build_agent,
    apply_task_limit,
    build_environment_config,
    build_record,
    configure_logging,
    empty_result,
    finalize_agent_run_outputs,
    logger,
    observe_agent_task_result,
    prepare_agent_run_state,
    prepare_output_dir,
    run_agent_loop,
    summarize_records,
    validate_run_args,
    write_run_config,
)


def _running_average_reward(records: list[dict[str, Any]], benchmark_name: str) -> str:
    rewards = [
        _score_reward(benchmark_name, record["result"]["meta"].get("reward"))
        for record in records
        if record["result"]["meta"].get("reward") is not None
    ]
    if not rewards:
        return "avg_reward=NA"
    return f"avg_reward={sum(rewards) / len(rewards):.3f}"


def _score_reward(benchmark_name: str, reward: float) -> float:
    if benchmark_name == "sciworld":
        return max(0.0, float(reward))
    return reward




def run(args) -> dict[str, Any]:
    benchmark = BENCHMARKS[args.benchmark]
    task_class = benchmark["task_class"]
    env_class = benchmark["env_class"]
    env_config = build_environment_config(args.benchmark, args)
    reward_mode = benchmark.get("reward_mode", "latest")
    all_tasks, total_tasks = task_class.load_tasks(args.split)
    selected_tasks = apply_task_limit(all_tasks, args.task_limit)
    output_dir = prepare_output_dir(args)
    task_output_dir = output_dir / "tasks"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    config_path = write_run_config(output_dir, args)
    configure_logging(output_dir)
    validate_run_args(args)

    records: list[dict[str, Any]] = []
    logger.info("Benchmark: %s", args.benchmark)
    logger.info("Agent: %s / %s", args.agent, args.model_name or "random")
    logger.info("Selected tasks: %s/%s", len(selected_tasks), total_tasks)
    agent = build_agent(args)
    run_state = prepare_agent_run_state(agent, args)
    with logging_redirect_tqdm():
        progress = tqdm(total=len(selected_tasks))
        progress.refresh()
        progress.set_postfix_str(_running_average_reward(records, args.benchmark), refresh=False)
        for task in selected_tasks:
            try:
                result = run_agent_loop(
                    task,
                    agent,
                    env_class,
                    env_config,
                    args=args,
                    benchmark_name=args.benchmark,
                    reward_mode=reward_mode,
                )
            except Exception as exc:
                logger.exception("Task %s failed; recording failure and continuing.", getattr(task, "task_id", "unknown"))
                result = _failed_task_result(exc, agent)
            record = build_record(task, result)
            records.append(record)
            observe_agent_task_result(agent, args, output_dir, run_state, result)
            (task_output_dir / f"{task.task_id}.json").write_text(json.dumps(_json_safe(record), indent=2))
            progress.set_postfix_str(_running_average_reward(records, args.benchmark), refresh=False)
            progress.update(1)
        progress.close()

    summary = summarize_records(args.benchmark, records)
    summary.update(
        {
            "benchmark": args.benchmark,
            "agent": args.agent,
            "model_name": args.model_name,
            "split": args.split,
            "selected_task_count": len(selected_tasks),
            "output_dir": str(output_dir),
            "task_output_dir": str(task_output_dir),
            "config_path": str(config_path),
        }
    )
    finalize_agent_run_outputs(args, agent, output_dir, run_state, summary)
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    logger.info("Summary:\n%s", json.dumps(_json_safe(summary), indent=2))
    return summary


def _failed_task_result(exc: Exception, agent) -> dict[str, Any]:
    result = empty_result()
    result["meta"].update(
        {
            "finished": True,
            "success": False,
            "terminate_reason": "exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )
    token_usage = getattr(agent, "episode_token_usage", lambda: None)()
    if token_usage is not None:
        result["meta"]["token_usage"] = token_usage
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_pyrunnable"):
        try:
            return _json_safe(value.to_pyrunnable())
        except Exception:
            return str(value)
    return str(value)


if __name__ == "__main__":
    run(resolve_args())
