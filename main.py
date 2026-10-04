from __future__ import annotations

import json
import logging
import time
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from benchmarks import BENCHMARKS
from config import resolve_args
from runner import (
    apply_task_limit,
    build_agent,
    build_environment_config,
    build_record,
    configure_logging,
    expand_trials,
    interactive_loop,
    logger,
    prepare_output_dir,
    serialize_args_config,
    summarize_agent_mechanism,
    summarize_records,
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


def _init_wandb(args, output_dir):
    if not args.wandb_project:
        return None
    import wandb

    run_id = args.wandb_run_id
    if args.resume and not run_id:
        run_id = _local_wandb_run_id(output_dir)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=output_dir.name,
        id=run_id,
        resume="must" if args.resume and run_id else None,
        config=serialize_args_config(args),
        dir=str(output_dir),
        tags=[args.benchmark, args.agent, args.model_name],
    )


def _log_wandb_progress(
    run,
    records,
    *,
    benchmark_name: str,
    total_tasks: int,
    started_at: float,
    initial_completed: int = 0,
) -> None:
    if run is None:
        return
    completed = len(records)
    record = records[-1]
    meta = record["result"]["meta"]
    elapsed = time.monotonic() - started_at
    session_completed = completed - initial_completed
    rewards = [
        _score_reward(benchmark_name, item["result"]["meta"]["reward"])
        for item in records
        if item["result"]["meta"].get("reward") is not None
    ]
    successes = sum(bool(item["result"]["meta"].get("success")) for item in records)
    actions = int(meta.get("actions") or 0)
    token_usage = meta.get("token_usage") or {}
    mechanism = summarize_agent_mechanism([record])
    run.log(
        {
            "task/reward": _score_reward(benchmark_name, meta.get("reward") or 0.0),
            "task/success": int(bool(meta.get("success"))),
            "task/actions": actions,
            "task/invalid_actions": int(meta.get("invalid_actions") or 0),
            "task/invalid_action_rate": int(meta.get("invalid_actions") or 0) / max(1, actions),
            "task/total_tokens": int(token_usage.get("total_tokens") or 0),
            "task/predictions": mechanism["predictions"],
            "task/prediction_coverage": mechanism["prediction_coverage"],
            "task/memory_available_decisions": mechanism["memory_available_decisions"],
            "task/memory_referenced_decisions": mechanism["memory_referenced_decisions"],
            "task/memory_reference_rate": mechanism["memory_reference_rate"],
            "task/foresight_revisions": mechanism["foresight_revisions"],
            "task/foresight_revision_rate": mechanism["foresight_revision_rate"],
            "task/post_revision_invalid_actions": mechanism["post_revision_invalid_actions"],
            "task/planned_knowledge_updates": mechanism["planned_knowledge_updates"],
            "task/written_knowledge_updates": mechanism["written_knowledge_updates"],
            "progress/completed_tasks": completed,
            "progress/fraction": completed / max(1, total_tasks),
            "progress/average_reward": sum(rewards) / max(1, len(rewards)),
            "progress/success_rate": successes / completed,
            "progress/tasks_per_hour": session_completed * 3600.0 / max(elapsed, 1e-9),
            "progress/eta_hours": (
                (total_tasks - completed) * elapsed / max(1, session_completed) / 3600.0
            ),
        },
        step=completed,
    )




def run(args) -> dict[str, Any]:
    benchmark = BENCHMARKS[args.benchmark]
    task_class = benchmark["task_class"]
    env_class = benchmark["env_class"]
    env_config = build_environment_config(args.benchmark, args)
    reward_mode = benchmark.get("reward_mode", "latest")
    if args.benchmark == "sciworld" and args.split == "agentboard_native":
        reward_mode = "latest"
    all_tasks, total_tasks = task_class.load_tasks(args.split)
    task_pool = list(all_tasks)[max(0, int(args.task_offset)) :]
    base_tasks = apply_task_limit(task_pool, args.task_limit)
    selected_tasks = expand_trials(base_tasks, args.num_trials)
    output_dir = prepare_output_dir(args)
    task_output_dir = output_dir / "tasks"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_completed_records(task_output_dir, selected_tasks) if args.resume else []
    pending_tasks = selected_tasks[len(records) :]
    args.output_dir = str(output_dir)
    config_path = write_run_config(output_dir, args)
    configure_logging(output_dir, append=args.resume)
    wandb_run = _init_wandb(args, output_dir)
    started_at = time.monotonic()
    logger.info("Benchmark: %s", args.benchmark)
    logger.info("Agent: %s / %s", args.agent, args.model_name or "random")
    logger.info(
        "Selected tasks: %s/%s with %s trial(s) each",
        len(base_tasks),
        total_tasks,
        args.num_trials,
    )
    if records:
        logger.info("Resuming after %s completed tasks.", len(records))
    agent = build_agent(args)
    with logging_redirect_tqdm():
        progress = tqdm(total=len(selected_tasks), initial=len(records))
        progress.refresh()
        progress.set_postfix_str(_running_average_reward(records, args.benchmark), refresh=False)
        for task in pending_tasks:
            result = interactive_loop(
                task,
                agent,
                env_class,
                env_config,
                benchmark_name=args.benchmark,
                reward_mode=reward_mode,
            )
            record = build_record(task, result)
            records.append(record)
            (task_output_dir / f"{task.task_id}.json").write_text(json.dumps(_json_safe(record), indent=2))
            checkpoint_saver = getattr(agent, "save_checkpoint", None)
            if callable(checkpoint_saver):
                checkpoint_saver()
            _log_wandb_progress(
                wandb_run,
                records,
                benchmark_name=args.benchmark,
                total_tasks=len(selected_tasks),
                started_at=started_at,
                initial_completed=len(selected_tasks) - len(pending_tasks),
            )
            progress.set_postfix_str(_running_average_reward(records, args.benchmark), refresh=False)
            progress.update(1)
        progress.close()

    summary = summarize_records(args.benchmark, records, num_trials=args.num_trials)
    summary.update(
        {
            "benchmark": args.benchmark,
            "agent": args.agent,
            "model_name": args.model_name,
            "split": args.split,
            "selected_task_count": len(base_tasks),
            "trial_record_count": len(selected_tasks),
            "output_dir": str(output_dir),
            "task_output_dir": str(task_output_dir),
            "config_path": str(config_path),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2))
    logger.info("Summary:\n%s", json.dumps(_json_safe(summary), indent=2))
    if wandb_run is not None:
        wandb_run.summary.update(_json_safe(summary))
        wandb_run.finish()
    return summary


def _load_completed_records(task_output_dir, selected_tasks) -> list[dict[str, Any]]:
    records = []
    for task in selected_tasks:
        path = task_output_dir / f"{task.task_id}.json"
        if not path.exists():
            break
        record = json.loads(path.read_text(encoding="utf-8"))
        recorded_id = record.get("task", {}).get("task_id")
        if str(recorded_id) != str(task.task_id):
            raise ValueError(f"Resume record {path} belongs to task {recorded_id}, expected {task.task_id}.")
        records.append(record)
    return records


def _local_wandb_run_id(output_dir) -> str | None:
    candidates = list((output_dir / "wandb").glob("run-*-*"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name.rsplit("-", 1)[-1]

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
