from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import agents
from benchmarks import BENCHMARKS

from .loops import interactive_loop, reflexion_loop


def build_agent(args: argparse.Namespace) -> agents.Agent:
    if args.agent == "reflexion":
        _require_model(args, "reflexion")
        return agents.ReflexionAgent(_base_config(args) | {"reflexion_reflection_limit": args.reflexion_reflection_limit})
    if args.agent == "ocm":
        _require_model(args, "ocm")
        return agents.OCMAgent(
            _base_config(args)
            | {
                "split": args.split,
                "benchmark": args.benchmark,
                "ocm_codegen_max_attempts": args.ocm_codegen_max_attempts,
                "ocm_knowledge_load_path": args.ocm_knowledge_load_path,
            }
        )
    if args.agent != "react":
        raise ValueError(f"Unsupported agent: {args.agent}")
    _require_model(args, "react")
    return agents.ReactAgent(_base_config(args))


def validate_run_args(args: argparse.Namespace) -> None:
    if args.benchmark == "alfworld" and int(args.alfworld_step_limit) <= 0:
        raise ValueError("--alfworld-step-limit must be a positive integer.")
    if args.benchmark == "sciworld" and int(args.sciworld_env_step_limit) <= 0:
        raise ValueError("--sciworld-env-step-limit must be a positive integer.")
    if args.benchmark == "plancraft" and int(args.plancraft_step_limit) <= 0:
        raise ValueError("--plancraft-step-limit must be a positive integer.")


def prepare_agent_run_state(agent: agents.Agent, args: argparse.Namespace) -> dict[str, Any] | None:
    del agent, args
    return None


def run_agent_loop(
    task,
    agent,
    env_class,
    env_config: dict[str, Any],
    *,
    args: argparse.Namespace,
    benchmark_name: str,
    reward_mode: str,
    progress_callback=None,
) -> dict[str, Any]:
    if args.agent == "reflexion":
        return reflexion_loop(
            task,
            agent,
            env_class,
            env_config,
            benchmark_name=benchmark_name,
            reward_mode=reward_mode,
            max_attempts=int(args.reflexion_max_attempts),
            progress_callback=progress_callback,
        )
    return interactive_loop(
        task,
        agent,
        env_class,
        env_config,
        benchmark_name=benchmark_name,
        reward_mode=reward_mode,
        progress_callback=progress_callback,
    )


def observe_agent_task_result(
    agent: agents.Agent,
    args: argparse.Namespace,
    output_dir: Path,
    run_state: dict[str, Any] | None,
    result: Mapping[str, Any],
) -> None:
    del agent, args, output_dir, run_state, result


def finalize_agent_run_outputs(
    args: argparse.Namespace,
    agent: agents.Agent | None,
    output_dir: Path,
    run_state: dict[str, Any] | None,
    summary: dict[str, Any],
) -> None:
    del output_dir, run_state
    finalize_agent_outputs(args, agent, summary)


def finalize_agent_outputs(args: argparse.Namespace, agent: agents.Agent | None, summary: dict[str, Any]) -> None:
    if args.agent == "reflexion":
        summary["reflexion_max_attempts"] = int(args.reflexion_max_attempts)
        summary["reflexion_reflection_limit"] = int(args.reflexion_reflection_limit)
    del agent


def build_environment_config(benchmark_name: str, args: argparse.Namespace) -> dict[str, Any]:
    if benchmark_name == "sciworld":
        return {
            "env_step_limit": int(args.sciworld_env_step_limit),
            "simplification_str": args.sciworld_simplification,
        }
    if benchmark_name == "alfworld":
        return {"max_steps": int(args.alfworld_step_limit)}
    if benchmark_name == "plancraft":
        return {"max_steps": int(args.plancraft_step_limit)}
    if benchmark_name == "webshop":
        return {"max_steps": int(args.webshop_step_limit)}
    if benchmark_name == "intercode":
        max_steps = None if args.intercode_step_limit is None else int(args.intercode_step_limit)
        return {"max_steps": max_steps}
    return {}


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "provider": args.provider,
        "temperature": args.temperature,
        "timeout": args.timeout,
        "output_dir": args.output_dir,
    }


def _require_model(args: argparse.Namespace, agent_name: str) -> None:
    if not args.model_name:
        raise ValueError(f"--model-name is required when using the {agent_name} agent.")
