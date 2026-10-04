from __future__ import annotations

import argparse
from typing import Any

import agents


def build_agent(args: argparse.Namespace) -> agents.Agent:
    if args.agent == "ocm":
        _require_model(args, args.agent)
        return agents.OCMAgent(
            _base_config(args)
            | {
                "ocm_codegen_max_attempts": args.ocm_codegen_max_attempts,
                "ocm_knowledge_load_path": args.ocm_knowledge_load_path,
            }
        )
    if args.agent == "worldevolver":
        _require_model(args, args.agent)
        return agents.WorldEvolverAgent(
            _base_config(args)
            | {
                "worldevolver_policy_max_tokens": args.worldevolver_policy_max_tokens,
                "worldevolver_world_model_max_tokens": args.worldevolver_world_model_max_tokens,
                "worldevolver_context_length": args.worldevolver_context_length,
                "worldevolver_sf_confidence_pct": args.worldevolver_sf_confidence_pct,
            }
        )
    if args.agent != "react":
        raise ValueError(f"Unsupported agent: {args.agent}")
    _require_model(args, "react")
    if getattr(args, "react_implementation", "local") == "worldevolver":
        from agents.reference_react_agent import ReferenceReactAgent
        return ReferenceReactAgent(_base_config(args))
    return agents.ReactAgent(_base_config(args))


def build_environment_config(benchmark_name: str, args: argparse.Namespace) -> dict[str, Any]:
    if benchmark_name == "sciworld":
        return {
            "env_step_limit": int(args.sciworld_env_step_limit),
            "agent_step_limit": int(args.sciworld_agent_step_limit),
            "simplification_str": args.sciworld_simplification,
        }
    if benchmark_name == "alfworld":
        return {"max_steps": int(args.alfworld_step_limit)}
    if benchmark_name == "plancraft":
        return {"max_steps": int(args.plancraft_step_limit)}
    if benchmark_name == "webshop":
        return {"max_steps": int(args.webshop_step_limit)}
    raise ValueError(f"Unsupported benchmark: {benchmark_name}")


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "timeout": args.timeout,
        "output_dir": args.output_dir,
        "resume": args.resume,
    }


def _require_model(args: argparse.Namespace, agent_name: str) -> None:
    if not args.model_name:
        raise ValueError(f"--model-name is required when using the {agent_name} agent.")
