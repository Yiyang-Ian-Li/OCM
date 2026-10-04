from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path

from benchmarks import BENCHMARKS


DEFAULT_CONFIG = {
    # Benchmark selection and task slicing.
    "benchmark": "plancraft",  # Which benchmark to run by default.
    "split": "test.small",  # Dataset split using the selected benchmark's native split names.
    "task_limit": None,  # Optional cap on number of tasks to execute.
    "task_offset": 0,
    "num_trials": 1,  # Consecutive attempts per task; metrics use best-of-trials.

    # Run output controls.
    "output_dir": "outputs",  # Parent directory for run outputs.
    "run_name": None,  # Explicit run folder name; otherwise auto-generated.
    "resume": False,  # Continue an interrupted run from its completed task records.
    "wandb_project": None,  # Enable Weights & Biases tracking when set.
    "wandb_entity": None,  # Optional Weights & Biases entity override.
    "wandb_run_id": None,  # Optional run id used when resuming W&B.

    # Benchmark-specific settings.
    "alfworld_step_limit": 50,
    "sciworld_simplification": "easy",
    "sciworld_env_step_limit": 100,
    "sciworld_agent_step_limit": 100,
    "plancraft_step_limit": 30,
    "webshop_step_limit": 15,

    # Agent backend selection.
    "agent": "ocm",  # Agent backend.
    "react_implementation": "local",

    # Base agent model settings.
    "provider": "openai",  # LLM provider used for OpenAI-compatible requests.
    "model_name": "gpt-4.1-mini",  # Model name for the selected provider.
    "base_url": None,  # Optional override for the selected provider base URL.
    "api_key": None,  # API key resolved from CLI or .env for the selected provider.
    "temperature": 0.0,  # Sampling temperature for model generation.
    "top_p": 0.5,
    "seed": 42,
    "timeout": 120,  # Request timeout in seconds.

    # OCM agent settings.
    "ocm_codegen_max_attempts": 3,
    "ocm_knowledge_load_path": None,

    # WorldEvolver baseline settings.
    "worldevolver_policy_max_tokens": 256,
    "worldevolver_world_model_max_tokens": 1000,
    "worldevolver_context_length": 32768,
    "worldevolver_sf_confidence_pct": 0.0,
}


PROVIDER_BASE_URLS = {
    "llamacpp": "http://127.0.0.1:8080/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}


PROVIDER_API_KEY_ENV_VARS = {
    "llamacpp": ("LLAMACPP_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


PROVIDER_BASE_URL_ENV_VARS = {
    "llamacpp": ("LLAMACPP_BASE_URL",),
    "openrouter": ("OPENROUTER_BASE_URL",),
    "openai": ("OPENAI_BASE_URL",),
}


def load_repo_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run benchmark evaluation.")

    # Benchmark selection and task slicing.
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default=None)  # Override default benchmark.
    parser.add_argument("--split", default=None)  # Override dataset split.
    parser.add_argument("--task-limit", type=int, default=None)  # Override task cap.
    parser.add_argument("--task-offset", type=int, default=None)
    parser.add_argument("--num-trials", type=int, default=None)

    # Run output controls.
    parser.add_argument("--output-dir", default=None)  # Override parent output directory.
    parser.add_argument("--run-name", default=None)  # Override run folder name.
    parser.add_argument("--resume", action="store_true", default=None)  # Resume completed tasks in the run folder.
    parser.add_argument("--wandb-project", default=None)  # Enable Weights & Biases tracking.
    parser.add_argument("--wandb-entity", default=None)  # Override the default Weights & Biases entity.
    parser.add_argument("--wandb-run-id", default=None)  # Override the W&B run id used for resume.
    parser.add_argument("--alfworld-step-limit", type=int, default=None)  # ALFWorld action budget.
    parser.add_argument("--sciworld-simplification", default=None)  # ScienceWorld simplification string, e.g. easy.
    parser.add_argument("--sciworld-env-step-limit", type=int, default=None)  # ScienceWorld internal move budget.
    parser.add_argument("--sciworld-agent-step-limit", type=int, default=None)
    parser.add_argument("--plancraft-step-limit", type=int, default=None)  # Plancraft action budget.
    parser.add_argument("--webshop-step-limit", type=int, default=None)  # WebShop action budget.

    # Agent backend selection.
    parser.add_argument("--agent", choices=["react", "ocm", "worldevolver"], default=None)  # Override agent backend.
    parser.add_argument("--react-implementation", choices=["local", "worldevolver"], default=None)

    # Base agent model settings.
    parser.add_argument("--provider", choices=sorted(PROVIDER_BASE_URLS), default=None)  # Override provider.
    parser.add_argument("--model-name", default=None)  # Override model name.
    parser.add_argument("--base-url", default=None)  # Override OpenAI-compatible API base URL.
    parser.add_argument("--api-key", default=None)  # Override API key.
    parser.add_argument("--temperature", type=float, default=None)  # Override sampling temperature.
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)  # Override request timeout.
    parser.add_argument("--ocm-codegen-max-attempts", type=int, default=None)
    parser.add_argument("--ocm-knowledge-load-path", default=None)
    parser.add_argument("--worldevolver-policy-max-tokens", type=int, default=None)
    parser.add_argument("--worldevolver-world-model-max-tokens", type=int, default=None)
    parser.add_argument("--worldevolver-context-length", type=int, default=None)
    parser.add_argument("--worldevolver-sf-confidence-pct", type=float, default=None)
    return parser


def resolve_api_key(provider: str, explicit_api_key: str | None) -> str | None:
    if explicit_api_key:
        return explicit_api_key
    for env_var in PROVIDER_API_KEY_ENV_VARS[str(provider)]:
        value = os.environ.get(env_var)
        if value:
            return value
    if provider == "llamacpp":
        return "local"
    return None


def resolve_base_url(provider: str, explicit_base_url: str | None) -> str:
    if explicit_base_url:
        return explicit_base_url
    for env_var in PROVIDER_BASE_URL_ENV_VARS[str(provider)]:
        value = os.environ.get(env_var)
        if value:
            return value
    return PROVIDER_BASE_URLS[str(provider)]


def resolve_args(cli_args: argparse.Namespace | None = None) -> argparse.Namespace:
    load_repo_env()
    parser = build_parser()
    parsed = cli_args if cli_args is not None else parser.parse_args()
    merged = deepcopy(DEFAULT_CONFIG)
    for key, value in vars(parsed).items():
        if value is not None:
            merged[key] = value
    merged["base_url"] = resolve_base_url(str(merged["provider"]), merged.get("base_url"))
    merged["api_key"] = resolve_api_key(str(merged["provider"]), merged.get("api_key"))
    return argparse.Namespace(**merged)
