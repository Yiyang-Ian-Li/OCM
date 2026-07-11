from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import SENSITIVE_CONFIG_KEYS


def configure_logging(output_dir: Path) -> None:
    class SafeScienceWorldLogFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.name.startswith("scienceworld") and record.msg == "ScienceWorld server running on port":
                record.msg = "ScienceWorld server running on port %s"
            return True

    file_handler = logging.FileHandler(output_dir / "run.log", mode="w")
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
        level=logging.INFO,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(SafeScienceWorldLogFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def prepare_output_dir(args: argparse.Namespace) -> Path:
    model_fragment = args.model_name.replace("/", "_") if args.model_name else "random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.benchmark}_{timestamp}_{args.split}_{args.agent}_{model_fragment}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def serialize_args_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(vars(args))
    for key in SENSITIVE_CONFIG_KEYS:
        if key in config and config[key] is not None:
            config[key] = "<redacted>"
    return config


def write_run_config(output_dir: Path, args: argparse.Namespace) -> Path:
    path = output_dir / "config.json"
    path.write_text(json.dumps(serialize_args_config(args), indent=2), encoding="utf-8")
    return path
