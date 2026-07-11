from __future__ import annotations

import os
import yaml
import logging
from pathlib import Path
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment

from benchmarks.base import Task


logger = logging.getLogger(__name__)
BENCHMARK_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BENCHMARK_DIR / "configs"
VALID_SPLITS = {"train", "eval_in_distribution", "eval_out_of_distribution"}

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        split: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.split = split
        self.task_type = next((name for name in PREFIXES if name in game_file), "unknown")

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "split": self.split,
            "game_file": self.game_file,
        }

    @classmethod
    def load_tasks(cls, split: str) -> Tuple[Iterable[Task], int]:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unsupported ALFWorld split: {split}. "
                f"Expected one of: {', '.join(sorted(VALID_SPLITS))}."
            )

        config = load_alfworld_config()
        env_class = alfworld.agents.environment.get_environment(config["env"]["type"])
        env_manager = env_class(config, train_eval=split)
        game_files = list(env_manager.game_files)
        task_items = [
            cls(
                task_id=idx,
                split=split,
                game_file=str(game_file),
            )
            for idx, game_file in enumerate(game_files)
        ]
        return task_items, len(task_items)


def resolve_alfworld_data_path() -> Path:
    data_candidates = []
    if os.environ.get("ALFWORLD_DATA"):
        data_candidates.append(Path(os.environ["ALFWORLD_DATA"]).expanduser())
    data_candidates.append(Path.home() / ".cache" / "alfworld")
    data_candidates.append(BENCHMARK_DIR / "data")

    alfworld_data_path = next(
        (path for path in data_candidates if (path / "json_2.1.1").exists()),
        data_candidates[0],
    )
    os.environ["ALFWORLD_DATA"] = str(alfworld_data_path)
    if not (alfworld_data_path / "json_2.1.1").exists():
        raise FileNotFoundError(
            f"Missing ALFWorld game files under {alfworld_data_path}. "
            "Install/download ALFWorld data first, or set ALFWORLD_DATA to the downloaded data directory."
        )
    return alfworld_data_path


def load_alfworld_config() -> dict:
    resolve_alfworld_data_path()
    config_path = CONFIG_DIR / "base_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
