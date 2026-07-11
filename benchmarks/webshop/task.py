from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from benchmarks.base import Task
from benchmarks.external_repos import RAW_BENCHMARKS_DIR, ensure_git_repo


WEBSHOP_REPO_URL = "https://github.com/princeton-nlp/WebShop.git"
DEFAULT_WEBSHOP_REPO_PATH = RAW_BENCHMARKS_DIR / "webshop"
VALID_SPLITS = {"train", "test", "test500", "full"}


def resolve_webshop_repo(path: str | Path | None = None) -> Path:
    return ensure_git_repo(
        Path(path).expanduser() if path else DEFAULT_WEBSHOP_REPO_PATH,
        WEBSHOP_REPO_URL,
        "web_agent_site",
    )


class WebShopTask(Task):
    task_name = "webshop"

    def __init__(self, session_id: int, split: str, ordinal: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session_id = int(session_id)
        self.split = split
        self.ordinal = int(ordinal)

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "session_id": self.session_id,
            "split": self.split,
            "ordinal": self.ordinal,
        }

    @classmethod
    def load_tasks(cls, split: str) -> tuple[Iterable[Task], int]:
        repo_path = resolve_webshop_repo()
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported WebShop split: {split}. Expected one of: {', '.join(sorted(VALID_SPLITS))}.")
        if split == "full":
            session_ids = list(range(12087))
        else:
            filename = {
                "train": "train_indices.json",
                "test": "test_indices.json",
                "test500": "test_indices_500.json",
            }[split]
            session_ids = json.loads((repo_path / "data" / filename).read_text(encoding="utf-8"))
        tasks = [
            cls(task_id=f"{split}_{idx}", session_id=int(session_id), split=split, ordinal=idx)
            for idx, session_id in enumerate(session_ids)
        ]
        return tasks, len(tasks)
