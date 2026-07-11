from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from benchmarks.base import Task
from benchmarks.external_repos import RAW_BENCHMARKS_DIR, ensure_git_repo


INTERCODE_REPO_URL = "https://github.com/princeton-nlp/intercode.git"
DEFAULT_INTERCODE_REPO_PATH = RAW_BENCHMARKS_DIR / "intercode"

SPLITS = {
    "bash": [
        ("bash_fs1", "bash", "data/nl2bash/nl2bash_fs_1.json"),
        ("bash_fs2", "bash", "data/nl2bash/nl2bash_fs_2.json"),
        ("bash_fs3", "bash", "data/nl2bash/nl2bash_fs_3.json"),
        ("bash_fs4", "bash", "data/nl2bash/nl2bash_fs_4.json"),
    ],
    "bash_fs1": [("bash_fs1", "bash", "data/nl2bash/nl2bash_fs_1.json")],
    "bash_fs2": [("bash_fs2", "bash", "data/nl2bash/nl2bash_fs_2.json")],
    "bash_fs3": [("bash_fs3", "bash", "data/nl2bash/nl2bash_fs_3.json")],
    "bash_fs4": [("bash_fs4", "bash", "data/nl2bash/nl2bash_fs_4.json")],
    "bash_quickstart": [("bash_quickstart", "bash", "data/nl2bash/test_queries.json")],
}


def resolve_intercode_repo(path: str | Path | None = None) -> Path:
    return ensure_git_repo(
        Path(path).expanduser() if path else DEFAULT_INTERCODE_REPO_PATH,
        INTERCODE_REPO_URL,
        "intercode",
    )


class InterCodeTask(Task):
    task_name = "intercode"

    def __init__(self, split: str, family: str, index: int, data_path: Path, record: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.split = split
        self.family = family
        self.index = int(index)
        self.data_path = Path(data_path)
        self.record = dict(record)

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "split": self.split,
            "family": self.family,
            "index": self.index,
            "query": self.record.get("query"),
            "db": self.record.get("db"),
            "difficulty": self.record.get("difficulty") or self.record.get("hardness"),
        }

    @classmethod
    def load_tasks(cls, split: str) -> tuple[Iterable[Task], int]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported InterCode split: {split}. Expected one of: {', '.join(sorted(SPLITS))}.")
        repo_path = resolve_intercode_repo()
        tasks = []
        for split_part, family, rel_path in SPLITS[split]:
            data_path = repo_path / rel_path
            data = pd.read_json(data_path)
            tasks.extend(
                cls(
                    task_id=f"{split_part}_{idx}",
                    split=split,
                    family=family,
                    index=idx,
                    data_path=data_path,
                    record=row.dropna().to_dict(),
                )
                for idx, row in data.iterrows()
            )
        return tasks, len(tasks)
