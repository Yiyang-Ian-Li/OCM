from __future__ import annotations

import logging
from typing import Iterable

from benchmarks.base import Task


logger = logging.getLogger(__name__)
VALID_SPLITS = {
    "train",
    "val",
    "val.small",
    "val.small.easy",
    "val.repeated",
    "test",
    "test.small",
    "test.small.easy",
}


class PlancraftTask(Task):
    task_name = "plancraft"

    def __init__(self, example, **kwargs):
        super().__init__(**kwargs)
        self.example = example

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "target": self.example.target,
            "impossible": self.example.impossible,
            "complexity": self.example.complexity,
            "complexity_bin": self.example.complexity_bin,
            "complexity_split": self.example.complexity_split,
            "recipe_type": self.example.recipe_type,
            "optimal_path_length": self.example.optimal_path_length,
            "unseen_in_train": self.example.unseen_in_train,
            "unseen_in_val": self.example.unseen_in_val,
            "split": self.example.split,
        }

    @classmethod
    def load_tasks(cls, split: str) -> tuple[Iterable[Task], int]:
        from plancraft.simple import get_plancraft_examples

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Unsupported PlanCraft split: {split}. "
                f"Expected one of: {', '.join(sorted(VALID_SPLITS))}."
            )
        examples = get_plancraft_examples(split)

        def generator():
            for example in examples:
                yield cls(task_id=example.id, example=example)

        return generator(), len(examples)
