from __future__ import annotations

from typing import Any


class TrialTask:
    def __init__(self, task, trial_index: int):
        self.source_task = task
        self.trial_index = int(trial_index)
        self.task_id = f"{task.task_id}__trial_{self.trial_index}"

    def __getattr__(self, name):
        return getattr(self.source_task, name)

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self.source_task.metadata())
        metadata["base_task_id"] = metadata.get("task_id")
        metadata["task_id"] = self.task_id
        metadata["trial"] = self.trial_index
        return metadata


def build_record(task, result: dict[str, Any]) -> dict[str, Any]:
    return {"task": task.metadata(), "result": result}


def apply_task_limit(all_tasks, task_limit: int | None) -> list:
    tasks = list(all_tasks)
    if task_limit is None:
        return tasks
    return tasks[: max(0, int(task_limit))]


def expand_trials(tasks, num_trials: int) -> list[TrialTask]:
    return [
        TrialTask(task, trial_index)
        for task in tasks
        for trial_index in range(max(1, int(num_trials)))
    ]
