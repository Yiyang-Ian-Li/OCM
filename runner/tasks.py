from __future__ import annotations

from typing import Any


def build_record(task, result: dict[str, Any]) -> dict[str, Any]:
    return {"task": task.metadata(), "result": result}


def apply_task_limit(all_tasks, task_limit: int | None) -> list:
    tasks = list(all_tasks)
    if task_limit is None:
        return tasks
    return tasks[: max(0, int(task_limit))]
