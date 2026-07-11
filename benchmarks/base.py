from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

class Task(ABC):
    task_name: str = "base"

    def __init__(self, **kwargs) -> None:
        self.task_id: Any = kwargs.get("task_id")

    def metadata(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "task_name": self.task_name}

    @classmethod
    @abstractmethod
    def load_tasks(cls, split: str) -> Tuple[Sequence["Task"], int]:
        raise NotImplementedError


class BenchmarkEnv(ABC):
    def __init__(
        self,
        max_steps: int = 10,
        **kwargs,
    ) -> None:
        self.max_steps = max_steps

    @abstractmethod
    def step(self, action: str) -> "StepResult":
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> str:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class StepResult:
    observation: str
    reward: float | None = None
    done: bool = False
    success: bool | None = None
    invalid_action: bool = False
    terminate_reason: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
