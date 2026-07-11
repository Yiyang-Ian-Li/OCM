from __future__ import annotations

import logging

from benchmarks.base import BenchmarkEnv, StepResult
from benchmarks.plancraft.task import PlancraftTask


logger = logging.getLogger(__name__)

INVALID_PREFIXES = (
    "Only select actions from the following:",
    "Format Error:",
)


class PlancraftEnv(BenchmarkEnv):
    def __init__(
        self,
        task: PlancraftTask,
        resolution: str = "low",
        use_text_inventory: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from plancraft.environment.actions import ImpossibleActionHandler, MoveActionHandler, SmeltActionHandler
        from plancraft.environment.search import GoldSearchActionHandler
        from plancraft.simple import PlancraftGymWrapper

        self.task = task
        self.env = PlancraftGymWrapper(
            example=task.example,
            actions=[GoldSearchActionHandler(), MoveActionHandler(), SmeltActionHandler(), ImpossibleActionHandler()],
            max_steps=self.max_steps,
            resolution=resolution,
            use_text_inventory=use_text_inventory,
        )

    def reset(self) -> str:
        observation, _, _, _, _ = self.env.step("")
        return str(observation.get("text", ""))

    def step(self, action: str) -> StepResult:
        observation, reward, terminated, truncated, info = self.env.step(action.strip())
        text = str(observation.get("text", ""))
        invalid_action = text.startswith(INVALID_PREFIXES)
        done = bool(terminated or truncated)
        success = bool(reward and reward > 0)
        terminate_reason = None
        if done:
            terminate_reason = info.get("reason") or ("success" if success else "done")
        return StepResult(
            observation=text,
            reward=reward,
            done=done,
            success=success,
            invalid_action=invalid_action,
            terminate_reason=terminate_reason,
            info=info,
        )

    def close(self) -> None:
        return None
