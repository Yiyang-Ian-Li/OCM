import logging

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH as package_jar_path

from benchmarks.base import BenchmarkEnv, StepResult
from benchmarks.sciworld.process import close_scienceworld_env, make_scienceworld_env
from benchmarks.sciworld.task import SciWorldTask


logger = logging.getLogger(__name__)
DEFAULT_SIMPLIFICATION = ""


class SciWorldEnv(BenchmarkEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv | None = None,
        simplification_str: str = DEFAULT_SIMPLIFICATION,
        env_step_limit: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env_step_limit = int(env_step_limit)
        self.max_steps = self.env_step_limit
        self.uses_native_step_limit = True
        self.env = env or make_scienceworld_env("", serverPath=package_jar_path, envStepLimit=self.env_step_limit)
        self.simplification_str = simplification_str

    def step(self, action: str) -> StepResult:
        try:
            observation, _, done, info = self.env.step(action.strip())
            reward = info["score"]
        except AssertionError:
            return StepResult(observation="Invalid action.", invalid_action=True, terminate_reason="invalid_action")
        invalid_action = "No known action matches that input." in observation
        success = info.get("score", 0) >= 100
        terminate_reason = None
        if done:
            if success:
                terminate_reason = "success"
            elif info.get("score", 0) < 0:
                terminate_reason = "negative_score"
            else:
                terminate_reason = "env_done"
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            success=success,
            invalid_action=invalid_action,
            terminate_reason=terminate_reason,
            info=info,
        )

    def reset(self) -> str:
        self.env.load(
            self.task.sub_task_name,
            self.task.variation_idx,
            simplificationStr=self.simplification_str,
            generateGoldPath=False,
        )
        observation, info = self.env.reset()
        return f"{info['taskDesc']}\n\n{observation}"

    def close(self) -> None:
        close_scienceworld_env(self.env)
