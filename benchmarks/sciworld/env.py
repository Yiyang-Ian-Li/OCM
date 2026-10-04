import logging
import re

from scienceworld import ScienceWorldEnv
from scienceworld.constants import JAR_PATH as package_jar_path

from benchmarks.base import BenchmarkEnv, StepResult
from benchmarks.sciworld.agentboard import PROTOCOL, CorrectedAgentBoard, read_event
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
        agent_step_limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env_step_limit = int(env_step_limit)
        self.max_steps = int(agent_step_limit or self.env_step_limit)
        self.uses_native_step_limit = True
        self.env = env or make_scienceworld_env("", serverPath=package_jar_path, envStepLimit=self.env_step_limit)
        self.simplification_str = simplification_str
        self._agentboard_progress: list[bool] = []
        self._corrected_scorer: CorrectedAgentBoard | None = None

    def step(self, action: str) -> StepResult:
        if self.task.subgoals:
            return self._agentboard_step(action)
        try:
            observation, _, done, info = self.env.step(action.strip())
            reward = info["score"] / 100.0
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
        self._agentboard_progress = [False] * len(self.task.subgoals)
        if self.task.evaluation_protocol == PROTOCOL:
            self._corrected_scorer = CorrectedAgentBoard.from_env(self.task, self.env)
            self._agentboard_progress = self._corrected_scorer.progress
        self._world_model_goal = str(self.task.goal_override or info["taskDesc"])
        return f"{self._world_model_goal}\n\n{observation}"

    def world_model_state(self, observation: str) -> str:
        state_observation = str(observation).strip()
        goal = self.world_model_goal().strip()
        if goal and state_observation.startswith(goal):
            state_observation = state_observation[len(goal) :].lstrip()
        return f"{state_observation}\n{self.env.inventory()}".strip()

    def world_model_goal(self) -> str:
        return getattr(self, "_world_model_goal", "")

    def close(self) -> None:
        close_scienceworld_env(self.env)

    def _agentboard_step(self, action: str) -> StepResult:
        normalized = action.strip()
        agent = None
        if self._corrected_scorer is not None and self._corrected_scorer.semantic:
            agent = self.env.server.agentInterface().get().agent()
            history_size = agent.getActionHistorySize()
        try:
            observation, _, _, info = self.env.step(normalized)
        except AssertionError:
            return StepResult(
                observation="Invalid action.",
                reward=self._agentboard_reward(),
                invalid_action=True,
                terminate_reason="invalid_action",
            )
        if self._corrected_scorer is not None and observation.startswith("Task description:\n"):
            observation = "Task description:\n" + self.world_model_goal()
        if self._corrected_scorer is not None:
            event = read_event(agent, history_size) if agent is not None else None
            self._corrected_scorer.update(observation, event)
        else:
            for index, pattern in enumerate(self.task.subgoals):
                if re.search(pattern, observation):
                    self._agentboard_progress[index] = True
        reward = self._agentboard_reward()
        done = all(self._agentboard_progress)
        invalid_action = "No known action matches that input." in observation
        protocol_info = dict(info or {})
        protocol_info.pop("moves", None)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            success=done,
            invalid_action=invalid_action,
            terminate_reason="success" if done else None,
            info=protocol_info,
        )

    def _agentboard_reward(self) -> float:
        return sum(self._agentboard_progress) / len(self._agentboard_progress)
