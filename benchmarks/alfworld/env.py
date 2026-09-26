import logging
import alfworld.agents.environment

from benchmarks.alfworld.task import AlfWorldTask, load_alfworld_config
from benchmarks.base import BenchmarkEnv, StepResult


logger = logging.getLogger(__name__)


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BenchmarkEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        config = load_alfworld_config()
        env_class = alfworld.agents.environment.get_environment(config["env"]["type"])
        env_manager = env_class(config, train_eval=task.split)
        env_manager.game_files = [task.game_file]
        env_manager.num_games = 1
        self.env = env_manager.init_env(batch_size=1)

    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, action: str) -> StepResult:
        try:
            observation, reward, done = self.conduct_action(action.strip())
        except Exception:
            return StepResult(observation="Invalid action.", reward=0, invalid_action=True)
        success = bool(reward)
        terminate_reason = None
        if done:
            terminate_reason = "success" if success else "done"
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            success=success,
            terminate_reason=terminate_reason,
        )

    def reset(self) -> str:
        observation, _info = self.env.reset()
        initial_observation = "\n".join(observation[0].split("\n\n")[1:])
        self._world_model_goal = next(
            (
                line.strip()
                for line in initial_observation.splitlines()
                if line.strip().lower().startswith("your task is")
            ),
            "",
        )
        return initial_observation

    def world_model_goal(self) -> str:
        return getattr(self, "_world_model_goal", "")

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
