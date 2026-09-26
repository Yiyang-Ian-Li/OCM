import logging
import json
from pathlib import Path

from scienceworld.constants import JAR_PATH, TASKS

from benchmarks.base import Task
from benchmarks.sciworld.process import close_scienceworld_env, make_scienceworld_env

logger = logging.getLogger(__name__)


TASK_NAME_TO_ID = {item["task_name"]: item["task_id"] for item in TASKS}
DEFAULT_SIMPLIFICATION = ""


class SciWorldTask(Task):
    """ScienceWorld task instance."""
    
    task_name = "sciworld"
    
    def __init__(
        self,
        sub_task_name: str,
        variation_idx: int,
        goal_override: str | None = None,
        subgoals: list[str] | None = None,
        difficulty: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name
        self.variation_idx = variation_idx
        self.goal_override = goal_override
        self.subgoals = list(subgoals or [])
        self.difficulty = difficulty

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "sub_task_name": self.sub_task_name,
            "variation_idx": self.variation_idx,
            "goal_override": self.goal_override,
            "subgoals": self.subgoals,
            "difficulty": self.difficulty,
        }
    
    @classmethod
    def load_tasks(cls, split: str):
        if split == "agentboard_test":
            return cls._load_agentboard_tasks()
        env = make_scienceworld_env("", serverPath=JAR_PATH, envStepLimit=1)
        try:
            task_indices: list[tuple[str, int]] = []
            for task_name in env.getTaskNames():
                env.load(task_name, 0, simplificationStr=DEFAULT_SIMPLIFICATION, generateGoldPath=False)
                if split == "train":
                    variations = list(env.getVariationsTrain())
                elif split == "dev":
                    variations = list(env.getVariationsDev())
                elif split in {"test", "test_full"}:
                    variations = list(env.getVariationsTest())
                    if split == "test":
                        variations = variations[:5]
                else:
                    raise ValueError(f"Unsupported SciWorld split: {split}")
                task_indices.extend((task_name, int(variation_idx)) for variation_idx in variations)
        finally:
            close_scienceworld_env(env)

        def generator():
            for task_name, variation_idx in task_indices:
                task_id = TASK_NAME_TO_ID.get(task_name, task_name)
                yield cls(
                    task_id=f"{task_id}_{variation_idx}",
                    sub_task_name=task_name,
                    variation_idx=variation_idx,
                )

        return generator(), len(task_indices)

    @classmethod
    def _load_agentboard_tasks(cls):
        label_path = (
            Path(__file__).resolve().parents[2]
            / "baselines"
            / "WorldEvolver"
            / "data"
            / "scienceworld"
            / "test.jsonl"
        )
        labels = [json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        def generator():
            for item in labels:
                info = item["additional_info"]
                yield cls(
                    task_id=f"agentboard_{int(item['id']):04d}",
                    sub_task_name=str(info["env_name"]),
                    variation_idx=int(info["var"]),
                    goal_override=str(item["goal"]),
                    subgoals=[str(value) for value in item["subgoals"]],
                    difficulty=str(item["difficulty"]),
                )

        return generator(), len(labels)
    
