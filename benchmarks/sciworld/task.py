import logging

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name
        self.variation_idx = variation_idx

    def metadata(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "sub_task_name": self.sub_task_name,
            "variation_idx": self.variation_idx,
        }
    
    @classmethod
    def load_tasks(cls, split: str):
        env = make_scienceworld_env("", serverPath=JAR_PATH, envStepLimit=1)
        try:
            task_indices: list[tuple[str, int]] = []
            for task_name in env.getTaskNames():
                env.load(task_name, 0, simplificationStr=DEFAULT_SIMPLIFICATION, generateGoldPath=False)
                if split == "train":
                    variations = list(env.getVariationsTrain())
                elif split == "dev":
                    variations = list(env.getVariationsDev())
                elif split == "test":
                    variations = list(env.getVariationsTest())
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
    
