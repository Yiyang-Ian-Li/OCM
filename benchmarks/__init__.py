from __future__ import annotations

from benchmarks.alfworld.env import AlfWorldEnv
from benchmarks.alfworld.metrics import summarize_alfworld
from benchmarks.alfworld.task import AlfWorldTask
from benchmarks.plancraft.env import PlancraftEnv
from benchmarks.plancraft.metrics import summarize_plancraft
from benchmarks.plancraft.task import PlancraftTask
from benchmarks.sciworld.env import SciWorldEnv
from benchmarks.sciworld.metrics import summarize_sciworld
from benchmarks.sciworld.task import SciWorldTask
from benchmarks.webshop.env import WebShopEnv
from benchmarks.webshop.metrics import summarize_webshop
from benchmarks.webshop.task import WebShopTask


BENCHMARKS = {
    "alfworld": {
        "task_class": AlfWorldTask,
        "env_class": AlfWorldEnv,
        "reward_mode": "latest",
        "summarize": summarize_alfworld,
    },
    "sciworld": {
        "task_class": SciWorldTask,
        "env_class": SciWorldEnv,
        "reward_mode": "max",
        "summarize": summarize_sciworld,
    },
    "plancraft": {
        "task_class": PlancraftTask,
        "env_class": PlancraftEnv,
        "reward_mode": "latest",
        "summarize": summarize_plancraft,
    },
    "webshop": {
        "task_class": WebShopTask,
        "env_class": WebShopEnv,
        "reward_mode": "latest",
        "summarize": summarize_webshop,
    },
}

__all__ = [
    "BENCHMARKS",
    "AlfWorldEnv",
    "AlfWorldTask",
    "PlancraftEnv",
    "PlancraftTask",
    "SciWorldEnv",
    "SciWorldTask",
    "WebShopEnv",
    "WebShopTask",
]
