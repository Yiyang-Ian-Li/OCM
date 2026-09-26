from __future__ import annotations

import unittest
from unittest.mock import patch

from benchmarks.sciworld.env import SciWorldEnv
from benchmarks.sciworld.task import SciWorldTask


class _FakeScienceWorldEnv:
    def __init__(self):
        self._responses = []

    def getTaskNames(self):
        return ["boil"]

    def load(self, *args, **kwargs):
        return None

    def getVariationsTrain(self):
        return [1, 2]

    def getVariationsDev(self):
        return [3]

    def getVariationsTest(self):
        return [21, 22, 23, 24, 25, 26, 27]

    def reset(self):
        return "A room is visible.", {"taskDesc": "Task Description:\nBoil lead."}

    def step(self, action):
        return self._responses.pop(0), 0, False, {}

    def getValidActionObjectCombinationsWithTemplates(self):
        return [{"action": "open box"}]

    def inventory(self):
        return "Inventory: a paper cup"


class SciWorldTaskTest(unittest.TestCase):
    def test_test_split_preserves_legacy_five_variation_protocol(self) -> None:
        with patch("benchmarks.sciworld.task.make_scienceworld_env", return_value=_FakeScienceWorldEnv()), patch(
            "benchmarks.sciworld.task.close_scienceworld_env"
        ):
            tasks, count = SciWorldTask.load_tasks("test")

        self.assertEqual(count, 5)
        self.assertEqual([task.variation_idx for task in tasks], [21, 22, 23, 24, 25])

    def test_test_full_split_keeps_every_variation(self) -> None:
        with patch("benchmarks.sciworld.task.make_scienceworld_env", return_value=_FakeScienceWorldEnv()), patch(
            "benchmarks.sciworld.task.close_scienceworld_env"
        ):
            tasks, count = SciWorldTask.load_tasks("test_full")

        self.assertEqual(count, 7)
        self.assertEqual([task.variation_idx for task in tasks], [21, 22, 23, 24, 25, 26, 27])

    def test_world_model_context_matches_worldevolver_inputs(self) -> None:
        task = SciWorldTask(task_id="1_21", sub_task_name="boil", variation_idx=21)
        environment = SciWorldEnv(task, env=_FakeScienceWorldEnv())

        observation = environment.reset()

        self.assertEqual(environment.world_model_goal(), "Task Description:\nBoil lead.")
        self.assertEqual(
            environment.world_model_state(observation),
            "A room is visible.\nInventory: a paper cup",
        )

    def test_agentboard_split_loads_release_tasks(self) -> None:
        tasks, count = SciWorldTask.load_tasks("agentboard_test")
        tasks = list(tasks)

        self.assertEqual(count, 90)
        self.assertEqual(tasks[0].task_id, "agentboard_0000")
        self.assertEqual(tasks[0].sub_task_name, "lifespan-longest-lived")
        self.assertTrue(tasks[0].goal_override)
        self.assertTrue(tasks[0].subgoals)

    def test_agentboard_reward_tracks_release_subgoals(self) -> None:
        fake = _FakeScienceWorldEnv()
        fake._responses = [
            "You move to the kitchen.",
            "The box is open.",
        ]
        task = SciWorldTask(
            task_id="agentboard_0000",
            sub_task_name="boil",
            variation_idx=1,
            goal_override="Open the box in the kitchen.",
            subgoals=["You move to the kitchen.", "The box is open."],
            difficulty="easy",
        )
        environment = SciWorldEnv(task, env=fake, agent_step_limit=30)

        initial = environment.reset()
        first = environment.step("go to kitchen")
        second = environment.step("open box")

        self.assertTrue(initial.startswith("Open the box in the kitchen."))
        self.assertEqual(environment.max_steps, 30)
        self.assertEqual(first.reward, 0.5)
        self.assertFalse(first.done)
        self.assertEqual(second.reward, 1.0)
        self.assertTrue(second.success)


if __name__ == "__main__":
    unittest.main()
