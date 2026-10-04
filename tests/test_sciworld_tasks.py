from __future__ import annotations

import unittest
import tempfile
from argparse import Namespace
from unittest.mock import Mock, patch

from benchmarks.sciworld.env import SciWorldEnv
from benchmarks.sciworld.task import SciWorldTask
from config import DEFAULT_CONFIG
from main import run
from runner.summary import summarize_records
from runner.tasks import expand_trials


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

    def test_native_split_preserves_selection_and_order_without_regex_goals(self) -> None:
        old_tasks, old_count = SciWorldTask.load_tasks("agentboard_test")
        new_tasks, new_count = SciWorldTask.load_tasks("agentboard_native")
        old_tasks, new_tasks = list(old_tasks), list(new_tasks)
        self.assertEqual(new_count, 90)
        self.assertEqual(new_count, old_count)
        self.assertEqual(
            [(t.sub_task_name, t.variation_idx) for t in new_tasks],
            [(t.sub_task_name, t.variation_idx) for t in old_tasks],
        )
        self.assertTrue(all(t.goal_override is None and t.subgoals == [] for t in new_tasks))
        self.assertTrue(all(t.metadata()["evaluation_protocol"] == "native" for t in new_tasks))
        self.assertTrue(all(t.metadata()["evaluation_protocol"] == "agentboard" for t in old_tasks))
        self.assertTrue({t.task_id for t in new_tasks}.isdisjoint(t.task_id for t in old_tasks))
        trials = expand_trials(new_tasks, 5)
        self.assertEqual(len(trials), 450)
        self.assertEqual(trials[0].task_id, "agentboard_native_0000__trial_0")
        self.assertEqual(trials[4].trial_index, 4)
        self.assertEqual(trials[5].source_task, new_tasks[1])

    def test_native_score_and_terminal_feedback(self) -> None:
        tasks, _ = SciWorldTask.load_tasks("agentboard_native")
        task = list(tasks)[34]
        cases = [
            (25, False, False, None),
            (100, True, True, "success"),
            (-100, True, False, "negative_score"),
            (40, True, False, "env_done"),
        ]
        for score, done, success, reason in cases:
            with self.subTest(score=score):
                fake = _FakeScienceWorldEnv()
                fake.step = Mock(return_value=("native observation", 7, done, {"score": score, "moves": 3}))
                environment = SciWorldEnv(task, env=fake, agent_step_limit=30)
                initial = environment.reset()
                self.assertTrue(initial.startswith("Task Description:\nBoil lead."))
                self.assertEqual(environment.world_model_goal(), "Task Description:\nBoil lead.")
                outcome = environment.step("look around")
                self.assertEqual(outcome.reward, score / 100)
                self.assertEqual(outcome.done, done)
                self.assertEqual(outcome.success, success)
                self.assertEqual(outcome.terminate_reason, reason)
                self.assertEqual(outcome.info["score"], score)
                self.assertEqual(outcome.info["moves"], 3)
                self.assertEqual(environment._agentboard_progress, [])
                record = {"task": task.metadata(), "result": {"meta": {
                    "reward": outcome.reward, "success": success, "steps": 3,
                    "invalid_actions": 0, "terminate_reason": reason,
                }}}
                summary = summarize_records("sciworld", [record])
                self.assertEqual(summary["average_reward"], max(0, score / 100))
                self.assertEqual(summary["success_rate"], float(success))

    def test_runner_selects_final_score_only_for_native_sampled_protocol(self) -> None:
        for split, expected_mode in (("agentboard_native", "latest"), ("agentboard_test", "max")):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as tmp:
                config = dict(DEFAULT_CONFIG)
                config.update(benchmark="sciworld", split=split, task_limit=1,
                              num_trials=1, output_dir=tmp, run_name="protocol-test")
                result = {"meta": {"reward": -1.0, "success": False, "steps": 2,
                                   "invalid_actions": 0, "terminate_reason": "negative_score"}}
                with patch("main.build_agent", return_value=object()), \
                     patch("main.configure_logging"), \
                     patch("main.interactive_loop", return_value=result) as loop:
                    summary = run(Namespace(**config))
                self.assertEqual(loop.call_args.kwargs["reward_mode"], expected_mode)
                self.assertEqual(summary["average_reward"], 0.0)
                self.assertEqual(summary["success_rate"], 0.0)

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
