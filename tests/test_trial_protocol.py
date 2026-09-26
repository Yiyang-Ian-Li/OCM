from __future__ import annotations

import unittest

from benchmarks.sciworld.task import SciWorldTask
from runner.summary import summarize_records
from runner.tasks import expand_trials


def _record(task, *, reward: float, success: bool) -> dict:
    return {
        "task": task.metadata(),
        "result": {
            "meta": {
                "reward": reward,
                "success": success,
                "steps": 3,
                "invalid_actions": 0,
                "terminate_reason": "success" if success else "max_steps",
                "token_usage": None,
            }
        },
    }


class TrialProtocolTest(unittest.TestCase):
    def test_expands_trials_in_task_major_order(self) -> None:
        tasks = [
            SciWorldTask(task_id="a", sub_task_name="boil", variation_idx=1),
            SciWorldTask(task_id="b", sub_task_name="freeze", variation_idx=2),
        ]

        expanded = expand_trials(tasks, 3)

        self.assertEqual(
            [task.task_id for task in expanded],
            [
                "a__trial_0",
                "a__trial_1",
                "a__trial_2",
                "b__trial_0",
                "b__trial_1",
                "b__trial_2",
            ],
        )
        self.assertEqual(expanded[0].metadata()["base_task_id"], "a")

    def test_summary_uses_best_of_trials_and_reports_success_at_one(self) -> None:
        tasks = expand_trials(
            [
                SciWorldTask(task_id="a", sub_task_name="boil", variation_idx=1),
                SciWorldTask(task_id="b", sub_task_name="freeze", variation_idx=2),
            ],
            2,
        )
        records = [
            _record(tasks[0], reward=0.5, success=False),
            _record(tasks[1], reward=1.0, success=True),
            _record(tasks[2], reward=0.25, success=False),
            _record(tasks[3], reward=0.75, success=False),
        ]

        summary = summarize_records("sciworld", records, num_trials=2)

        self.assertEqual(summary["num_tasks"], 2)
        self.assertEqual(summary["num_trial_records"], 4)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["success_at_1"], 0.0)
        self.assertEqual(summary["average_reward"], 0.875)
        self.assertEqual(summary["cumulative_by_trial"][1]["success_rate"], 0.5)

    def test_summary_reports_ocm_mechanism_activity(self) -> None:
        task = SciWorldTask(task_id="a", sub_task_name="boil", variation_idx=1)
        record = _record(task, reward=0.5, success=False)
        record["result"]["agent_artifact"] = {
            "decision_log": [
                {
                    "prediction": {"action": "open container", "outcome": "opens"},
                    "knowledge_available": True,
                    "knowledge_references": ["object.Container"],
                    "validity_correct": True,
                },
                {
                    "prediction": {"action": "open door", "outcome": "opens"},
                    "knowledge_available": True,
                    "knowledge_references": [],
                    "validity_correct": False,
                },
            ],
            "online_update": {
                "planned_updates": 1,
                "written_knowledge": {"object": ["Container"], "procedural": []},
            },
        }

        metrics = summarize_records("sciworld", [record])["agent_metrics"]

        self.assertEqual(metrics["predictions"], 2)
        self.assertEqual(metrics["prediction_coverage"], 1.0)
        self.assertEqual(metrics["memory_referenced_decisions"], 1)
        self.assertEqual(metrics["validity_prediction_accuracy"], 0.5)
        self.assertEqual(metrics["planned_knowledge_updates"], 1)
        self.assertEqual(metrics["written_knowledge_updates"], 1)


if __name__ == "__main__":
    unittest.main()
