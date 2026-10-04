import time
import unittest
from unittest.mock import Mock

from main import _log_wandb_progress


class WandbProgressTest(unittest.TestCase):
    def test_component_prediction_metrics_log_without_old_validity_field(self):
        run = Mock()
        records = [{"result": {"meta": {"reward": 1.0, "success": True, "actions": 3},
                    "agent_artifact": {"decision_log": [{
                        "prediction": {"transition": {"prediction": "changed", "knowledge": []},
                                       "progress": {"prediction": "goal achieved", "knowledge": []}},
                        "knowledge_available": False,
                        "knowledge_references": [],
                    }]}}}]
        _log_wandb_progress(run, records, benchmark_name="sciworld", total_tasks=450,
                            started_at=time.monotonic())
        payload = run.log.call_args.args[0]
        self.assertEqual(payload["progress/completed_tasks"], 1)
        self.assertEqual(payload["progress/average_reward"], 1.0)
        self.assertEqual(payload["progress/success_rate"], 1.0)
        self.assertEqual(payload["task/predictions"], 1)
        self.assertNotIn("task/validity_prediction_accuracy", payload)
