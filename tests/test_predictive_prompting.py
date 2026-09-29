import json
import unittest

from agents.ocm_agent.prompting import (
    knowledge_update_messages,
    policy_messages,
    prediction_messages,
    reflection_messages,
)


class PredictivePromptingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = [
            {"role": "user", "content": "initial state"},
            {"role": "assistant", "content": "Thought: inspect\nAction: look around"},
            {"role": "user", "content": "a rabbit and a crocodile"},
        ]

    def test_policy_uses_memory_and_prediction_residuals(self) -> None:
        messages = policy_messages(
            benchmark_instruction="Use primitive actions.",
            benchmark_examples="Thought: x\nAction: y",
            knowledge_context="class Animal:\n    longer = [('crocodile', 'rabbit')]",
            observation="a rabbit and a crocodile",
            trajectory=self.trajectory,
        )
        self.assertIn("fallible prior", messages[0]["content"])
        self.assertIn("following environment observation as ground truth", messages[0]["content"])
        self.assertIn("('crocodile', 'rabbit')", messages[0]["content"])
        self.assertEqual(messages[-1], self.trajectory[-1])

    def test_world_model_predicts_the_proposed_action_with_full_memory(self) -> None:
        messages = prediction_messages(
            benchmark_instruction="Use primitive actions.",
            knowledge_context="class Animal:\n    longer = [('crocodile', 'rabbit')]",
            observation="a rabbit and a crocodile",
            trajectory=self.trajectory,
            action="focus on crocodile",
        )
        system = messages[0]["content"]
        self.assertIn("explicit world model", system)
        self.assertIn("('crocodile', 'rabbit')", system)
        self.assertIn("prediction is a hypothesis", system)
        self.assertEqual(
            json.loads(messages[-1]["content"].split("\n", 1)[1]),
            {"action": "focus on crocodile"},
        )

    def test_reflection_receives_only_executed_transition_as_evidence(self) -> None:
        messages = reflection_messages(
            benchmark_name="sciworld",
            task_context="Find the animal with the longest life span.",
            object_signatures="# empty",
            procedural_index="PROCEDURAL_KNOWLEDGE = {}",
            steps=[
                {
                    "step": 3,
                    "observation": "a rabbit and a crocodile",
                    "prediction": {"action": "focus rabbit", "outcome": "task completes"},
                    "action": "focus rabbit",
                    "next_observation": "not complete",
                    "actual": {"success": False, "reward_delta": 0.0, "invalid_action": False},
                },
                {
                    "step": 4,
                    "observation": "not complete",
                    "prediction": {"action": "focus crocodile", "outcome": "task completes"},
                    "action": "focus crocodile",
                    "next_observation": "task complete",
                    "actual": {"success": True, "reward_delta": 0.5, "invalid_action": False},
                }
            ],
            result_meta={"success": True},
        )
        self.assertIn("prediction is a hypothesis", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        trace = payload["executed_prediction_traces"][0]
        self.assertEqual(trace["prediction"]["outcome"], "task completes")
        self.assertEqual(trace["action"], "focus rabbit")
        self.assertNotIn("executed_outcome_contrasts", payload)
        traces = payload["executed_prediction_traces"]
        self.assertEqual(traces[0]["observation"], "a rabbit and a crocodile")
        self.assertNotIn("observation", traces[1])
        self.assertEqual([t["next_observation"] for t in traces], ["not complete", "task complete"])
        self.assertEqual([t["actual"]["success"] for t in traces], [False, True])

    def test_empty_reflection_trajectory(self) -> None:
        messages = reflection_messages(
            benchmark_name="test", task_context="initial state", object_signatures="",
            procedural_index="", steps=[], result_meta={})
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["executed_prediction_traces"], [])
        self.assertEqual(payload["task_context"], "initial state")

    def test_update_prompt_implements_only_plan(self) -> None:
        messages = knowledge_update_messages(
            object_signatures="# empty",
            procedural_index="PROCEDURAL_KNOWLEDGE = {}",
            update_plan={"object_knowledge_updates": [], "procedural_knowledge_updates": []},
            editable_source="",
        )
        self.assertIn("Implement exactly", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
