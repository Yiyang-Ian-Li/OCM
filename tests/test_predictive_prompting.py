from __future__ import annotations

import json
import unittest

from agents.ocm_agent.prompting import action_messages, plain_action_messages, reflection_messages


class PredictivePromptingTest(unittest.TestCase):
    def test_action_prompt_requests_prediction_and_grounding(self) -> None:
        messages = action_messages(
            benchmark_instruction="",
            object_signatures="class Container",
            procedural_index="PROCEDURAL_KNOWLEDGE = {}",
            expanded_source="",
            observation="A closed box is here.",
            trajectory=None,
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Prediction: <immediate observable outcome", system_prompt)
        self.assertIn("Grounding: <comma-separated object.* and procedural.* symbols", system_prompt)

    def test_plain_prompt_still_commits_to_prediction(self) -> None:
        messages = plain_action_messages(
            benchmark_instruction="",
            observation="You are in a room.",
            trajectory=None,
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Prediction: <immediate observable outcome", system_prompt)
        self.assertIn("Grounding: none", system_prompt)

    def test_reflection_receives_pre_action_prediction_record(self) -> None:
        step = {
            "action": "open box",
            "prediction": "The box will open.",
            "object_symbols": ["object.Container.open"],
            "procedural_symbols": ["procedural.open_container"],
            "next_observation": "The box is locked.",
        }
        messages = reflection_messages(
            benchmark_name="test",
            object_signatures="class Container",
            procedural_index="PROCEDURAL_KNOWLEDGE = {}",
            steps=[step],
            result_meta={"success": False},
        )

        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["episode_steps"][0], step)
        content_rules = payload["rules"]["content"]
        self.assertTrue(any("prediction" in rule.lower() for rule in content_rules))
        selection_rules = payload["rules"]["selection"]
        self.assertTrue(any("object knowledge" in rule.lower() for rule in selection_rules))
        self.assertTrue(any("procedural knowledge" in rule.lower() for rule in selection_rules))


if __name__ == "__main__":
    unittest.main()
