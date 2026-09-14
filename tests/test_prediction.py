from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.ocm_agent.agent import OCMAgent
from agents.ocm_agent.prediction import parse_prediction_provenance
from benchmarks.base import StepResult


class PredictionProvenanceTest(unittest.TestCase):
    def test_parses_prediction_and_typed_grounding(self) -> None:
        parsed = parse_prediction_provenance(
            "\n".join(
                [
                    "Thought: Heat the water.",
                    "Prediction: The water temperature will increase.",
                    "Grounding: object.Stove.heat, procedural.heating_substance",
                    "Action: turn on stove",
                ]
            )
        )

        self.assertEqual(parsed["prediction"], "The water temperature will increase.")
        self.assertEqual(parsed["object_symbols"], ["object.Stove.heat"])
        self.assertEqual(parsed["procedural_symbols"], ["procedural.heating_substance"])
        self.assertEqual(parsed["unresolved_symbols"], [])

    def test_normalizes_long_prefixes_and_deduplicates(self) -> None:
        parsed = parse_prediction_provenance(
            "Prediction: The container will open.\n"
            "Grounding: object_knowledge.Container.open, object.Container.open; "
            "procedural_knowledge.open_container"
        )

        self.assertEqual(parsed["object_symbols"], ["object.Container.open"])
        self.assertEqual(parsed["procedural_symbols"], ["procedural.open_container"])

    def test_accepts_missing_or_empty_grounding(self) -> None:
        missing = parse_prediction_provenance("Action: look around")
        empty = parse_prediction_provenance(
            "Prediction: The room description will be shown.\nGrounding: none\nAction: look around"
        )

        self.assertEqual(missing["prediction"], "")
        self.assertEqual(missing["object_symbols"], [])
        self.assertEqual(empty["object_symbols"], [])
        self.assertEqual(empty["procedural_symbols"], [])

    def test_preserves_untyped_references_for_audit(self) -> None:
        parsed = parse_prediction_provenance(
            "Prediction: The item will move.\nGrounding: Inventory.move, `object.Item`"
        )

        self.assertEqual(parsed["object_symbols"], ["object.Item"])
        self.assertEqual(parsed["unresolved_symbols"], ["Inventory.move"])

    def test_prediction_reaches_episode_reflection(self) -> None:
        responses = iter(
            [
                "Thought: Open it.\nPrediction: The box will open.\nGrounding: none\nAction: open box",
                '{"object_knowledge_updates": [], "procedural_knowledge_updates": [], '
                '"inspect_objects": [], "inspect_procedural": []}',
                '{"object_knowledge": [], "procedural_knowledge": []}',
            ]
        )
        fake_client = MagicMock()
        fake_client.client = MagicMock()
        fake_client.create_chat_completion.side_effect = lambda **kwargs: SimpleNamespace(content=next(responses))

        with TemporaryDirectory() as tmp, patch(
            "agents.ocm_agent.agent.OpenAIChatClient",
            return_value=fake_client,
        ):
            agent = OCMAgent(
                {
                    "model_name": "gpt-4.1-mini",
                    "api_key": "test",
                    "base_url": "https://example.invalid/v1",
                    "output_dir": Path(tmp),
                }
            )
            agent.on_episode_start(benchmark_name="test", observation="A closed box is here.")
            action = agent.act("A closed box is here.")
            agent.on_step(
                step_idx=1,
                observation="A closed box is here.",
                action=action,
                outcome=StepResult("The box is locked.", invalid_action=True),
                next_observation="The box is locked.",
            )
            artifact = agent.on_episode_end({"success": False})

        self.assertEqual(action, "open box")
        step = artifact["decision_log"][0]
        self.assertEqual(step["prediction"], "The box will open.")
        self.assertEqual(step["next_observation"], "The box is locked.")
        reflection_call = fake_client.create_chat_completion.call_args_list[1]
        reflection_payload = reflection_call.kwargs["messages"][1]["content"]
        self.assertIn('"prediction": "The box will open."', reflection_payload)
        self.assertIn('"next_observation": "The box is locked."', reflection_payload)


if __name__ == "__main__":
    unittest.main()
