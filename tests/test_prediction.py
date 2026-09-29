import tempfile
import unittest
from pathlib import Path

from agents.ocm_agent.knowledge import (
    KnowledgeStore,
    parse_knowledge_updates,
    parse_reflection_plan,
    validate_knowledge_updates,
    validate_reflection_plan,
)
from agents.ocm_agent.prediction import (
    evaluate_observed_transition,
    parse_action_prediction,
)


class PredictionTest(unittest.TestCase):
    def test_prediction_is_bound_to_executed_action_and_known_evidence(self) -> None:
        prediction = parse_action_prediction(
            '{"outcome":"task completes","valid":true,"effect":"complete",'
            '"knowledge":["object.Animal","invented.Rule"]}',
            "focus on crocodile",
            available_knowledge=["object.Animal"],
        )
        self.assertEqual(prediction["action"], "focus on crocodile")
        self.assertEqual(prediction["knowledge"], ["object.Animal"])
        self.assertEqual(prediction["effect"], "complete")

    def test_simple_format_fallback_handles_wrappers_and_missing_delimiters(self) -> None:
        parsed = parse_action_prediction(
            '{"outcome":"door opens","valid":true,"effect":"advance","knowledge":[]',
            "open door",
        )
        self.assertEqual(parsed["outcome"], "door opens")
        wrapped = parse_action_prediction(
            '<tool_call>\n</think>\n'
            '{"outcome":"door opens","valid":true,"effect":"advance","knowledge":[]}',
            "open door",
        )
        self.assertEqual(wrapped["outcome"], "door opens")
        with self.assertRaises(ValueError):
            parse_action_prediction(
                '{"outcome":"unterminated,"valid":true,"effect":"advance"}',
                "open door",
            )

    def test_observation_checks_only_observable_fields(self) -> None:
        result = evaluate_observed_transition(
            {"valid": False, "success": None},
            reward=0.5,
            previous_reward=0.0,
            done=False,
            success=False,
            invalid_action=True,
        )
        self.assertTrue(result["validity_correct"])
        self.assertEqual(result["actual"]["reward_delta"], 0.5)


class KnowledgeStoreTest(unittest.TestCase):
    def test_patch_keeps_helpers_and_applies_helper_only_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            store.object_knowledge_path.write_text("class Door:\n    pass\n\nLIMIT = 1\n")
            plan = {"object_knowledge_updates": [{"target": "Door", "operation": "update"}],
                    "procedural_knowledge_updates": [{"target": "opening", "operation": "create"}]}
            patch = ("*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n"
                     "-LIMIT = 1\n+LIMIT = 2\n+def threshold():\n+    return LIMIT\n"
                     "*** Add File: procedural_knowledge/opening.py\n"
                     "+from object_knowledge import threshold\n+VALUE = threshold()\n*** End Patch")
            objects, procedures = store.updates_from_patch(patch, plan)
            changed = store.commit_knowledge_updates(object_updates=objects, procedural_updates=procedures)
            self.assertEqual(changed["object"], ["Door"])
            self.assertIn("LIMIT = 2", store.object_source())
            self.assertIn("def threshold", store.object_source())
            removal = ("*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n"
                       "-LIMIT = 2\n def threshold():\n-    return LIMIT\n+    return 3\n*** End Patch")
            objects, procedures = store.updates_from_patch(removal, {
                "object_knowledge_updates": plan["object_knowledge_updates"], "procedural_knowledge_updates": []})
            store.commit_knowledge_updates(object_updates=objects, procedural_updates=procedures)
            self.assertNotIn("LIMIT", store.object_source())

    def test_patch_source_includes_module_context_and_reports_syntax_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            source = "import math\n\nLIMIT = math.inf\n\nclass Door:\n    locked = True\n"
            store.object_knowledge_path.write_text(source)
            plan = {"object_knowledge_updates": [{"target": "Door", "operation": "update"}],
                    "procedural_knowledge_updates": []}
            self.assertEqual(source, store.patch_source(plan)["object_knowledge.py"])
            patch = ("*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n"
                     "-    locked = True\n+    locked = (\n*** End Patch")
            with self.assertRaises(SyntaxError):
                store.updates_from_patch(patch, plan)
            self.assertEqual(store.object_source(), source)

    def test_model_context_contains_complete_object_and_procedure_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            store.commit_knowledge_updates(
                object_updates=[
                    {
                        "existing_name": "",
                        "name": "Animal",
                        "code": "class Animal:\n    lifespan_order = [('crocodile', 'rabbit')]\n",
                    }
                ],
                procedural_updates=[
                    {
                        "existing_name": "",
                        "name": "choose_longest",
                        "code": "from object_knowledge import Animal\n\ndef known_pairs():\n    return Animal.lifespan_order\n",
                    }
                ],
            )
            context = store.model_context()
            self.assertIn("('crocodile', 'rabbit')", context)
            self.assertIn("procedural_knowledge/choose_longest.py", context)
            self.assertIn("Knowledge ids: object.Animal", context)
            self.assertIn("Knowledge id: procedural.choose_longest", context)
            self.assertEqual(
                store.knowledge_ids(),
                ["object.Animal", "procedural.choose_longest"],
            )

    def test_plan_and_implementation_validation(self) -> None:
        plan = parse_reflection_plan(
            '{"object_knowledge_updates":[{"target":"Animal","operation":"create",'
            '"lesson":"crocodiles outlive rabbits","evidence":"successful focus"}],'
            '"procedural_knowledge_updates":[]}'
        )
        validate_reflection_plan(
            plan, object_names=set(), procedural_names=set()
        )
        objects, procedures = parse_knowledge_updates(
            '{"object_knowledge":[{"existing_name":"","name":"Animal",'
            '"code":"class Animal:\\n    pass\\n"}],"procedural_knowledge":[]}'
        )
        validate_knowledge_updates(
            plan,
            object_updates=objects,
            procedural_updates=procedures,
        )

    def test_plan_normalizes_procedure_to_file_and_rejects_location_like_class_name(self) -> None:
        plan = parse_reflection_plan(
            '{"object_knowledge_updates":[],"procedural_knowledge_updates":['
            '{"target":"procedural.navigation.recover","operation":"update",'
            '"lesson":"retry with observed syntax","evidence":"invalid then valid"}]}'
        )
        self.assertEqual(
            plan["procedural_knowledge_updates"][0]["target"], "navigation"
        )
        validate_reflection_plan(
            plan, object_names=set(), procedural_names={"navigation"}
        )
        bad = parse_reflection_plan(
            '{"object_knowledge_updates":[{"target":"outside","operation":"create",'
            '"lesson":"stores the room","evidence":"room observation"}],'
            '"procedural_knowledge_updates":[]}'
        )
        with self.assertRaisesRegex(ValueError, "PascalCase"):
            validate_reflection_plan(
                bad, object_names=set(), procedural_names={"navigation"}
            )
if __name__ == "__main__":
    unittest.main()
