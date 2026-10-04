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
    def test_components_keep_separate_references(self) -> None:
        import json
        raw = json.dumps({
            "transition": {"prediction": "door opens", "knowledge": ["procedural.opening", "unknown"]},
            "progress": {"prediction": "route becomes accessible", "knowledge": ["object.Door", "object.Door"]},
        })
        result = parse_action_prediction(raw, "open door", available_knowledge=["procedural.opening", "object.Door"])
        self.assertEqual(result["transition"]["knowledge"], ["procedural.opening"])
        self.assertEqual(result["progress"]["knowledge"], ["object.Door"])
        self.assertEqual(set(result), {"action", "transition", "progress"})

    def test_simple_format_fallback_and_missing_component(self) -> None:
        raw = '{"transition":{"prediction":"door opens","knowledge":[]},"progress":{"prediction":"uncertain","knowledge":[]}}'
        for value in [raw[:-1], "<tool_call>\\n</think>\\n" + raw]:
            self.assertEqual(parse_action_prediction(value, "open door")["progress"]["prediction"], "uncertain")
        with self.assertRaises(ValueError):
            parse_action_prediction('{"transition":{"prediction":"door opens"}}', "open door")

    def test_missing_prediction_content_is_not_invented(self) -> None:
        for raw in ('{"transition":{"prediction":"door opens"}}',
                    '{"transition":{"prediction":""},"progress":{"prediction":"route available"}}',
                    '{"name":"predict_transition","arguments":{"prediction":"door opens"}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_action_prediction(raw, "open door")

    def test_observation_does_not_label_progress_from_reward(self) -> None:
        result = evaluate_observed_transition(reward=0.0, previous_reward=0.0,
            done=False, success=False, invalid_action=False)
        self.assertEqual(result["actual"]["reward_delta"], 0.0)
        self.assertEqual(set(result), {"actual"})


class KnowledgeStoreTest(unittest.TestCase):
    def test_reports_missing_file_and_definition_together_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            plan = {"object_knowledge_updates": [{"target": "Animal", "operation": "create"}],
                    "procedural_knowledge_updates": [{"target": "finding", "operation": "create"}]}
            patch = "*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n+HELPER = 1\n*** End Patch"
            with self.assertRaises(ValueError) as caught:
                store.updates_from_patch(patch, plan)
            self.assertIn("Missing planned files: procedural_knowledge/finding.py", str(caught.exception))
            self.assertIn("missing top-level definition 'Animal'", str(caught.exception))
            self.assertEqual(store.object_source(), "")
            self.assertFalse((store.procedural_dir / "finding.py").exists())

    def test_add_header_normalization_preserves_literal_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            plan = {"object_knowledge_updates": [],
                    "procedural_knowledge_updates": [{"target": "finding", "operation": "create"}]}
            patch = '*** Begin Patch\n*** Add File: procedural_knowledge/finding.py\n@@\n+TEXT = """\n+@@\n+"""\n*** End Patch'
            objects, procedures = store.updates_from_patch(patch, plan)
            self.assertIn('\n@@\n', procedures[0]['code'])
            self.assertFalse((store.procedural_dir / 'finding.py').exists())
            store.commit_knowledge_updates(object_updates=objects, procedural_updates=procedures)
            self.assertTrue((store.procedural_dir / 'finding.py').exists())

    def test_add_error_identifies_file_line_and_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            plan = {"object_knowledge_updates": [],
                    "procedural_knowledge_updates": [{"target": "finding", "operation": "create"}]}
            with self.assertRaises(ValueError) as caught:
                store.updates_from_patch('*** Begin Patch\n*** Add File: procedural_knowledge/finding.py\nTEXT = 1\n*** End Patch', plan)
            self.assertIn('procedural_knowledge/finding.py: patch line 3', str(caught.exception))
            self.assertIn("prefix every content line", str(caught.exception))

    def test_reports_syntax_errors_in_multiple_files_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = KnowledgeStore.create(Path(tmp))
            plan = {"object_knowledge_updates": [{"target": "Animal", "operation": "create"}],
                    "procedural_knowledge_updates": [{"target": "finding", "operation": "create"}]}
            patch = '*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n+class Animal(\n*** Add File: procedural_knowledge/finding.py\n+def finding(\n*** End Patch'
            with self.assertRaises(ValueError) as caught:
                store.updates_from_patch(patch, plan)
            self.assertIn('object_knowledge.py:1:', str(caught.exception))
            self.assertIn('procedural_knowledge/finding.py:1:', str(caught.exception))
            self.assertEqual(store.object_source(), '')
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
            with self.assertRaisesRegex(ValueError, "object_knowledge.py:"):
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
