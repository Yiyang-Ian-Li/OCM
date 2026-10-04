import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agents.ocm_agent.agent import OCMAgent


def make_agent(root: str) -> OCMAgent:
    return OCMAgent(
        {
            "model_name": "test-model",
            "api_key": "test-key",
            "base_url": "http://127.0.0.1:1/v1",
            "temperature": 0,
            "output_dir": root,
        }
    )


class OCMAgentTest(unittest.TestCase):
    def test_only_predictor_requests_schema_constrained_output(self) -> None:
        from agents.ocm_agent.prediction import PREDICTION_RESPONSE_FORMAT
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.llm.create_chat_completion = Mock(return_value=SimpleNamespace(content="response"))
            for call_type in ("ocm_policy", "ocm_predict", "ocm_reflect", "ocm_knowledge_update"):
                agent._chat([{"role": "user", "content": "test"}], call_type=call_type)
                kwargs = agent.llm.create_chat_completion.call_args.kwargs
                expected = PREDICTION_RESPONSE_FORMAT if call_type == "ocm_predict" else None
                self.assertEqual(kwargs["response_format"], expected)
            schema = PREDICTION_RESPONSE_FORMAT["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["transition", "progress"])
            for component in schema["properties"].values():
                self.assertEqual(component["required"], ["prediction", "knowledge"])
                self.assertFalse(component["additionalProperties"])

    def test_exhausted_format_retries_save_all_failed_outputs(self) -> None:
        from agents.ocm_agent.prediction import parse_action_prediction
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            messages = [{"role": "user", "content": "Predict open door."}]
            failed = ['{}', '{"transition":{}}', '{"progress":{}}']
            agent._chat = Mock(side_effect=failed)
            with self.assertRaisesRegex(RuntimeError, "after three attempts"):
                agent._call_with_format_retry(messages, call_type="ocm_predict",
                                              parser=lambda raw: parse_action_prediction(raw, "open door"))
            files = list((Path(tmp) / "ocm_agent/tasks/prompts").glob("*_ocm_predict.json"))
            self.assertEqual(len(files), 1)
            log = json.loads(files[0].read_text())
            self.assertEqual(log["messages"], messages)
            self.assertEqual(log["raw_output"], failed[-1])
            self.assertEqual([a["raw_output"] for a in log["parsed"]["attempts"]], failed)
            self.assertTrue(all(a["error"] for a in log["parsed"]["attempts"]))
            self.assertIsNone(log["parsed"]["value"])
            self.assertEqual(agent._chat.call_count, 3)

    def test_reflection_links_executed_predictions_to_original_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            source = "LIMIT = 2\nclass Door:\n    locked = True\n"
            agent.knowledge.object_knowledge_path.write_text(source)
            agent.knowledge.commit_knowledge_updates(object_updates=[], procedural_updates=[
                {"name": "opening", "code": "def opening():\n    return 'open door'\n"},
                {"name": "unrelated", "code": "def unrelated():\n    return 'wait'\n"},
            ])
            agent.on_episode_start("test", "Open the door")
            steps = [{"step": i, "prediction": {"transition": {"prediction": "door opens", "knowledge": ["object.Door"]},
                "progress": {"prediction": "route accessible", "knowledge": ["procedural.opening"]}},
                "draft_prediction": {"knowledge": ["procedural.unrelated"]},
                "action": "open door", "next_observation": "It is locked.",
                "actual": {"success": False}} for i in [1, 2]]
            agent._task_log["steps"] = steps
            agent._chat = Mock(return_value=json.dumps({
                "object_knowledge_updates": [], "procedural_knowledge_updates": []}))
            agent._update_knowledge({"success": False})
            payload = json.loads(agent._chat.call_args.args[0][1]["content"])
            self.assertEqual(payload["knowledge_prediction_steps"], {
                "object.Door": [{"step": i, "component": "transition"} for i in [1, 2]],
                "procedural.opening": [{"step": i, "component": "progress"} for i in [1, 2]]})
            self.assertEqual(set(payload["referenced_source"]), {
                "object_knowledge.py", "procedural_knowledge/opening.py"})
            self.assertEqual(payload["referenced_source"]["object_knowledge.py"], source)
            self.assertEqual(payload["executed_prediction_traces"][0]["actual"], {"success": False})
            self.assertIn("unrelated", payload["procedural_knowledge_catalog"])
            self.assertEqual(agent._chat.call_count, 1)

    def test_patch_retry_sees_rejected_patch_and_preserves_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.knowledge.commit_knowledge_updates(
                object_updates=[{"name": "Door", "code": "class Door:\n    locked = True\n"}],
                procedural_updates=[],
            )
            agent.on_episode_start("test", "Open the door")
            bad_patch = "*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n-class Missing:\n+class Door:\n*** End Patch"
            good_patch = "*** Begin Patch\n*** Update File: object_knowledge.py\n@@\n-    locked = True\n+    locked = False\n*** End Patch"
            agent._chat = Mock(side_effect=[
                json.dumps({"object_knowledge_updates": [{"target": "Door", "operation": "update",
                    "lesson": "The door is unlocked", "evidence": "The door opened"}],
                    "procedural_knowledge_updates": []}),
                bad_patch, good_patch,
            ])
            meta = agent._update_knowledge({"success": True})
            self.assertIsNone(meta["knowledge_update_error"])
            self.assertIn("locked = False", agent.knowledge.object_source())
            retry = agent._chat.call_args_list[2].args[0]
            self.assertEqual(retry[-2], {"role": "assistant", "content": bad_patch})
            self.assertIn("Nothing was applied", retry[-1]["content"])
            reflection = json.loads(agent._chat.call_args_list[0].args[0][1]["content"])
            self.assertNotIn("existing_knowledge", reflection)
            self.assertEqual(reflection["knowledge_prediction_steps"], {})
            self.assertEqual(reflection["referenced_source"], {})

    def test_action_is_predicted_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "Find the longest-lived animal.")
            agent._chat = Mock(
                side_effect=[
                    "Thought: The crocodile is longer lived.\n"
                    "Prediction: I expect it to finish.\n"
                    "Action: focus on crocodile",
                    '{"transition":{"prediction":"task completes","knowledge":[]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: Keep the proposed action.\n"
                    "Prediction: This extra output must not enter history.\n"
                    "Action: focus on crocodile",
                ]
            )
            action = agent.act(
                "A rabbit and crocodile are outside.",
                [{"role": "user", "content": "A rabbit and crocodile are outside."}],
            )
            self.assertEqual(action, "focus on crocodile")
            decision = agent._decision_log[-1]
            self.assertEqual(decision["prediction"]["transition"]["prediction"], "task completes")
            self.assertFalse(decision["action_changed_by_foresight"])
            self.assertEqual(
                agent.last_raw_output,
                "Thought: Keep the proposed action.\nAction: focus on crocodile",
            )
            self.assertIn("Prediction:", decision["policy_raw_output"])

    def test_executed_prediction_receives_the_actual_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "goal")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Open the door to expose the route.\nAction: open door",
                    '{"transition":{"prediction":"door opens","knowledge":[]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: The predicted result is useful.\nAction: open door",
                ]
            )
            action = agent.act("closed door", [{"role": "user", "content": "closed door"}])
            agent.on_step(
                1,
                "closed door",
                action,
                SimpleNamespace(
                    reward=0.2,
                    done=False,
                    success=False,
                    invalid_action=False,
                ),
                "door is open",
            )
            step = agent._task_log["steps"][0]
            self.assertEqual(step["action"], "open door")
            self.assertEqual(step["actual"]["reward_delta"], 0.2)
            self.assertEqual(step["prediction"]["transition"]["prediction"], "door opens")
            self.assertNotIn("validity_correct", step)

    def test_history_excludes_predictions_but_review_and_reflection_keep_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "goal")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Try the rabbit.\nAction: focus rabbit",
                    '{"transition":{"prediction":"task completes","knowledge":[]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: Keep the test.\nAction: focus rabbit",
                    "Thought: The rabbit action did not complete the task.\nAction: focus crocodile",
                    '{"transition":{"prediction":"task completes","knowledge":[]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: Try the alternative.\nAction: focus crocodile",
                ]
            )
            first = agent.act("animals", [{"role": "user", "content": "animals"}])
            self.assertEqual(first, "focus rabbit")
            first_output = agent.last_raw_output
            agent.on_step(
                1,
                "animals",
                first,
                SimpleNamespace(
                    reward=0.0,
                    done=False,
                    success=False,
                    invalid_action=False,
                ),
                "You focus on the rabbit.",
            )
            trajectory = [
                {"role": "user", "content": "animals"},
                {"role": "assistant", "content": first_output},
                {"role": "user", "content": "You focus on the rabbit."},
            ]
            second = agent.act(
                "You focus on the rabbit.",
                trajectory,
            )
            self.assertEqual(second, "focus crocodile")
            self.assertEqual(first_output, "Thought: Keep the test.\nAction: focus rabbit")
            for index in (3, 4, 5):
                messages = agent._chat.call_args_list[index].args[0]
                self.assertEqual(messages[1:4], trajectory)
                for message in messages[1:4]:
                    self.assertNotIn("Prediction:", message["content"])
                    self.assertNotIn('"prediction":', message["content"])
            review = agent._chat.call_args_list[5].args[0][-1]["content"]
            current_prediction = json.loads(review.split("\n")[1])
            self.assertEqual(current_prediction["action"], "focus crocodile")
            self.assertEqual(current_prediction["transition"]["prediction"], "task completes")
            agent._chat = Mock(return_value=json.dumps({
                "object_knowledge_updates": [], "procedural_knowledge_updates": []}))
            agent._update_knowledge({"success": False})
            payload = json.loads(agent._chat.call_args.args[0][1]["content"])
            trace = payload["executed_prediction_traces"][0]
            self.assertEqual(trace["prediction"]["transition"]["prediction"], "task completes")
            self.assertEqual(trace["action"], "focus rabbit")
            self.assertEqual(trace["next_observation"], "You focus on the rabbit.")
            self.assertFalse(trace["actual"]["success"])

    def test_episode_residual_can_create_reusable_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "goal")
            agent._task_log["steps"] = [
                {
                    "step": 1,
                    "observation": "rabbit and crocodile",
                    "prediction": {
                        "action": "focus rabbit",
                        "transition": {"prediction": "rabbit is focused", "knowledge": []},
                        "progress": {"prediction": "task completes", "knowledge": []},
                    },
                    "action": "focus rabbit",
                    "next_observation": "not complete",
                    "actual": {"reward": 0.5, "success": False},
                }
            ]
            agent._chat = Mock(
                side_effect=[
                    '{"object_knowledge_updates":[{"target":"Animal","operation":"create",'
                    '"lesson":"crocodiles outlive rabbits","evidence":"focusing rabbit did not complete; '
                    'focusing crocodile later completed"}],"procedural_knowledge_updates":[]}',
                    "*** Begin Patch\n"
                    "*** Update File: object_knowledge.py\n"
                    "@@\n"
                    "+class Animal:\n"
                    '+    """Known lifespan orderings."""\n'
                    '+    longer_lived_pairs = {("crocodile", "rabbit")}\n'
                    "*** End Patch",
                ]
            )
            meta = agent._update_knowledge({"success": True, "reward": 1.0})
            self.assertIsNone(meta["knowledge_update_error"])
            self.assertEqual(meta["written_knowledge"]["object"], ["Animal"])
            self.assertIn("crocodile", agent.knowledge.model_context())

    def test_existing_memory_is_available_to_prediction_and_logged_when_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.knowledge.commit_knowledge_updates(
                object_updates=[
                    {
                        "existing_name": "",
                        "name": "Animal",
                        "code": "class Animal:\n    longer = [('crocodile', 'rabbit')]\n",
                    }
                ],
                procedural_updates=[],
            )
            agent.on_episode_start("sciworld", "goal")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Use the known longer-lived animal.\nAction: focus crocodile",
                    '{"transition":{"prediction":"task completes","knowledge":["object.Animal"]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: Keep the knowledge-backed action.\nAction: focus crocodile",
                ]
            )
            agent.act("animals", [{"role": "user", "content": "animals"}])
            decision = agent._decision_log[-1]
            self.assertTrue(decision["knowledge_available"])
            self.assertEqual(decision["knowledge_references"], ["object.Animal"])
            prediction_prompt = agent._chat.call_args_list[1].args[0]
            self.assertIn("('crocodile', 'rabbit')", prediction_prompt[0]["content"])

    def test_fallible_foresight_can_revise_an_action_without_memory_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.knowledge.commit_knowledge_updates(
                object_updates=[],
                procedural_updates=[
                    {
                        "existing_name": "",
                        "name": "navigation",
                        "code": "def directly_connected(destination, doors):\n    return destination in doors\n",
                    }
                ],
            )
            agent.on_episode_start("sciworld", "hallway")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Go directly.\nAction: go to bathroom",
                    '{"transition":{"prediction":"No door leads there.","knowledge":[]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                    "Thought: Use a visible exit.\nAction: go to kitchen",
                    '{"transition":{"prediction":"Move to the kitchen.","knowledge":["procedural.navigation"]},"progress":{"prediction":"May advance the goal.","knowledge":[]}}',
                ]
            )

            action = agent.act(
                "The hallway has a door to the kitchen.",
                [{"role": "user", "content": "The hallway has a door to the kitchen."}],
            )

            self.assertEqual(action, "go to kitchen")
            decision = agent._decision_log[-1]
            self.assertTrue(decision["action_changed_by_foresight"])
            self.assertEqual(decision["draft_action"], "go to bathroom")
            self.assertEqual(decision["prediction"]["action"], "go to kitchen")
            revision_prompt = agent._chat.call_args_list[2].args[0]
            self.assertIn("fallible", revision_prompt[0]["content"])
            self.assertIn("may be the same action or a revision", revision_prompt[-1]["content"])
            self.assertIn("go to bathroom", revision_prompt[-1]["content"])
            self.assertEqual(agent._chat.call_count, 4)
            agent.on_step(1, "hallway", action,
                SimpleNamespace(reward=0, done=False, success=False, invalid_action=False),
                "You are in the kitchen.")
            agent._chat = Mock(return_value=json.dumps({
                "object_knowledge_updates": [], "procedural_knowledge_updates": []}))
            agent._update_knowledge({"success": False})
            payload = json.loads(agent._chat.call_args.args[0][1]["content"])
            self.assertEqual(payload["executed_prediction_traces"][0]["prediction"]["action"], "go to kitchen")
            self.assertNotIn("draft_prediction", payload["executed_prediction_traces"][0])
            self.assertEqual(payload["knowledge_prediction_steps"], {
                "procedural.navigation": [{"step": 1, "component": "transition"}]})

    def test_redundant_update_does_not_discard_an_independent_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            container_code = "class Container:\n    pass\n"
            agent.knowledge.commit_knowledge_updates(
                object_updates=[
                    {
                        "existing_name": "",
                        "name": "Container",
                        "code": container_code,
                    }
                ],
                procedural_updates=[],
            )
            agent.on_episode_start("sciworld", "goal")
            agent._task_log["steps"] = []
            agent._chat = Mock(
                side_effect=[
                    '{"object_knowledge_updates":[{"target":"Container","operation":"update",'
                    '"lesson":"preserve the known rule","evidence":"observed transition"}],'
                    '"procedural_knowledge_updates":[{"target":"measurement","operation":"create",'
                    '"lesson":"open a container before measuring its contents",'
                    '"evidence":"measurement failed before opening and succeeded after"}]}',
                    "*** Begin Patch\n"
                    "*** Update File: object_knowledge.py\n"
                    "@@\n"
                    " class Container:\n"
                    "-    pass\n"
                    "+    pass\n"
                    "*** Add File: procedural_knowledge/measurement.py\n"
                    "+def requires_open_container():\n"
                    "+    return True\n"
                    "*** End Patch",
                ]
            )

            meta = agent._update_knowledge({"success": False, "reward": 0.5})

            self.assertIsNone(meta["knowledge_update_error"])
            self.assertEqual(meta["written_knowledge"]["object"], [])
            self.assertEqual(meta["written_knowledge"]["procedural"], ["measurement"])


if __name__ == "__main__":
    unittest.main()
