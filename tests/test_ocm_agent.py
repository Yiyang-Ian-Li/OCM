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
    def test_action_is_predicted_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "Find the longest-lived animal.")
            agent._chat = Mock(
                side_effect=[
                    "Thought: The crocodile is longer lived.\n"
                    "Prediction: I expect it to finish.\n"
                    "Action: focus on crocodile",
                    '{"outcome":"task completes","valid":true,"effect":"complete","knowledge":[]}',
                    "Thought: Keep the proposed action.\nAction: focus on crocodile",
                ]
            )
            action = agent.act(
                "A rabbit and crocodile are outside.",
                [{"role": "user", "content": "A rabbit and crocodile are outside."}],
            )
            self.assertEqual(action, "focus on crocodile")
            decision = agent._decision_log[-1]
            self.assertEqual(decision["prediction"]["outcome"], "task completes")
            self.assertFalse(decision["action_changed_by_foresight"])
            self.assertIn("Prediction: task completes", agent.last_raw_output)
            self.assertEqual(agent.last_raw_output.count("Prediction:"), 1)
            self.assertEqual(agent.last_raw_output.splitlines()[-1], "Action: focus on crocodile")

    def test_executed_prediction_receives_the_actual_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "goal")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Open the door to expose the route.\nAction: open door",
                    '{"outcome":"door opens","valid":true,"effect":"advance","knowledge":[]}',
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
            self.assertEqual(step["prediction"]["outcome"], "door opens")
            self.assertTrue(step["validity_correct"])

    def test_prediction_residual_is_visible_on_the_next_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(tmp)
            agent.on_episode_start("sciworld", "goal")
            agent._chat = Mock(
                side_effect=[
                    "Thought: Try the rabbit.\nAction: focus rabbit",
                    '{"outcome":"task completes","valid":true,"effect":"complete","knowledge":[]}',
                    "Thought: Keep the test.\nAction: focus rabbit",
                    "Thought: The rabbit prediction was contradicted.\nAction: focus crocodile",
                    '{"outcome":"task completes","valid":true,"effect":"complete","knowledge":[]}',
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
            second_policy_prompt = agent._chat.call_args_list[3].args[0]
            self.assertIn("Prediction: task completes", second_policy_prompt[-2]["content"])
            self.assertIn("You focus on the rabbit.", second_policy_prompt[-1]["content"])

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
                        "outcome": "task completes",
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
                    '{"outcome":"task completes","valid":true,"effect":"complete",'
                    '"knowledge":["object.Animal"]}',
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
                    '{"outcome":"No door leads there.","valid":false,"effect":"harm",'
                    '"knowledge":[]}',
                    "Thought: Use a visible exit.\nAction: go to kitchen",
                    '{"outcome":"Move to the kitchen.","valid":true,"effect":"advance",'
                    '"knowledge":["procedural.navigation"]}',
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
