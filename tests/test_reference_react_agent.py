import unittest

from agents.reference_react_agent import ReferenceReactAgent
from benchmarks.base import StepResult


class ReferenceReactTest(unittest.TestCase):
    def make_agent(self):
        return ReferenceReactAgent({"model_name": "test", "api_key": "local",
                                    "base_url": "http://127.0.0.1:8080/v1",
                                    "temperature": 0, "top_p": 0.5, "seed": 42, "timeout": 300})

    def test_reference_settings_and_episode_memory_reset(self):
        agent = self.make_agent()
        self.assertEqual(agent.reference_llm.max_tokens, 100)
        self.assertEqual(agent.reference_llm.stop, ["\n"])
        self.assertEqual(agent.reference_llm.context_length, 32768)
        self.assertIsNone(agent.policy.check_actions)
        agent.reference_llm.llm_inference = lambda messages: "Thought: Inspect the room. Action: look around"
        agent.on_episode_start(benchmark_name="sciworld", observation="goal\nroom")
        agent.set_world_model_goal("goal")
        agent.set_world_model_state("room\nInventory: empty")
        expected = agent.policy.make_prompt(need_goal=True, check_actions=None, check_inventory=True,
                                            system_message=agent.policy.init_prompt_dict["system_msg"])
        self.assertEqual(agent.act("room"), "look around")
        self.assertEqual(agent._calls[0]["prompt"], expected)
        self.assertIn(("Thought", "Inspect the room"), agent.policy.memory)
        agent.on_step(step_idx=1, observation="room", action="look around",
                      outcome=StepResult(observation="new observation", reward=0),
                      next_observation="new observation", benchmark_name="sciworld")
        self.assertEqual(agent.policy.memory[-2:], [("Action", "look around"), ("Observation", "new observation")])
        agent.on_episode_start(benchmark_name="sciworld", observation="second episode")
        agent.set_world_model_goal("second goal")
        agent.set_world_model_state("second room")
        self.assertEqual(agent.policy.memory, [("Observation", "second room")])

    def test_reference_usage_is_reported_to_shared_runner(self):
        agent = self.make_agent()
        agent.reference_llm._accumulate_usage({"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})
        self.assertEqual(agent.episode_token_usage()["total_tokens"], 25)
        self.assertEqual(agent.episode_token_usage()["call_count"], 1)
