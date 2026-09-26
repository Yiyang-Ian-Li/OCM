from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.worldevolver_agent import WorldEvolverAgent


class _FakeLLM:
    def __init__(self) -> None:
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "n_calls": 0,
        }


class _FakeWorldModel:
    def __init__(self) -> None:
        self.predicted_actions: list[str] = []
        self.updates: list[dict] = []

    def reset(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {"n_records": len(self.updates), "n_rules": 0, "pending": 0}

    def predict(self, *, state: str, action: str, goal: str) -> dict:
        self.predicted_actions.append(action)
        return {
            "prediction": f"Prediction: outcome of {action}",
            "foresight": f"Prediction: outcome of {action}",
            "mean_logprob": -0.1,
            "system": "large repeated prompt",
        }

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class WorldEvolverAgentTest(unittest.TestCase):
    def _agent(self) -> tuple[WorldEvolverAgent, _FakeWorldModel]:
        llm = _FakeLLM()
        world_model = _FakeWorldModel()
        with patch.object(
            WorldEvolverAgent,
            "_create_world_model",
            return_value=(llm, world_model),
        ):
            agent = WorldEvolverAgent(
                {
                    "model_name": "test-model",
                    "api_key": "local",
                    "base_url": "http://127.0.0.1:1/v1",
                    "temperature": 0,
                    "timeout": 1,
                }
            )
        agent.on_episode_start("sciworld", "initial observation")
        agent.set_world_model_state("state plus inventory")
        agent.set_world_model_goal("goal")
        return agent, world_model

    def test_changed_action_is_repredicted_and_aligned_for_update(self) -> None:
        agent, world_model = self._agent()
        decisions = iter(
            [
                ("open door", "try the door", "Thought: try the door\nAction: open door"),
                ("take key", "get the key first", "Thought: get the key first\nAction: take key"),
            ]
        )
        agent._policy_decision = lambda *args, **kwargs: next(decisions)

        action = agent.act("observation", [{"role": "user", "content": "observation"}])
        self.assertEqual(action, "take key")
        self.assertEqual(world_model.predicted_actions, ["open door", "take key"])
        self.assertNotIn("large repeated prompt", str(agent._decision_log[-1]))

        agent.on_step(1, "observation", action, object(), "next observation", "sciworld")
        self.assertEqual(world_model.updates[0]["action"], "take key")
        self.assertEqual(world_model.updates[0]["prediction"], "Prediction: outcome of take key")

    def test_unchanged_action_exposes_foresight_as_observation_addendum(self) -> None:
        agent, world_model = self._agent()
        decisions = iter(
            [
                ("look around", "inspect", "Thought: inspect\nAction: look around"),
                ("look around", "still inspect", "Thought: still inspect\nAction: look around"),
            ]
        )
        agent._policy_decision = lambda *args, **kwargs: next(decisions)

        action = agent.act("observation", None)
        self.assertEqual(action, "look around")
        self.assertEqual(world_model.predicted_actions, ["look around"])
        self.assertTrue(agent.trajectory_observation_addendum().startswith("Foresight:"))
        self.assertTrue(agent.last_raw_output.startswith("Thought:"))


if __name__ == "__main__":
    unittest.main()
