"""Adapt the previously evaluated WorldEvolver ReAct policy to the shared runner."""

import importlib.util
from pathlib import Path
import sys

from .react_agent import ReactAgent


def load_reference_classes():
    import agents

    source = Path(__file__).resolve().parents[1] / "baselines/WorldEvolver/src"
    if str(source) not in sys.path:
        sys.path.append(str(source))
    if str(source / "agents") not in agents.__path__:
        agents.__path__.append(str(source / "agents"))
    from llm.openai_gpt import OPENAI_GPT

    name = "agents._worldevolver_reference_react"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, source / "agents/react_agent.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return OPENAI_GPT, sys.modules[name].ReactAgent, source


class ReferenceReactAgent(ReactAgent):
    def __init__(self, config):
        super().__init__(config)
        llm_class, policy_class, source = load_reference_classes()
        self.reference_llm = llm_class(
            engine=self.model_name, api_key=self.api_key, base_url=config["base_url"],
            temperature=self.temperature, top_p=self.top_p, seed=self.seed,
            max_tokens=100, stop=["\n"], context_length=32768,
            timeout=self.timeout, retry_delays=1, max_retry_iters=3,
        )
        self.policy = policy_class(
            self.reference_llm, memory_size=100, need_goal=True, use_parser=True,
            max_think_iters=3, check_actions=None, check_inventory=True,
            init_prompt_path=str(source / "prompts/ReactAgent/scienceworld_react.json"),
        )
        accumulate = self.reference_llm._accumulate_usage

        def record_usage(usage):
            accumulate(usage)
            self._record_token_usage(usage, "react_policy")

        self.reference_llm._accumulate_usage = record_usage
        generate = self.reference_llm.generate

        def record_generation(system_message, prompt):
            success, response = generate(system_message, prompt)
            self._calls.append({"system": system_message, "prompt": prompt, "response": response})
            return success, response

        self.reference_llm.generate = record_generation
        self._calls = []
        self._needs_reset = True
        if config.get("resume") and self.output_dir is not None:
            self._episode_index = max(
                (int(path.stem.removeprefix("episode_"))
                 for path in (self.output_dir / "react_agent/tasks").glob("episode_*.json")),
                default=0,
            )

    def _build_prompt_preamble(self):
        examples = self.policy.examples
        if isinstance(examples, list):
            examples = "\n".join(examples)
        return [{"role": "system", "content": self.policy.init_prompt_dict["system_msg"]},
                {"role": "user", "content": self.policy.instruction + "\n" + examples}]

    def on_episode_start(self, **kwargs):
        super().on_episode_start(**kwargs)
        self._needs_reset = True

    def set_world_model_goal(self, goal):
        self._goal = goal

    def set_world_model_state(self, state):
        if self._needs_reset:
            self.policy.reset(goal=self._goal, init_obs=state)
            self._needs_reset = False

    def act(self, observation, trajectory=None):
        self._calls = []
        success, action = self.policy.run()
        if not success:
            raise RuntimeError("Reference ReAct exhausted its LLM retries.")
        self.last_raw_output = "\n".join(call["response"] or "" for call in self._calls)
        self._decision_log.append({"step": len(self._decision_log), "observation": observation,
                                   "raw_output": self.last_raw_output, "parsed_action": action,
                                   "calls": self._calls, "token_usage": self.last_token_usage()})
        return action

    def on_step(self, **kwargs):
        self.policy.update(action=kwargs["action"], state=kwargs["next_observation"])
        super().on_step(**kwargs)
