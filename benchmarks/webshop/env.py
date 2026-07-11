from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment

from benchmarks.base import BenchmarkEnv, StepResult
from benchmarks.webshop.task import WebShopTask, resolve_webshop_repo


_WEBSHOP_SERVER = None


class WebShopEnv(BenchmarkEnv):
    def __init__(
        self,
        task: WebShopTask,
        max_steps: int = 15,
        num_products: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_steps=max_steps, **kwargs)
        self.task = task
        self.num_products = num_products
        self.repo_path = resolve_webshop_repo()
        self.env = None
        self.reward = 0.0
        self.steps = 0

    def reset(self) -> str:
        self._prepare_imports()
        from web_agent_site.envs import WebAgentTextEnv

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = WebAgentTextEnv(
                observation_mode="text",
                server=_get_shared_server(self.repo_path, self.num_products),
                num_products=self.num_products,
                human_goals=True,
                quiet=True,
            )
            observation, _info = self.env.reset(session=self.task.session_id)
        self.reward = 0.0
        self.steps = 0
        return _format_observation(observation, self.env.get_available_actions())

    def step(self, action: str) -> StepResult:
        if self.env is None:
            return StepResult("WebShop environment is not initialized.", reward=0.0, done=True, success=False)
        action = action.strip()
        if not action:
            return StepResult("Empty action. Use search[...] or click[...].", reward=0.0, invalid_action=True)
        if action.lower() in {"finish", "done", "submit"}:
            return StepResult(
                observation=f"Finished without purchase. Current reward={self.reward:.3f}.",
                reward=self.reward,
                done=True,
                success=self.reward >= 1.0,
                terminate_reason="agent_finish",
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                observation, reward, done, _info = self.env.step(action)
        except Exception as exc:
            return StepResult(
                observation=f"Invalid WebShop action: {type(exc).__name__}: {exc}",
                reward=self.reward,
                invalid_action=True,
                info={"steps": self.steps},
            )
        self.steps += 1
        self.reward = float(reward or 0.0)
        success = self.reward >= 1.0 if done else None
        return StepResult(
            observation=_format_observation(observation, self.env.get_available_actions()),
            reward=self.reward,
            done=bool(done),
            success=success,
            terminate_reason="success" if success else ("done" if done else None),
            info={"steps": self.steps},
        )

    def close(self) -> None:
        self.env = None

    def _prepare_imports(self) -> None:
        data_dir = self.repo_path / "data"
        required = ["items_shuffle.json", "items_ins_v2.json", "items_human_ins.json"]
        missing = [name for name in required if not (data_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing WebShop data files under {data_dir}: {', '.join(missing)}")
        sys.path.insert(0, str(self.repo_path))
        import web_agent_site.utils as utils
        import web_agent_site.engine.engine as engine

        utils.DEFAULT_FILE_PATH = str(data_dir / "items_shuffle.json")
        utils.DEFAULT_ATTR_PATH = str(data_dir / "items_ins_v2.json")
        utils.HUMAN_ATTR_PATH = str(data_dir / "items_human_ins.json")
        engine.DEFAULT_FILE_PATH = utils.DEFAULT_FILE_PATH
        engine.DEFAULT_ATTR_PATH = utils.DEFAULT_ATTR_PATH
        engine.HUMAN_ATTR_PATH = utils.HUMAN_ATTR_PATH


def _get_shared_server(repo_path: Path, num_products: int | None):
    global _WEBSHOP_SERVER
    if _WEBSHOP_SERVER is None:
        from web_agent_site.envs.web_agent_text_env import SimServer

        _WEBSHOP_SERVER = SimServer(
            base_url="http://127.0.0.1:3000",
            file_path=str(repo_path / "data" / "items_shuffle.json"),
            filter_goals=None,
            limit_goals=-1,
            num_products=num_products,
            human_goals=True,
            show_attrs=False,
        )
    return _WEBSHOP_SERVER


def _format_observation(observation: str, actions: dict[str, Any] | None = None) -> str:
    text = _text_from_html_or_text(observation)
    lines = ["WebShop observation:", text]
    if actions:
        if actions.get("has_search_bar"):
            lines.append("\nAvailable search action: search[query]")
        clickables = actions.get("clickables") or []
        if clickables:
            shown = ", ".join(f"click[{item}]" for item in clickables[:80])
            if len(clickables) > 80:
                shown += f", ... ({len(clickables) - 80} more)"
            lines.append(f"Available click actions: {shown}")
    return "\n".join(lines)


def _text_from_html_or_text(observation: str) -> str:
    if "<html" not in observation.lower() and "[SEP]" in observation:
        return observation.replace(" [SEP] ", "\n").strip()
    html_obj = BeautifulSoup(observation, "html.parser")
    texts = html_obj.find_all(string=True)
    visible = [item.strip() for item in texts if item.strip() and _tag_visible(item)]
    return "\n".join(visible).strip() or observation


def _tag_visible(element) -> bool:
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)
