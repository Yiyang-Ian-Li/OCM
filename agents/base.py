from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


logger = logging.getLogger("benchmark_runner")


class Agent(ABC):
    """Minimal interface for benchmark agents."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.stop_sequences = list(self.config.get("stop_sequences", []))
        self._episode_token_usage = self._new_token_usage_bucket()
        self._lifetime_token_usage = self._new_token_usage_bucket()
        self._last_token_usage: dict[str, int] | None = None
        logger.debug("Initialized %s with config keys: %s", self.__class__.__name__, sorted(self.config))

    @staticmethod
    def _new_token_usage_bucket() -> dict[str, Any]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "by_call_type": {},
        }

    @staticmethod
    def _clone_token_usage_bucket(bucket: Mapping[str, Any]) -> dict[str, Any]:
        cloned = {
            "prompt_tokens": int(bucket.get("prompt_tokens", 0)),
            "completion_tokens": int(bucket.get("completion_tokens", 0)),
            "total_tokens": int(bucket.get("total_tokens", 0)),
            "call_count": int(bucket.get("call_count", 0)),
            "by_call_type": {},
        }
        by_call_type = bucket.get("by_call_type", {})
        for key, value in by_call_type.items():
            cloned["by_call_type"][str(key)] = {
                "prompt_tokens": int(value.get("prompt_tokens", 0)),
                "completion_tokens": int(value.get("completion_tokens", 0)),
                "total_tokens": int(value.get("total_tokens", 0)),
                "call_count": int(value.get("call_count", 0)),
            }
        return cloned

    @staticmethod
    def _add_usage(bucket: dict[str, Any], usage: Mapping[str, int], call_type: str) -> None:
        bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        bucket["completion_tokens"] += int(usage.get("completion_tokens", 0))
        bucket["total_tokens"] += int(usage.get("total_tokens", 0))
        bucket["call_count"] += 1
        by_type = bucket["by_call_type"].setdefault(
            call_type,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "call_count": 0,
            },
        )
        by_type["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        by_type["completion_tokens"] += int(usage.get("completion_tokens", 0))
        by_type["total_tokens"] += int(usage.get("total_tokens", 0))
        by_type["call_count"] += 1

    def reset_episode_token_usage(self) -> None:
        self._episode_token_usage = self._new_token_usage_bucket()
        self._last_token_usage = None

    def _record_token_usage(self, usage: Mapping[str, int] | None, call_type: str) -> None:
        if not usage:
            self._last_token_usage = None
            return
        normalized = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        self._add_usage(self._episode_token_usage, normalized, call_type)
        self._add_usage(self._lifetime_token_usage, normalized, call_type)
        self._last_token_usage = dict(normalized)

    def last_token_usage(self) -> dict[str, int] | None:
        return dict(self._last_token_usage) if self._last_token_usage is not None else None

    def episode_token_usage(self) -> dict[str, Any] | None:
        if self._episode_token_usage["call_count"] <= 0:
            return None
        return self._clone_token_usage_bucket(self._episode_token_usage)

    def lifetime_token_usage(self) -> dict[str, Any] | None:
        if self._lifetime_token_usage["call_count"] <= 0:
            return None
        return self._clone_token_usage_bucket(self._lifetime_token_usage)

    @abstractmethod
    def act(
        self,
        observation: str,
        trajectory: Sequence[dict[str, str]] | None = None,
    ) -> str:
        raise NotImplementedError

    def on_episode_start(
        self,
        benchmark_name: str | None = None,
        observation: str | None = None,
    ) -> None:
        pass

    def on_step(
        self,
        step_idx: int,
        observation: str,
        action: str,
        outcome: Any,
        next_observation: str,
        benchmark_name: str | None = None,
    ) -> None:
        pass

    def on_episode_end(
        self,
        result_meta: Mapping[str, Any] | None = None,
        benchmark_name: str | None = None,
    ) -> dict[str, Any] | None:
        return None
