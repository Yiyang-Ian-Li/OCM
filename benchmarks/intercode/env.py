from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
import os
import re
import time
from pathlib import Path
from typing import Any

import docker

from benchmarks.base import BenchmarkEnv, StepResult
from benchmarks.intercode.task import InterCodeTask, resolve_intercode_repo


class InterCodeEnv(BenchmarkEnv):
    DEFAULT_MAX_OBSERVATION_CHARS = 20000

    def __init__(
        self,
        task: InterCodeTask,
        max_steps: int | None = 10,
        verbose: bool = False,
        max_observation_chars: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_steps=max_steps, **kwargs)
        self.task = task
        self.verbose = verbose
        self.max_observation_chars = (
            self.DEFAULT_MAX_OBSERVATION_CHARS if max_observation_chars is None else int(max_observation_chars)
        )
        self.repo_path = resolve_intercode_repo()
        self.env = None
        self.last_reward = 0.0
        self._active_image_name: str | None = None

    def reset(self) -> str:
        self._prepare_imports()
        self._patch_container_lookup()
        if self.task.family == "bash":
            fs_version = _bash_filesystem_version(self.task)
            base_image = f"intercode-nl2bash-fs{fs_version}"
            image_name = self._runtime_image_name(base_image)
            self._ensure_image(base_image, "docker/nl2bash.Dockerfile", fs_version=fs_version)
            self._ensure_image_alias(base_image, image_name)
            self._register_bash_entrypoint(image_name)
            self._remove_containers(image_name)
            from intercode.envs import BashEnv

            self.env = BashEnv(
                image_name,
                data_path=str(self.task.data_path),
                verbose=self.verbose,
            )
            self._active_image_name = image_name
        else:
            raise ValueError(f"Unsupported InterCode family: {self.task.family}. Only Bash splits are enabled.")
        observation, _info = self.env.reset(self.task.index)
        self.last_reward = 0.0
        return _format_initial_observation(self.task, observation)

    def step(self, action: str) -> StepResult:
        if self.env is None:
            return StepResult("InterCode environment is not initialized.", reward=0.0, done=True, success=False)
        action = action.strip()
        if not action:
            return StepResult("Empty action.", reward=self.last_reward, invalid_action=True)
        observation, reward, done, info = self.env.step(action)
        observation_text, truncation_info = _truncate_observation(str(observation), self.max_observation_chars)
        if truncation_info:
            info = dict(info)
            info["observation_truncated"] = truncation_info
        self.last_reward = float(reward or 0.0)
        success = self.last_reward >= 1.0 if done else None
        return StepResult(
            observation=observation_text,
            reward=self.last_reward,
            done=bool(done),
            success=success,
            invalid_action=not bool(info.get("action_executed", True)),
            terminate_reason="success" if success else ("done" if done else None),
            info=info,
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        if self._active_image_name:
            self._remove_containers(self._active_image_name)
            self._active_image_name = None

    def _prepare_imports(self) -> None:
        sys.path.insert(0, str(self.repo_path))

    def _patch_container_lookup(self) -> None:
        from intercode import utils as intercode_utils

        intercode_utils.get_container = _get_container_by_name
        try:
            from intercode.envs.bash import bash_env
        except Exception:
            return
        bash_env.get_container = _get_container_by_name

    def _ensure_image(self, image_name: str, dockerfile: str, *, fs_version: int | None = None) -> None:
        client = docker.from_env()
        try:
            client.images.get(image_name)
            return
        except docker.errors.ImageNotFound:
            pass
        if image_name.startswith("intercode-nl2bash"):
            self._build_nl2bash_image(client, image_name, fs_version=fs_version or 1)
            return
        client.images.build(
            path=str(self.repo_path),
            dockerfile=dockerfile,
            tag=image_name,
            rm=True,
        )

    def _build_nl2bash_image(self, client, image_name: str, *, fs_version: int) -> None:
        with tempfile.TemporaryDirectory(prefix="ocm-intercode-build-") as tmp:
            tmp_path = Path(tmp)
            shutil.copytree(self.repo_path / "docker", tmp_path / "docker")
            dockerfile = (self.repo_path / "docker" / "nl2bash.Dockerfile").read_text(encoding="utf-8")
            dockerfile = dockerfile.replace("COPY ../docker/", "COPY docker/")
            dockerfile = re.sub(r"ENV\s+file_system_version=\d+", f"ENV file_system_version={int(fs_version)}", dockerfile)
            (tmp_path / "nl2bash.Dockerfile").write_text(dockerfile, encoding="utf-8")
            client.images.build(
                path=str(tmp_path),
                dockerfile="nl2bash.Dockerfile",
                tag=image_name,
                rm=True,
            )

    def _runtime_image_name(self, base_image: str) -> str:
        return f"{base_image}-ocm-{os.getpid()}"

    def _ensure_image_alias(self, source_image: str, alias: str) -> None:
        if source_image == alias:
            return
        client = docker.from_env()
        try:
            client.images.get(alias)
            return
        except docker.errors.ImageNotFound:
            pass
        image = client.images.get(source_image)
        image.tag(alias)

    def _register_bash_entrypoint(self, image_name: str) -> None:
        from intercode.envs.bash import bash_env

        bash_env.IMAGE_TO_SETTINGS.setdefault(image_name, "/bin/bash")

    def _remove_containers(self, image_name: str) -> None:
        client = docker.from_env()
        for name in (f"{image_name}_ic_ctr", f"{image_name}_ic_ctr_eval"):
            try:
                container = client.containers.get(name)
            except docker.errors.NotFound:
                continue
            container.remove(force=True)

def _format_initial_observation(task: InterCodeTask, observation: str) -> str:
    return (
        "InterCode Bash task. Execute shell commands. "
        "When complete, output submit exactly to score the task; submit is not a shell command.\n\n"
        f"Task:\n{observation}"
    )


def _truncate_observation(observation: str, max_chars: int | None) -> tuple[str, dict[str, int] | None]:
    text = str(observation)
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text, None
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    omitted = len(text) - max_chars
    truncated = (
        text[:keep_head].rstrip()
        + f"\n... [truncated {omitted} chars of command output] ...\n"
        + text[-keep_tail:].lstrip()
    )
    return truncated, {"original_chars": len(text), "max_chars": max_chars, "omitted_chars": omitted}


def _get_container_by_name(ctr_name: str, image_name: str, **kwargs):
    client = docker.from_env()
    try:
        container = client.containers.get(ctr_name)
        if container.status != "running":
            container.start()
    except docker.errors.NotFound:
        for key in list(kwargs.keys()):
            if key not in {"command", "environment", "ports", "volumes"}:
                del kwargs[key]
        image = client.images.get(image_name)
        container = client.containers.run(
            image=image,
            name=ctr_name,
            detach=True,
            tty=True,
            **kwargs,
        )
    time.sleep(3)
    return container


def _bash_filesystem_version(task: InterCodeTask) -> int:
    match = re.search(r"_fs([1-4])(?:_|\b)", str(task.task_id))
    if match:
        return int(match.group(1))
    match = re.search(r"nl2bash_fs_([1-4])\\.json$", str(task.data_path))
    if match:
        return int(match.group(1))
    return 1
