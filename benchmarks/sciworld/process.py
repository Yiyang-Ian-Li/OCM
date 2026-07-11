from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import scienceworld.scienceworld as scienceworld_module
from scienceworld import ScienceWorldEnv


def make_scienceworld_env(*args: Any, **kwargs: Any) -> ScienceWorldEnv:
    original_launch_gateway = scienceworld_module.launch_gateway
    process_holder: dict[str, subprocess.Popen] = {}

    def launch_gateway_with_process(*launch_args: Any, **launch_kwargs: Any) -> Any:
        launch_kwargs["return_proc"] = True
        output = original_launch_gateway(*launch_args, **launch_kwargs)
        if isinstance(output, tuple):
            port = output[0]
            process = output[-1]
        else:
            raise RuntimeError("Py4J launch_gateway did not return a process handle.")
        process_holder["process"] = process
        return port

    scienceworld_module.launch_gateway = launch_gateway_with_process
    try:
        env = ScienceWorldEnv(*args, **kwargs)
    finally:
        scienceworld_module.launch_gateway = original_launch_gateway

    process = process_holder.get("process")
    if process is not None:
        setattr(env, "_gateway_process", process)
    return env


def close_scienceworld_env(env: ScienceWorldEnv | None) -> None:
    if env is None:
        return
    gateway = getattr(env, "_gateway", None)
    if gateway is not None:
        callback_server = getattr(gateway, "_callback_server", None)
        if callback_server is not None and getattr(callback_server, "thread", None) is not None:
            _ignore_errors(callback_server.shutdown)
        _ignore_errors(lambda: setattr(gateway, "_callback_server", None))
        _call_if_present(gateway, "shutdown_callback_server")
        _call_if_present(gateway, "shutdown")
        _call_if_present(gateway, "close")
    _terminate_process(getattr(env, "_gateway_process", None))


def _call_if_present(obj: Any, name: str) -> None:
    method = getattr(obj, name, None)
    if callable(method):
        _ignore_errors(method)


def _ignore_errors(callback: Callable[[], Any]) -> None:
    try:
        callback()
    except Exception:
        pass


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
