from __future__ import annotations

import json
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any


class BridgeProcessError(RuntimeError):
    pass


class JsonLineBridge:
    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path,
        startup_timeout: float = 300.0,
        response_timeout: float = 300.0,
    ) -> None:
        self.command = list(command)
        self.cwd = Path(cwd)
        self.response_timeout = float(response_timeout)
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.request({"cmd": "ping"}, timeout=startup_timeout)

    def _drain_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_tail.append(line.rstrip())

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise BridgeProcessError(self._format_error("Bridge process exited before request."))
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        line = self._readline_with_timeout(timeout or self.response_timeout)
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeProcessError(self._format_error(f"Invalid bridge JSON: {line!r}")) from exc
        if not response.get("ok", False):
            raise BridgeProcessError(self._format_error(str(response.get("error") or "Bridge request failed.")))
        return response

    def _readline_with_timeout(self, timeout: float) -> str:
        result: dict[str, str | BaseException] = {}

        def target() -> None:
            try:
                assert self._process.stdout is not None
                result["line"] = self._process.stdout.readline()
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=float(timeout))
        if thread.is_alive():
            self.close()
            raise BridgeProcessError(self._format_error(f"Bridge response timed out after {timeout:.1f}s."))
        if "error" in result:
            raise BridgeProcessError(self._format_error("Bridge stdout read failed.")) from result["error"]  # type: ignore[arg-type]
        line = str(result.get("line") or "")
        if not line:
            raise BridgeProcessError(self._format_error("Bridge process closed stdout."))
        return line

    def _format_error(self, message: str) -> str:
        stderr = "\n".join(self._stderr_tail)
        if stderr:
            return f"{message}\nBridge stderr tail:\n{stderr}"
        return message

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                self._process.stdin.flush()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            self._process.kill()
