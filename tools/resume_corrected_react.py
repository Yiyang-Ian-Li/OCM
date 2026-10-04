"""Resume ReAct after the already-running corrected v8 batch exits."""

import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/runtime/batches/scienceworld-corrected-3trials-20261004"


def main():
    predecessor = psutil.Process(int(sys.argv[1]))
    print("Waiting for batch", predecessor.pid, flush=True)
    predecessor.wait()
    state_path = BATCH / "status.json"
    state = json.loads(state_path.read_text())
    job = state["jobs"]["react"]
    command = job["command"] + ["--resume", "--wandb-run-id", "yfqrw193"]
    with (BATCH / "react.resume.console.log").open("w") as log:
        process = subprocess.Popen(command, cwd=state["source"], stdout=log, stderr=subprocess.STDOUT)
        job.update(status="running", pid=process.pid, command=command)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        result = process.wait()
    job.update(status="completed" if result == 0 else "failed", returncode=result)
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("ReAct finished", result, flush=True)


if __name__ == "__main__":
    main()
