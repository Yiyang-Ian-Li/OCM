"""Run the corrected three-trial ReAct/v8 pair sequentially on one model server."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/runtime/batches/scienceworld-corrected-3trials-20261004"
SOURCE = BATCH / "source"
WORKTREE = ROOT / "outputs/code-grounded-revision-v6"


def main():
    BATCH.mkdir(parents=True, exist_ok=False)
    SOURCE.mkdir()
    for name in ("agents", "benchmarks", "runner"):
        shutil.copytree(WORKTREE / name, SOURCE / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("main.py", "config.py", "requirements.txt", "README.md"):
        shutil.copy2(WORKTREE / name, SOURCE / name)
    vendor = SOURCE / "baselines/WorldEvolver"
    shutil.copytree(ROOT / "baselines/WorldEvolver/src", vendor / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (vendor / "data/scienceworld").mkdir(parents=True)
    shutil.copy2(ROOT / "baselines/WorldEvolver/data/scienceworld/test.jsonl", vendor / "data/scienceworld/test.jsonl")
    common = [
        sys.executable, "-u", str(SOURCE / "main.py"),
        "--benchmark", "sciworld", "--split", "agentboard_corrected", "--num-trials", "3",
        "--provider", "llamacpp", "--model-name", "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "--base-url", "http://127.0.0.1:8080/v1", "--temperature", "0", "--top-p", "0.5", "--seed", "42",
        "--sciworld-simplification", "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction",
        "--sciworld-env-step-limit", "100", "--sciworld-agent-step-limit", "30",
        "--wandb-project", "ocm-worldevolver-paper-protocol",
        "--wandb-entity", "l1y1yang-university-of-notre-dame",
    ]
    jobs = [
        ("react", ["--agent", "react", "--react-implementation", "worldevolver", "--timeout", "300",
                   "--output-dir", str(ROOT / "outputs/baselines/react/sciworld"),
                   "--run-name", "react-qwen38-q4-agentboard-corrected-3trials"]),
        ("v8", ["--agent", "ocm", "--timeout", "120",
                "--output-dir", str(ROOT / "outputs/predictive_ocm/scienceworld-agentboard-corrected"),
                "--run-name", "ocm-v8-qwen38-q4-agentboard-corrected-3trials"]),
    ]
    state = {"source": str(SOURCE), "jobs": {name: {"status": "queued", "command": common + args}
                                             for name, args in jobs}}
    status_path = BATCH / "status.json"
    status_path.write_text(json.dumps(state, indent=2) + "\n")
    for name, args in jobs:
        print("Starting", name, flush=True)
        with (BATCH / (name + ".console.log")).open("w") as log:
            process = subprocess.Popen(common + args, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            state["jobs"][name].update(status="running", pid=process.pid)
            status_path.write_text(json.dumps(state, indent=2) + "\n")
            result = process.wait()
        state["jobs"][name].update(status="completed" if result == 0 else "failed", returncode=result)
        status_path.write_text(json.dumps(state, indent=2) + "\n")
        print("Finished", name, "exit", result, flush=True)


if __name__ == "__main__":
    main()
