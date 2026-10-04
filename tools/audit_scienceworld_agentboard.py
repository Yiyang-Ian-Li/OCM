"""Audit labels and replay native gold paths; never change benchmark data."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.sciworld.process import close_scienceworld_env, make_scienceworld_env
from scienceworld.constants import JAR_PATH

OUT = ROOT / "outputs/diagnostics/scienceworld-agentboard-audit"
LABELS = ROOT / "baselines/WorldEvolver/data/scienceworld/test.jsonl"
SIMPLIFICATION = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in LABELS.read_text().splitlines()]
    snapshots = json.loads((ROOT / "outputs/task-object-graphs/scienceworld-source.json").read_text())
    assert snapshots["jar_sha256"] == digest(JAR_PATH)
    assert snapshots["sha256"] == digest(LABELS)
    assert len(rows) == len(snapshots["instances"]) == 90
    assert len({(r["additional_info"]["env_name"], r["additional_info"]["var"]) for r in rows}) == 90
    url = "https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data/scienceworld/test.jsonl"
    official_bytes = urlopen(url, timeout=30).read()
    official = {r["id"]: r for r in map(json.loads, official_bytes.decode().splitlines())}
    report = {
        "scienceworld_version": importlib.metadata.version("scienceworld"),
        "jar_sha256": digest(JAR_PATH), "labels_sha256": digest(LABELS),
        "official_viewer_url": url,
        "official_viewer_sha256": hashlib.sha256(official_bytes).hexdigest(),
        "simplification": SIMPLIFICATION,
        "caveat": "Native gold paths need not satisfy AgentBoard's additional route requirements. Missing matches alone do not prove label defects.",
        "tasks": [],
    }
    run_root = ROOT / "outputs/predictive_ocm/scienceworld-agentboard"
    runs = [run_root / "ocm-grounded-revision-qwen38-q4-scienceworld-paper-v6",
            run_root / "ocm-knowledge-delta-qwen38-q4-scienceworld-paper-v8"]
    env = make_scienceworld_env("", envStepLimit=100)
    try:
        for row, snapshot in zip(rows, snapshots["instances"]):
            assert row == snapshot["row"]
            task_id = row["id"]
            patterns = row["subgoals"]
            compiled = [re.compile(p) for p in patterns]
            upstream = official[task_id]
            upstream_patterns = [p.strip() for p in re.split(r"Subgoal \d+: ", upstream["subgoals"])[1:]]
            item = {
                "id": task_id, "additional_info": row["additional_info"],
                "goal": row["goal"], "subgoals": patterns,
                "official_goal_equal": row["goal"] == upstream["goal"],
                "official_patterns_equal_after_strip": [p.strip() for p in patterns] == upstream_patterns,
                "native_description": snapshot["native_description"],
                "modifiers": snapshot["modifiers"], "native_goals": snapshot["native_goals"],
                "historical": [],
            }
            for run in runs:
                for path in sorted((run / "tasks").glob(f"agentboard_{task_id:04d}__trial_*.json")):
                    data = json.loads(path.read_text())
                    hits = [False] * len(patterns)
                    discrepancies = []
                    evidence = [None] * len(patterns)
                    for step in data["result"]["step_records"]:
                        obs = step["next_observation"]
                        for idx, pattern in enumerate(compiled):
                            if pattern.search(obs):
                                hits[idx] = True
                                evidence[idx] = {"action": step["action"], "observation": obs}
                        if abs(sum(hits) / len(hits) - step["reward"]) > 1e-8:
                            discrepancies.append(step["step"])
                    item["historical"].append({
                        "run": run.name, "trial": data["task"]["trial"],
                        "success": data["result"]["meta"]["success"],
                        "hits": hits, "matching_evidence": evidence,
                        "reward_discrepancy_steps": discrepancies,
                    })
            start = time.monotonic()
            info = row["additional_info"]
            env.load(info["env_name"], info["var"], simplificationStr=SIMPLIFICATION, generateGoldPath=True)
            gold = env.getGoldActionSequence()
            observation, reset_info = env.reset()
            steps, hits = [], [False] * len(patterns)
            for action in gold:
                observation, _, done, result = env.step(action)
                matched = [i for i, p in enumerate(compiled) if p.search(observation)]
                for idx in matched:
                    hits[idx] = True
                steps.append({"action": action, "observation": observation,
                              "native_score": result["score"], "native_done": done,
                              "matched_subgoals": matched})
            item["gold_replay"] = {
                "seconds": time.monotonic() - start, "steps": steps,
                "native_max_score": max((s["native_score"] for s in steps), default=0),
                "agentboard_reward": sum(hits) / len(hits),
                "unmatched_subgoals": [p for p, hit in zip(patterns, hits) if not hit],
            }
            report["tasks"].append(item)
            (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
            print(task_id, info["env_name"], "native", item["gold_replay"]["native_max_score"],
                  "AB", round(sum(hits) / len(hits), 3), "missing", item["gold_replay"]["unmatched_subgoals"], flush=True)
    finally:
        close_scienceworld_env(env)
    print("Saved", OUT / "audit.json", flush=True)


def probe_counterexamples():
    """Reproduce false negatives in fresh JVMs without gold-generation side effects."""
    report = json.loads((OUT / "audit.json").read_text())
    probes = []
    for task_id in [17, 48, 51, 52, 56, 76]:
        task = report["tasks"][task_id]
        actions = [step["action"] for step in task["gold_replay"]["steps"]]
        if task_id == 48:
            actions = [action for action in actions
                       if not action.startswith("open door") and action not in {"look around", "wait1"}]
            actions[-1] = "move unknown substance B in inventory to yellow box"
        elif task_id == 56:
            actions = [action for action in actions if not action.startswith("focus") and action != "wait1"]
            actions += ["wait"] * 4 + ["look around", "focus on adult mouse"]
        elif task_id == 76:
            path = ROOT / "outputs/predictive_ocm/scienceworld-agentboard/ocm-knowledge-delta-qwen38-q4-scienceworld-paper-v8/tasks/agentboard_0076__trial_0.json"
            actions = [step["action"] for step in json.loads(path.read_text())["result"]["step_records"][:6]]
        env = make_scienceworld_env("", envStepLimit=100)
        try:
            info = task["additional_info"]
            env.load(info["env_name"], info["var"], simplificationStr=SIMPLIFICATION, generateGoldPath=False)
            env.reset()
            hits = [False] * len(task["subgoals"])
            steps = []
            for action in actions:
                obs, _, _, state = env.step(action)
                hits = [hit or bool(re.search(pattern, obs)) for hit, pattern in zip(hits, task["subgoals"])]
                steps.append({"action": action, "observation": obs,
                              "native_score": state["score"], "agentboard_reward": sum(hits) / len(hits)})
            assert steps[-1]["native_score"] == 100
            assert steps[-1]["agentboard_reward"] < 1
            probes.append({"task_id": task_id, "steps": steps})
            print("Confirmed false negative:", task_id, steps[-1]["agentboard_reward"], flush=True)
        finally:
            close_scienceworld_env(env)
    (OUT / "counterexamples.json").write_text(json.dumps(probes, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--probes"]:
        probe_counterexamples()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: audit_scienceworld_agentboard.py [--probes]")
