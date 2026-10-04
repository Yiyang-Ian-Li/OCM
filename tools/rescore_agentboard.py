"""Replay fixed historical actions under corrected scoring, without any LLM calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "outputs/code-grounded-revision-v6"
sys.path.insert(0, str(CODE))

from benchmarks.sciworld.agentboard import CorrectedAgentBoard, read_event
from benchmarks.sciworld.env import SciWorldEnv
from benchmarks.sciworld.process import close_scienceworld_env, make_scienceworld_env
from benchmarks.sciworld.task import SciWorldTask

SIMPLIFICATION = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"
OUT = ROOT / "outputs/diagnostics/scienceworld-agentboard-corrected"


def write_summary(results):
    summary = {
        "protocol": "agentboard_corrected_v1",
        "interpretation": "Fixed historical trajectories, not new online experiments. Reported posthoc metrics use recorded observations; live replay is a validation and can differ in simulator physics or ambiguous action resolution.",
        "live_success_witness_tasks": len({r["task_id"] for r in results if r["corrected_success"]}),
        "live_success_witness_tasks_within_30_actions": len({r["task_id"] for r in results
                                                             if r["first_success_step"] is not None
                                                             and r["first_success_step"] <= 30}),
        "versions": {},
    }
    for version in sorted(v for v in {r["version"] for r in results} if not v.startswith("gold")):
        rows = [r for r in results if r["version"] == version]
        entry = {
            "episodes": len(rows),
            "old_successes": sum(r["old_success"] for r in rows),
            "posthoc_successes": sum(r["recorded_corrected_success"] for r in rows),
            "old_average_reward": sum(r["old_reward"] for r in rows) / len(rows),
            "posthoc_average_reward": sum(r["recorded_corrected_reward"] for r in rows) / len(rows),
            "live_replay_successes": sum(r["corrected_success"] for r in rows),
            "old_action_steps": sum(r["old_steps"] for r in rows),
            "actions_after_posthoc_success": sum(r["old_steps"] - (r["recorded_first_success_step"] or r["old_steps"]) for r in rows),
            "posthoc_lost_successes": sum(r["old_success"] and not r["recorded_corrected_success"] for r in rows),
            "semantic_scorer_replay_disagreements": [
                [r["task_id"], r["trial"]] for r in rows
                if (r["family"].startswith("lifespan-") or r["family"] == "use-thermometer")
                and r["corrected_reward"] != r["recorded_corrected_reward"]],
        }
        for group in ("lifespan", "other"):
            subset = [r for r in rows if r["family"].startswith("lifespan-") == (group == "lifespan")]
            entry[group] = {"episodes": len(subset),
                            "posthoc_successes": sum(r["recorded_corrected_success"] for r in subset)}
        summary["versions"][version] = entry
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def replay_batch(jobs):
    tasks, _ = SciWorldTask.load_tasks("agentboard_corrected")
    tasks = list(tasks)
    audit = json.loads((ROOT / "outputs/diagnostics/scienceworld-agentboard-audit/audit.json").read_text())
    raw = make_scienceworld_env("", envStepLimit=100)
    results = []
    try:
        for number, (version, index, filename) in enumerate(jobs):
            if version.startswith("gold"):
                data = None
                steps = audit["tasks"][index]["gold_replay"]["steps"]
                if version == "gold_compact":
                    steps = [step for step in steps if not step["action"].startswith("open door")]
            else:
                data = json.loads(Path(filename).read_text())
                steps = data["result"]["step_records"]
            environment = SciWorldEnv(tasks[index], env=raw, simplification_str=SIMPLIFICATION, agent_step_limit=30)
            environment.reset()
            scorer = environment._corrected_scorer
            recorded_scorer = CorrectedAgentBoard(scorer.name, scorer.patterns, scorer.values, scorer.targets)
            release_patterns = audit["tasks"][index]["subgoals"]
            release_hits = [False] * len(release_patterns)
            agent = raw.server.agentInterface().get().agent()
            first_success = None
            recorded_first_success = None
            mismatches, semantic_mismatches = [], []
            trace = []
            for step_index, step in enumerate(steps, 1):
                since = agent.getActionHistorySize()
                result = environment.step(step["action"])
                event = read_event(agent, since) if scorer.semantic else None
                observation = result.observation
                release_hits = [hit or bool(re.search(pattern, observation))
                                for hit, pattern in zip(release_hits, release_patterns)]
                recorded_observation = step.get("next_observation", step.get("observation"))
                recorded_scorer.update(recorded_observation, event)
                if result.success and first_success is None:
                    first_success = step_index
                if all(recorded_scorer.progress) and recorded_first_success is None:
                    recorded_first_success = step_index
                if observation != recorded_observation:
                    mismatch = {"step": step_index, "action": step["action"],
                                "recorded": recorded_observation, "replayed": observation}
                    mismatches.append(mismatch)
                    if event is not None:
                        semantic_mismatches.append(mismatch)
                trace.append({"step": step_index, "action": step["action"],
                              "observation": observation, "event": event,
                              "reward": result.reward,
                              "recorded_reward": sum(recorded_scorer.progress) / len(recorded_scorer.progress),
                              "native_score": result.info.get("score")})
                if first_success is not None and recorded_first_success is not None:
                    break
            record = {
                "version": version, "task_id": index, "family": tasks[index].sub_task_name,
                "trial": None if data is None else data["task"]["trial"],
                "old_reward": None if data is None else data["result"]["meta"]["reward"],
                "old_success": None if data is None else data["result"]["meta"]["success"],
                "old_steps": len(steps), "corrected_reward": result.reward,
                "corrected_success": first_success is not None, "first_success_step": first_success,
                "recorded_corrected_reward": sum(recorded_scorer.progress) / len(recorded_scorer.progress),
                "recorded_corrected_success": recorded_first_success is not None,
                "recorded_first_success_step": recorded_first_success,
                "release_reward_on_replay": sum(release_hits) / len(release_hits),
                "unmatched": [p for p, hit in zip(scorer.patterns, scorer.progress) if not hit],
                "mismatch_count": len(mismatches), "semantic_mismatches": semantic_mismatches,
                "trace": trace,
            }
            results.append(record)
            if number % 10 == 0:
                print(version, "batch progress", number + 1, "/", len(jobs), "task", index, flush=True)
    finally:
        close_scienceworld_env(raw)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="*", default=["v6", "v8"])
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.gold:
        jobs += [("gold", index, None) for index in range(90)]
    run_root = ROOT / "outputs/predictive_ocm/scienceworld-agentboard"
    for version in args.versions:
        runs = list(run_root.glob("*-" + version))
        assert len(runs) == 1, runs
        for filename in sorted((runs[0] / "tasks").glob("*.json")):
            index = int(filename.stem.split("__")[0].split("_")[-1])
            jobs.append((version, index, str(filename)))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(replay_batch, jobs[offset::args.workers]) for offset in range(args.workers)]
        for future in as_completed(pending):
            results.extend(future.result())
            (OUT / "replay.json").write_text(json.dumps(results, indent=2) + "\n")
    results.sort(key=lambda r: (r["version"], r["task_id"], r["trial"] or 0))
    if args.gold:
        witnessed = {r["task_id"] for r in results
                     if r["first_success_step"] is not None and r["first_success_step"] <= 30}
        results.extend(replay_batch([("gold_compact", index, None) for index in range(90) if index not in witnessed]))
    (OUT / "replay.json").write_text(json.dumps(results, indent=2) + "\n")
    write_summary(results)
    for version in sorted({r["version"] for r in results}):
        subset = [r for r in results if r["version"] == version]
        print(version, "n", len(subset), "SR", sum(r["corrected_success"] for r in subset) / len(subset),
              "reward", sum(r["corrected_reward"] for r in subset) / len(subset), flush=True)


if __name__ == "__main__":
    main()
