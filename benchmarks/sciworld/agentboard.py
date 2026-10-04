"""Minimal corrections to AgentBoard's released ScienceWorld evaluation.

The original labels remain available through agentboard_test. Semantic checks
read simulator objects only inside the evaluator, never in the agent context.
"""

from __future__ import annotations

import copy
import re

PROTOCOL = "agentboard_corrected_v1"


def correct_label(item: dict) -> dict:
    item = copy.deepcopy(item)
    task_id = int(item["id"])
    if task_id == 11:
        item["goal"] = item["goal"].replace("below 50.0", "below 10.0")
    if task_id == 17:
        item["subgoals"] = [p.removeprefix(",") for p in item["subgoals"]]
    if task_id == 51:
        item["subgoals"][1] = "You focus on the baby brown bear."
    if task_id == 52:
        item["subgoals"][1] = "You focus on the giant tortoise egg."
    if task_id == 76:
        item["subgoals"] = [p.replace("(Sherlock Holmes)", r"\(Sherlock Holmes\)")
                           for p in item["subgoals"]]
    return item


def read_event(agent, since: int) -> dict | None:
    """Ignore automatic look/inventory/task calls in ScienceWorld.step()."""
    for action in agent.getActionHistorySince(since):
        kind = action.getClass().getSimpleName()
        if kind not in {"ActionFocus", "ActionUseDevice", "ActionMoveObject"}:
            continue
        bindings = {}
        assignments = action.assignments()
        keys = assignments.keysIterator()
        while keys.hasNext():
            key = keys.next()
            obj = assignments.apply(key)
            bindings[key] = {"id": obj.uuid(), "name": obj.name()}
        return {"kind": kind, "bindings": bindings}
    return None


class CorrectedAgentBoard:
    def __init__(self, name: str, patterns: list[str], values: dict, targets: dict[str, set[int]]):
        self.name = name
        self.patterns = patterns
        self.values = values
        self.targets = targets
        self.progress = [False] * len(patterns)
        self.measurements: dict[int, float] = {}
        self.focus_indices = [i for i, p in enumerate(patterns) if p.startswith("You focus on")]
        self.semantic = name.startswith("lifespan-") or name == "use-thermometer"

    @classmethod
    def from_env(cls, task, env):
        values, targets = {}, {}
        if task.sub_task_name.startswith("lifespan-") or task.sub_task_name == "use-thermometer":
            definition = env.server.taskMaker().getTask(task.sub_task_name).get()
            for modifier in definition.getCombination(task.variation_idx):
                if modifier.getClass().getSimpleName() in {"TaskValueStr", "TaskValueDouble"}:
                    values[modifier.key()] = modifier.value()
            names = {values[key] for key in ("long", "short", "objectName") if key in values}
            targets = {name: set() for name in names}
            objects = env.server.agentInterface().get().universe().getContainedObjectsRecursive().iterator()
            while objects.hasNext():
                obj = objects.next()
                name = obj.name()
                if name not in targets:
                    continue
                parent = obj.getContainer()
                while parent.isDefined():
                    if parent.get().name() == values["location"]:
                        targets[name].add(obj.uuid())
                        break
                    parent = parent.get().getContainer()
        return cls(task.sub_task_name, task.subgoals, values, targets)

    def update(self, observation: str, event: dict | None = None) -> None:
        matches = [bool(re.search(pattern, observation)) for pattern in self.patterns]
        bindings = event["bindings"] if event else {}
        kind = event["kind"] if event else None
        if self.name.startswith("lifespan-"):
            roles = ["long", "short"] if "then" in self.name else [
                "short" if self.name == "lifespan-shortest-lived" else "long"]
            for position, (index, role) in enumerate(zip(self.focus_indices, roles)):
                obj = bindings.get("obj", {})
                matches[index] = bool(
                    kind == "ActionFocus" and observation.startswith("You focus on the ")
                    and obj.get("id") in self.targets[self.values[role]]
                    and (position == 0 or self.progress[self.focus_indices[position - 1]])
                )
        elif self.name == "use-thermometer":
            target_ids = self.targets[self.values["objectName"]]
            reading = re.fullmatch(r"the thermometer measures a temperature of (-?\d+) degrees celsius", observation)
            patient = bindings.get("patient", {})
            measured_target = bool(
                kind == "ActionUseDevice" and reading
                and bindings["device"]["name"] == "thermometer"
                and patient.get("id") in target_ids
            )
            if measured_target:
                self.measurements[patient["id"]] = float(reading[1])
            for index, pattern in enumerate(self.patterns):
                if "thermometer measures" in pattern:
                    matches[index] = measured_target
                elif pattern.startswith("You move the ") and " box" in pattern:
                    obj = bindings.get("obj", {})
                    temperature = self.measurements.get(obj.get("id"))
                    box = bindings.get("moveTo", {}).get("name")
                    threshold = self.values["temperaturePoint"]
                    matches[index] = bool(
                        kind == "ActionMoveObject" and observation.startswith("You move the ")
                        and obj.get("id") in target_ids and temperature is not None
                        and ((box == self.values["boxAbove"] and temperature >= threshold)
                             or (box == self.values["boxBelow"] and temperature <= threshold))
                    )
        self.progress[:] = [old or new for old, new in zip(self.progress, matches)]
