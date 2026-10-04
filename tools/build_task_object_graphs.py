"""Build reproducible task/object graphs without running an agent or an LLM."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sre_parse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = ROOT / "outputs/task-object-graphs"
SCI_DATA = ROOT / "baselines/WorldEvolver/data/scienceworld/test.jsonl"
SIMPLIFICATION = "selfWateringFlowerPots,openContainers,openDoors,noElectricalAction"
ROOMS = {"kitchen", "bathroom", "bedroom", "living room", "hallway", "outside",
         "foundry", "workshop", "art studio", "greenhouse", "inventory", "universe"}
IGNORED = ROOMS | {"agent", "nothing", "substance", "living thing", "organism", "room"}
# Chemistry.scala creates GreenBluePaint but describes it with this alternate name.
CHEMISTRY_ALIASES = {"blue green paint": "green-blue paint"}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(name):
    """Keep benchmark nouns; merge only spelling/case and explicit state adjectives."""
    name = re.sub(r"\s+", " ", str(name).lower()).strip(" .'")
    name = re.sub(r"^(?:a |an |the )", "", name)
    name = re.sub(r"^(?:solid |liquid |gaseous )", "", name)
    if " containing " in name:
        name = name.split(" containing ")[0]
    return CHEMISTRY_ALIASES.get(name, name)


def regex_variants(pattern):
    """Expand textual alternatives, replacing numeric regex constructs with a gap."""
    def expand(tokens):
        result = [""]
        for op, value in tokens:
            if op == sre_parse.LITERAL:
                parts = [chr(value)]
            elif op == sre_parse.SUBPATTERN:
                parts = expand(value[-1])
            elif op == sre_parse.BRANCH:
                parts = [s for branch in value[1] for s in expand(branch)]
            elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
                lo, hi, child = value
                parts = expand(child) if lo == hi == 1 else [" "]
            else:
                parts = [" "]
            result = [a + b for a in result for b in parts]
            if len(result) > 256:
                raise ValueError(f"Too many textual alternatives: {pattern}")
        return sorted(set(result))
    return expand(sre_parse.parse(pattern))


def match_names(text, vocabulary):
    """Longest non-overlapping noun matches avoid orange inside orange juice."""
    text = text.lower()
    candidates = []
    for name in vocabulary:
        plural = "(?:es)?" if name.endswith(("box", "brush")) else "s?"
        for m in re.finditer(r"(?<![\w-])" + re.escape(name) + plural + r"(?![\w-])", text):
            candidates.append((m.start(), m.end(), name))
    occupied = set()
    result = set()
    for start, end, name in sorted(candidates, key=lambda x: (-(x[1]-x[0]), x[0])):
        positions = set(range(start, end))
        if not occupied.intersection(positions):
            occupied.update(positions)
            result.add(normalize(name))
    return result - IGNORED


def score_nouns(text):
    """Collect explicit object slots from the benchmark's observation templates."""
    patterns = [
        r"^You focus on (.+?)\.?$",
        r"^You move (?:the )?(.+?) to (?:the )?(.+?)\.?$",
        r"^You pour (?:the )?(?:contents of (?:the )?)?(.+?) into (?:the )?(.+?)\.?$",
        r"^The (.+?) is now (?:open|activated|closed|deactivated)\.?$",
        r"^You mix (?:the )?(.+?)\.?$",
        r"^a substance called (.+?)\.?$",
    ]
    out = set()
    for pattern in patterns:
        m = re.match(pattern, text, flags=re.I)
        if m:
            out.update(normalize(x) for x in m.groups())
    if re.match(r"^(solid|liquid|gaseous) ", text, re.I):
        out.add(normalize(text))
    if "thermometer measures" in text.lower():
        out.add("thermometer")
    if "recipe reads" in text.lower():
        out.add("recipe")
    if re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", text.strip()) and not out:
        out.add(normalize(text))
    # Shared suffix in paint observations: "Blue and yellow paint mix ...".
    m = re.search(r"([\w-]+) and ([\w-]+) paint mix to produce ([\w -]+) paint", text, re.I)
    if m:
        out.update(normalize(x + " paint") for x in m.groups())
    return out - IGNORED


def collect_sciworld(out):
    from benchmarks.sciworld.process import make_scienceworld_env, close_scienceworld_env
    from scienceworld.constants import JAR_PATH

    rows = [json.loads(line) for line in SCI_DATA.read_text().splitlines()]
    snapshots = []
    env = make_scienceworld_env("", envStepLimit=1)
    try:
        for row in rows:
            info = row["additional_info"]
            definition = env.server.taskMaker().getTask(info["env_name"]).get()
            modifiers = []
            for mod in definition.getCombination(info["var"]):
                kind = mod.getClass().getSimpleName()
                item = {"type": kind}
                if kind in {"TaskObject", "TaskDisable"}:
                    item["name"] = mod.name()
                elif kind in {"TaskValueStr", "TaskValueDouble"}:
                    item.update(key=mod.key(), value=mod.value())
                modifiers.append(item)
            native_goals = []
            for goal in definition.setupGoals(info["var"]).goalSequence().subgoals():
                fields = {}
                for field in goal.getClass().getDeclaredFields():
                    if not field.isSynthetic():
                        field.setAccessible(True)
                        value = field.get(goal)
                        fields[field.getName()] = str(value)
                native_goals.append({"type": goal.getClass().getSimpleName(), "fields": fields})
            env.load(info["env_name"], info["var"], simplificationStr=SIMPLIFICATION,
                     generateGoldPath=False)
            _, reset_info = env.reset()
            lut = env.getAllObjectIdsTypesReferentsLUTJSON()
            root = env.server.agentInterface().get().universe()
            pending, seen, objects = [root], set(), []
            while pending:
                obj = pending.pop()
                uid = obj.uuid()
                if uid in seen:
                    continue
                seen.add(uid)
                entry = {"id": uid, "name": obj.name(),
                         "class": obj.getClass().getSimpleName(),
                         "referents": lut.get(str(uid), {}).get("referents", [])}
                if obj.propLife().isDefined():
                    lifecycle = obj.lifecycle()
                    if lifecycle.isDefined():
                        stage_it = lifecycle.get().stages().iterator()
                        entry["life_stage_names"] = []
                        while stage_it.hasNext():
                            stage = stage_it.next()
                            stage_name = str(stage.stageName())
                            name = str(obj.name())
                            alias = stage.getCanonicalName() if stage.hasCanonicalName() else (
                                f"{name} egg" if stage_name == "egg" else f"{stage_name} {name}")
                            entry["life_stage_names"].append(str(alias))
                objects.append(entry)
                it = obj.getContainedObjects(False).iterator()
                while it.hasNext():
                    pending.append(it.next())
            snapshots.append({"row": row, "modifiers": modifiers, "native_goals": native_goals,
                              "native_description": reset_info.get("taskDesc"), "objects": objects})
            if len(snapshots) % 10 == 0:
                print(f"ScienceWorld: inspected {len(snapshots)}/{len(rows)} instances", flush=True)
    finally:
        close_scienceworld_env(env)
    data = {"source": str(SCI_DATA.relative_to(ROOT)), "sha256": sha256(SCI_DATA),
            "jar_sha256": sha256(Path(JAR_PATH)), "simplification": SIMPLIFICATION,
            "instances": snapshots}
    dump(out / "scienceworld-source.json", data)
    return data


class Graph:
    def __init__(self, benchmark, semantics, source):
        self.data = {"schema_version": 1, "benchmark": benchmark, "semantics": semantics,
                     "source": source, "tasks": [], "objects": [], "links": []}
        self.links = {}

    def link(self, task, obj, role, evidence):
        key = (task, obj)
        edge = self.links.setdefault(key, {"task": task, "object": obj, "roles": [], "evidence": []})
        if role not in edge["roles"]:
            edge["roles"].append(role)
        entry = {"role": role, **evidence}
        if entry not in edge["evidence"]:
            edge["evidence"].append(entry)

    def finish(self, active_roles):
        import networkx as nx
        self.data["links"] = sorted(self.links.values(), key=lambda x: (x["task"], x["object"]))
        names = sorted({e["object"] for e in self.links.values()})
        self.data["objects"] = [{"id": name, "label": name.replace("_", " ")} for name in names]
        self.data["default_roles"] = sorted(active_roles)
        active = [e for e in self.links.values() if active_roles.intersection(e["roles"])]
        graph = nx.Graph()
        graph.add_nodes_from("t:" + t["id"] for t in self.data["tasks"])
        graph.add_edges_from(("t:" + e["task"], "o:" + e["object"]) for e in active)
        self.data["layout"] = component_layout(graph)
        full = nx.Graph()
        full.add_nodes_from("t:" + t["id"] for t in self.data["tasks"])
        full.add_edges_from(("t:" + e["task"], "o:" + e["object"]) for e in self.links.values())
        self.data["layout_all"] = component_layout(full)
        self.data["stats"] = {"tasks": len(self.data["tasks"]), "objects": len(names),
                              "links": len(self.links), "default_objects": len({e["object"] for e in active}),
                              "default_links": len(active), "components": nx.number_connected_components(graph)}
        assert len({t["id"] for t in self.data["tasks"]}) == len(self.data["tasks"])
        assert all(e["evidence"] for e in self.links.values())
        return self.data


def component_layout(graph):
    """Pack independently laid-out components so disconnected nodes do not shrink the core."""
    import networkx as nx
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c))
    positions = {}

    def pack(groups, x, y, w, h):
        if len(groups) == 1:
            ids = groups[0]
            if len(ids) == 1:
                positions[ids[0]] = [x + w/2, y + h/2]
                return
            local = nx.spring_layout(graph.subgraph(ids), seed=42, iterations=180)
            lo = [min(p[d] for p in local.values()) for d in (0, 1)]
            span = [max(p[d] for p in local.values()) - lo[d] for d in (0, 1)]
            for node, p in local.items():
                positions[node] = [
                    round(x + w*(0.1 + 0.8*(float(p[0])-lo[0])/span[0]), 5) if span[0] else x+w/2,
                    round(y + h*(0.1 + 0.8*(float(p[1])-lo[1])/span[1]), 5) if span[1] else y+h/2,
                ]
            return
        total = sum(map(len, groups))
        cut = min(range(1, len(groups)), key=lambda i: abs(sum(map(len, groups[:i]))-total/2))
        fraction = sum(map(len, groups[:cut]))/total
        if w*1.8 >= h:
            pack(groups[:cut], x, y, w*fraction, h)
            pack(groups[cut:], x+w*fraction, y, w*(1-fraction), h)
        else:
            pack(groups[:cut], x, y, w, h*fraction)
            pack(groups[cut:], x, y+h*fraction, w, h*(1-fraction))
    pack(components, -1, -1, 2, 2)
    return positions


def build_sciworld(source):
    vocabulary = set()
    runtime_names = set()
    for snap in source["instances"]:
        for obj in snap["objects"]:
            vocabulary.add(normalize(obj["name"]))
            vocabulary.update(normalize(n) for n in obj.get("life_stage_names", []))
            runtime_names.add(normalize(obj["name"]))
            runtime_names.update(normalize(n) for n in obj.get("life_stage_names", []))
    vocabulary.update(CHEMISTRY_ALIASES)
    for snap in source["instances"]:
        for mod in snap["modifiers"]:
            if "name" in mod:
                vocabulary.add(normalize(mod["name"]))
            elif mod["type"] == "TaskValueStr" and mod["key"] in {
                "objectName", "inputChemicals", "inputChemicalsSecondary", "inputChemicalsTertiary",
                "secondaryColor", "tertiaryColor", "animal", "plant", "substanceName",
            }:
                vocabulary.update(normalize(v) for v in mod["value"].split(","))
        for pattern in snap["row"]["subgoals"]:
            for variant in regex_variants(pattern):
                vocabulary.update(score_nouns(variant))
        for goal in snap["native_goals"]:
            for key, value in goal["fields"].items():
                if key in {"objectName", "containerName", "speciesName", "livingThingType"}:
                    vocabulary.add(normalize(value))
    vocabulary -= IGNORED | {""}
    graph = Graph("ScienceWorld", {
        "score_required": "Named in every textual branch of at least one scored observation pattern; not a physical-necessity proof.",
        "score_alternative": "Named in some, but not all, textual branches of a scored observation pattern.",
        "task_mentioned": "Mentioned in the AgentBoard task instruction; availability does not imply necessity.",
        "native_goal": "Explicit object name in the underlying simulator goal; may differ from AgentBoard.",
        "setup": "TaskObject introduced by the generator; may be a distractor.",
        "identity": "Case and explicit solid/liquid/gaseous prefixes normalized. Unambiguous life-stage aliases use the instance's runtime lifecycle. Chemistry's blue green paint maps to green-blue paint. Ice and water remain distinct benchmark nouns.",
    }, {k: v for k, v in source.items() if k != "instances"})
    audit = []
    for snap in source["instances"]:
        row = snap["row"]
        tid = f"agentboard_{row['id']:04d}"
        aliases = defaultdict(set)
        for obj in snap["objects"]:
            for alias in obj.get("life_stage_names", []):
                aliases[normalize(alias)].add(normalize(obj["name"]))

        def canonical(name):
            candidates = aliases.get(name, set())
            return next(iter(candidates)) if len(candidates) == 1 else name

        def add(name, role, evidence):
            graph.link(tid, canonical(name), role, {**evidence, "matched_name": name,
                       "identity_basis": "runtime lifecycle" if canonical(name) != name else "literal name"})
        task = {"id": tid, "index": row["id"], "label": f"{row['id']:02d} · {row['additional_info']['env_name']} · v{row['additional_info']['var']}",
                "family": row["additional_info"]["env_name"], "variation": row["additional_info"]["var"],
                "goal": row["goal"], "subgoals": row["subgoals"], "difficulty": row["difficulty"],
                "impossible": False}
        graph.data["tasks"].append(task)
        for index, pattern in enumerate(row["subgoals"]):
            variants = regex_variants(pattern)
            sets = [match_names(v, vocabulary) | score_nouns(v) for v in variants]
            common, union = set.intersection(*sets), set.union(*sets)
            for obj in sorted(union):
                role = "score_required" if obj in common else "score_alternative"
                add(obj, role, {"source": f"test.jsonl row {row['id']}, subgoals[{index}]", "text": pattern})
                if obj not in runtime_names and re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", pattern.strip()):
                    audit.append({"task": tid, "subgoal_index": index, "pattern": pattern,
                                  "reason": "score-only-name", "object": obj})
            if not union:
                audit.append({"task": tid, "subgoal_index": index, "pattern": pattern,
                              "reason": "location-only" if "You move to " in pattern else "unmapped"})
        for obj in sorted(match_names(row["goal"], vocabulary)):
            add(obj, "task_mentioned", {"source": f"test.jsonl row {row['id']}, goal", "text": row["goal"]})
        for i, goal in enumerate(snap["native_goals"]):
            for key, value in goal["fields"].items():
                if key in {"objectName", "containerName", "speciesName", "livingThingType"} and normalize(value) not in IGNORED:
                    add(normalize(value), "native_goal", {"source": f"native_goals[{i}].{key}", "text": value})
        for i, mod in enumerate(snap["modifiers"]):
            if mod["type"] == "TaskObject" and normalize(mod["name"]) not in IGNORED:
                add(normalize(mod["name"]), "setup", {"source": f"modifiers[{i}].name", "text": mod["name"]})
    graph.data["audit"] = audit
    return graph.finish({"score_required", "score_alternative", "task_mentioned"})


def build_plancraft():
    from plancraft.simple import get_plancraft_examples
    from plancraft.environment.planner import get_ancestors
    from plancraft.environment.recipes import RECIPES, ShapedRecipe, ShapelessRecipe, SmeltingRecipe, id_to_item

    def recipe_requirements(recipe):
        if isinstance(recipe, ShapedRecipe):
            return {"kind": "shaped", "grid": [
                [[id_to_item(i) for i in sorted(cell)] for cell in row] for row in recipe.kernel]}
        if isinstance(recipe, ShapelessRecipe):
            return {"kind": "shapeless", "ingredient_variants": recipe.ingredients}
        if isinstance(recipe, SmeltingRecipe):
            return {"kind": "smelting", "one_of": sorted(recipe.ingredient), "count": 1}
        raise TypeError(type(recipe))

    examples = get_plancraft_examples("test.small")
    graph = Graph("PlanCraft", {
        "input": "Consumed item type on the provided reference solution, present in initial inventory.",
        "intermediate": "Produced non-target item type on the provided reference solution.",
        "target": "Requested output item; for impossible tasks this is not an achieved output.",
        "dependency_available": "Impossible task: recipe-ancestor item present in inventory; quantity may be insufficient.",
        "dependency_absent": "Impossible task: absent recipe-ancestor candidate, not necessarily required on every alternative route.",
        "necessity": "Reference-path incidence, not a proof of uniqueness. Alternative recipe branches are retained separately.",
    }, {"package": "plancraft", "version": importlib.metadata.version("plancraft"), "split": "test.small"})
    for index, ex in enumerate(examples):
        tid = ex.id
        task = {"id": tid, "index": index, "label": f"{tid} · {ex.target}", "family": ex.complexity_split,
                "goal": f"Craft {ex.target}", "target": ex.target, "impossible": ex.impossible,
                "inventory": ex.inventory, "steps": [], "optimal_path_length": ex.optimal_path_length}
        graph.data["tasks"].append(task)
        graph.link(tid, ex.target, "target", {"source": "target", "text": ex.target})
        if ex.impossible:
            candidates = sorted(set(get_ancestors(ex.target)))
            for obj in candidates:
                qty = ex.inventory.get(obj, 0)
                graph.link(tid, obj, "dependency_available" if qty else "dependency_absent",
                           {"source": "recipe dependency graph + initial inventory", "quantity": qty,
                            "text": f"{obj}: initial quantity {qty}; ancestor candidate, not proven necessary"})
            task["recipe_dependencies"] = [
                {"output": output, "recipe_index": i, "result_count": r.result.count,
                 "input_candidates": sorted(r.inputs), "recipe_type": r.recipe_type,
                 "requirements": recipe_requirements(r)}
                for output in candidates + [ex.target] for i, r in enumerate(RECIPES.get(output, []))
            ]
            continue
        seq = [ex.inventory] + ex.inventory_trace
        assert len(ex.inventory_trace) == ex.optimal_path_length
        assert seq[-1].get(ex.target, 0) > 0
        for step, (before, after) in enumerate(zip(seq, seq[1:]), 1):
            consumed = {o: before[o] - after.get(o, 0) for o in before if before[o] > after.get(o, 0)}
            produced = {o: after[o] - before.get(o, 0) for o in after if after[o] > before.get(o, 0)}
            assert consumed and produced
            task["steps"].append({"step": step, "consumed": consumed, "produced": produced})
            for obj, qty in consumed.items():
                role = "input" if obj in ex.inventory else "intermediate"
                graph.link(tid, obj, role, {"source": f"inventory_trace[{step-1}]", "step": step,
                                          "quantity": qty, "text": f"step {step}: consumed {qty} {obj}"})
            for obj, qty in produced.items():
                role = "target" if obj == ex.target else "intermediate"
                graph.link(tid, obj, role, {"source": f"inventory_trace[{step-1}]", "step": step,
                                          "quantity": qty, "text": f"step {step}: produced {qty} {obj}"})
    graph.data["source"]["examples_sha256"] = hashlib.sha256(
        json.dumps([e.model_dump() for e in examples], sort_keys=True).encode()).hexdigest()
    return graph.finish({"input", "intermediate", "target", "dependency_available"})


def write_visualization(out, name, graph):
    template = (Path(__file__).with_name("task_object_graph.html")).read_text()
    # The full graph remains in JSON; the inline view does not need recipe alternatives.
    visual = {**graph, "tasks": [{k: v for k, v in t.items() if k != "recipe_dependencies"}
                                for t in graph["tasks"]]}
    data = json.dumps(visual, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    fragment = (template.replace("__ROOT__", f"{name}-object-network")
                .replace("__TITLE__", graph["benchmark"])
                .replace("__DEFAULT_LABEL__", "评分 + 任务文本" if name == "scienceworld" else "参考解 + 不可解任务的库存依赖")
                .replace("__DATA__", data))
    assert len(fragment.encode()) < 1_000_000
    (out / f"{name}-network.html").write_text(fragment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-source", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "scienceworld-source.json"
    source = collect_sciworld(out) if args.refresh_source or not cache.exists() else json.loads(cache.read_text())
    for name, graph in [("scienceworld", build_sciworld(source)), ("plancraft", build_plancraft())]:
        dump(out / f"{name}-graph.json", graph)
        write_visualization(out, name, graph)
        print(name, json.dumps(graph["stats"]), flush=True)


if __name__ == "__main__":
    main()
