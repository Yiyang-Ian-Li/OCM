from __future__ import annotations

import importlib.util
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .knowledge_code import (
    build_signature_index,
    dedupe_top_level_imports,
    extract_definitions,
    merge_top_level_definitions,
    remove_top_level_definitions,
    remove_top_level_self_imports,
    top_level_definition_names,
)
from .json_utils import load_jsonish
from .openai_apply_patch import (
    identify_files_added,
    identify_files_needed,
    process_patch,
)


class KnowledgeStore:
    def __init__(
        self,
        *,
        root: Path,
        object_knowledge_path: Path,
        object_signature_path: Path,
        procedural_dir: Path,
        procedural_index_path: Path,
    ) -> None:
        self.root = root
        self.object_knowledge_path = object_knowledge_path
        self.object_signature_path = object_signature_path
        self.procedural_dir = procedural_dir
        self.procedural_index_path = procedural_index_path

    @classmethod
    def create(cls, output_dir: Path, load_path: str | None = None) -> "KnowledgeStore":
        root = output_dir / "ocm_agent" / "knowledge"
        root.mkdir(parents=True, exist_ok=True)
        store = cls(
            root=root,
            object_knowledge_path=root / "object_knowledge.py",
            object_signature_path=root / "object_knowledge_signatures.md",
            procedural_dir=root / "procedural_knowledge",
            procedural_index_path=root / "procedural_knowledge" / "index.py",
        )
        store.procedural_dir.mkdir(parents=True, exist_ok=True)
        if load_path and not store.object_knowledge_path.exists():
            store._load_from(Path(load_path).expanduser())
        if not store.object_knowledge_path.exists():
            store.object_knowledge_path.write_text("", encoding="utf-8")
        store.refresh_indexes()
        return store

    def _load_from(self, source: Path) -> None:
        source_root = source / "knowledge" if (source / "knowledge").exists() else source
        object_source = source_root / "object_knowledge.py"
        procedural_source = source_root / "procedural_knowledge"
        if object_source.exists():
            shutil.copyfile(object_source, self.object_knowledge_path)
        if procedural_source.exists():
            for path in procedural_source.glob("*.py"):
                if path.name != "index.py":
                    shutil.copyfile(path, self.procedural_dir / path.name)

    def object_source(self) -> str:
        return self.object_knowledge_path.read_text(encoding="utf-8")

    def object_signatures(self) -> str:
        return self.object_signature_path.read_text(encoding="utf-8")

    def procedural_index(self) -> str:
        return self.procedural_index_path.read_text(encoding="utf-8")

    def object_names(self) -> set[str]:
        return top_level_definition_names(self.object_source())

    def procedural_names(self) -> set[str]:
        return {
            path.stem
            for path in self.procedural_dir.glob("*.py")
            if path.name != "index.py"
        }

    def knowledge_ids(self) -> list[str]:
        return [
            *(f"object.{name}" for name in sorted(self.object_names())),
            *(f"procedural.{name}" for name in sorted(self.procedural_names())),
        ]

    def has_knowledge(self) -> bool:
        return bool(self.object_source().strip() or self.procedural_names())

    def model_context(self) -> str:
        chunks = []
        if self.object_source().strip():
            object_ids = ", ".join(
                f"object.{name}" for name in sorted(self.object_names())
            )
            chunks.append(
                "# object_knowledge.py\n"
                + "# Knowledge ids: "
                + object_ids
                + "\n"
                + self.object_source().strip()
            )
        for path in sorted(self.procedural_dir.glob("*.py")):
            if path.name != "index.py":
                chunks.append(
                    f"# procedural_knowledge/{path.name}\n"
                    f"# Knowledge id: procedural.{path.stem}\n"
                    + path.read_text(encoding="utf-8").strip()
                )
        return "\n\n".join(chunks)

    def inspect(self, references: Sequence[str]) -> str:
        object_names = []
        procedural_names = []
        for raw in references:
            value = str(raw).strip().strip("`")
            if value.startswith("object."):
                object_names.append(value.split(".", 2)[1])
            elif value.startswith("procedural."):
                procedural_names.append(value.split(".", 2)[1])
            else:
                object_names.append(value.split(".", 1)[0])
                procedural_names.append(value.split(".", 1)[0])
        chunks = []
        object_source = extract_definitions(self.object_source(), object_names)
        if object_source.strip():
            chunks.append("# object_knowledge.py\n" + object_source)
        for name in dict.fromkeys(procedural_names):
            path = self.procedural_dir / f"{_safe_stem(name)}.py"
            if path.exists() and path.name != "index.py":
                chunks.append(
                    f"# procedural_knowledge/{path.name}\n"
                    + path.read_text(encoding="utf-8")
                )
        return "\n\n".join(chunks)

    def refresh_indexes(self) -> None:
        source = dedupe_top_level_imports(
            remove_top_level_self_imports(self.object_source(), "object_knowledge")
        )
        self.object_knowledge_path.write_text(source, encoding="utf-8")
        self.object_signature_path.write_text(
            build_signature_index(source), encoding="utf-8"
        )
        entries = {}
        for path in sorted(self.procedural_dir.glob("*.py")):
            if path.name == "index.py":
                continue
            entries[path.stem] = {
                "file": path.name,
                "signatures": build_signature_index(
                    path.read_text(encoding="utf-8"),
                    namespace=f"procedural.{path.stem}",
                ).strip(),
            }
        self.procedural_index_path.write_text(
            "PROCEDURAL_KNOWLEDGE = "
            + json.dumps(entries, ensure_ascii=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

    def preview_object_updates(self, updates: Sequence[Mapping[str, str]]) -> str:
        source = self.object_source()
        for item in updates:
            existing_name = _safe_identifier(str(item.get("existing_name") or ""))
            name = _safe_identifier(str(item.get("name") or ""))
            code = remove_top_level_self_imports(
                str(item.get("code") or ""), "object_knowledge"
            )
            if not name or not code.strip():
                continue
            compile(code, f"object_knowledge/{name}.py", "exec")
            if name not in top_level_definition_names(code):
                raise ValueError(f"object update does not define {name}")
            if existing_name and existing_name != name:
                source = remove_top_level_definitions(source, [existing_name])
            source = merge_top_level_definitions(source, code)
        compile(source or "\n", "object_knowledge.py", "exec")
        return dedupe_top_level_imports(source)

    def changed_update_targets(
        self,
        *,
        object_updates: Sequence[Mapping[str, str]],
        procedural_updates: Sequence[Mapping[str, str]],
    ) -> dict[str, list[str]]:
        candidate = self.preview_object_updates(object_updates)
        changed_objects = []
        for item in object_updates:
            old_name = _safe_identifier(
                str(item.get("existing_name") or item.get("name") or "")
            )
            new_name = _safe_identifier(str(item.get("name") or ""))
            before = extract_definitions(self.object_source(), [old_name])
            after = extract_definitions(candidate, [new_name])
            if old_name != new_name or _fingerprint(before) != _fingerprint(after):
                changed_objects.append(new_name)
        changed_procedural = []
        for item in procedural_updates:
            old_name = _safe_stem(
                str(item.get("existing_name") or item.get("name") or "")
            )
            new_name = _safe_stem(str(item.get("name") or ""))
            path = self.procedural_dir / f"{old_name}.py"
            before = path.read_text(encoding="utf-8") if path.exists() else ""
            after = str(item.get("code") or "")
            if old_name != new_name or _fingerprint(before) != _fingerprint(after):
                changed_procedural.append(new_name)
        return {"object": changed_objects, "procedural": changed_procedural}

    def commit_knowledge_updates(
        self,
        *,
        object_updates: Sequence[Mapping[str, str]],
        procedural_updates: Sequence[Mapping[str, str]],
    ) -> dict[str, list[str]]:
        candidate_object = self.preview_object_updates(object_updates)
        self._check_candidate(candidate_object, procedural_updates)
        changed = self.changed_update_targets(
            object_updates=object_updates,
            procedural_updates=procedural_updates,
        )
        if changed["object"]:
            self.object_knowledge_path.write_text(candidate_object, encoding="utf-8")
        for item in procedural_updates:
            existing_name = _safe_stem(str(item.get("existing_name") or ""))
            name = _safe_stem(str(item.get("name") or ""))
            code = str(item.get("code") or "").strip() + "\n"
            if name not in changed["procedural"]:
                continue
            if existing_name and existing_name != name:
                old_path = self.procedural_dir / f"{existing_name}.py"
                if old_path.exists() and old_path.name != "index.py":
                    old_path.unlink()
            (self.procedural_dir / f"{name}.py").write_text(code, encoding="utf-8")
        self.refresh_indexes()
        return changed

    def updates_from_patch(
        self,
        raw: str,
        plan: Mapping[str, Any],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        patch_text = _extract_patch(raw)
        object_plan = list(plan.get("object_knowledge_updates") or [])
        procedural_plan = list(plan.get("procedural_knowledge_updates") or [])
        allowed_paths = set()
        if object_plan:
            allowed_paths.add("object_knowledge.py")
        allowed_paths.update(
            f"procedural_knowledge/{_safe_stem(str(item['target']))}.py"
            for item in procedural_plan
        )
        touched_paths = set(identify_files_needed(patch_text)) | set(
            identify_files_added(patch_text)
        )
        if touched_paths != allowed_paths:
            raise ValueError(
                "knowledge patch must touch exactly the planned files: "
                + ", ".join(sorted(allowed_paths))
            )

        sources: dict[str, str] = {}
        if object_plan:
            sources["object_knowledge.py"] = self.object_source()
        for item in procedural_plan:
            target = _safe_stem(str(item["target"]))
            path = self.procedural_dir / f"{target}.py"
            sources[f"procedural_knowledge/{target}.py"] = (
                path.read_text(encoding="utf-8") if path.exists() else ""
            )

        def open_source(path: str) -> str:
            if path not in allowed_paths:
                raise ValueError(f"knowledge patch cannot read {path}")
            return sources[path]

        def write_source(path: str, content: str) -> None:
            if path not in allowed_paths:
                raise ValueError(f"knowledge patch cannot write {path}")
            sources[path] = content

        def remove_source(path: str) -> None:
            raise ValueError(f"knowledge patch cannot delete {path}")

        process_patch(patch_text, open_source, write_source, remove_source)

        object_updates = []
        for item in object_plan:
            target = _safe_identifier(str(item["target"]))
            code = extract_definitions(sources["object_knowledge.py"], [target])
            if not code.strip():
                raise ValueError(f"knowledge patch does not define object target {target}")
            object_updates.append(
                {
                    "existing_name": "" if item.get("operation") == "create" else target,
                    "name": target,
                    "code": code,
                }
            )

        procedural_updates = []
        for item in procedural_plan:
            target = _safe_stem(str(item["target"]))
            code = sources[f"procedural_knowledge/{target}.py"]
            if not code.strip():
                raise ValueError(f"knowledge patch leaves procedural target {target} empty")
            procedural_updates.append(
                {
                    "existing_name": "" if item.get("operation") == "create" else target,
                    "name": target,
                    "code": code,
                }
            )
        return object_updates, procedural_updates

    def _check_candidate(
        self,
        object_source: str,
        procedural_updates: Sequence[Mapping[str, str]],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc_dir = root / "procedural_knowledge"
            proc_dir.mkdir()
            (root / "object_knowledge.py").write_text(
                object_source.rstrip() + "\n", encoding="utf-8"
            )
            for path in self.procedural_dir.glob("*.py"):
                if path.name != "index.py":
                    shutil.copyfile(path, proc_dir / path.name)
            for item in procedural_updates:
                existing_name = _safe_stem(str(item.get("existing_name") or ""))
                name = _safe_stem(str(item.get("name") or ""))
                code = str(item.get("code") or "")
                if not name or not code.strip():
                    raise ValueError("procedural update requires a name and code")
                compile(code, f"procedural_knowledge/{name}.py", "exec")
                if existing_name and existing_name != name:
                    old_path = proc_dir / f"{existing_name}.py"
                    if old_path.exists():
                        old_path.unlink()
                (proc_dir / f"{name}.py").write_text(
                    code.strip() + "\n", encoding="utf-8"
                )

            old_path = list(sys.path)
            saved_object = sys.modules.get("object_knowledge")
            sys.path.insert(0, str(root))
            try:
                sys.modules.pop("object_knowledge", None)
                _load_module(root / "object_knowledge.py", "object_knowledge")
                for path in proc_dir.glob("*.py"):
                    _load_module(path, f"_ocm_check_{path.stem}")
                    sys.modules.pop(f"_ocm_check_{path.stem}", None)
            finally:
                sys.path[:] = old_path
                sys.modules.pop("object_knowledge", None)
                if saved_object is not None:
                    sys.modules["object_knowledge"] = saved_object


def parse_reflection_plan(raw: str) -> dict[str, list[dict[str, str]]]:
    payload = load_jsonish(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("reflection plan must be a JSON object")
    return {
        "object_knowledge_updates": _parse_plan_updates(
            payload.get("object_knowledge_updates"), "object"
        ),
        "procedural_knowledge_updates": _parse_plan_updates(
            payload.get("procedural_knowledge_updates"), "procedural"
        ),
    }


def validate_reflection_plan(
    plan: Mapping[str, Any],
    *,
    object_names: set[str],
    procedural_names: set[str],
) -> None:
    errors = []
    for key, existing in (
        ("object_knowledge_updates", object_names),
        ("procedural_knowledge_updates", procedural_names),
    ):
        items = list(plan.get(key) or [])
        if len(items) > 1:
            errors.append(f"{key} may contain at most one item")
        identities = {_identity(name): name for name in existing}
        for item in items:
            target = str(item.get("target") or "")
            operation = str(item.get("operation") or "")
            lesson = str(item.get("lesson") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            exists = _identity(target) in identities
            if not target or not lesson or not evidence:
                errors.append(f"{key} requires target, lesson, and evidence")
            if key == "object_knowledge_updates" and not re.fullmatch(
                r"[A-Z][A-Za-z0-9_]*", target
            ):
                errors.append(
                    f"object target {target!r} must be a PascalCase semantic category or mechanism"
                )
            if operation not in {"create", "update", "replace"}:
                errors.append(f"{target} has invalid operation {operation!r}")
            elif operation == "create" and exists:
                errors.append(f"{target} already exists")
            elif operation in {"update", "replace"} and not exists:
                errors.append(f"{target} does not exist")
            if exists:
                item["target"] = identities[_identity(target)]
    if errors:
        raise ValueError("invalid reflection plan: " + "; ".join(errors))


def parse_knowledge_updates(
    raw: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = load_jsonish(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("knowledge update must be a JSON object")
    return (
        _parse_code_updates(payload.get("object_knowledge")),
        _parse_code_updates(payload.get("procedural_knowledge")),
    )


def validate_knowledge_updates(
    plan: Mapping[str, Any],
    *,
    object_updates: Sequence[Mapping[str, str]],
    procedural_updates: Sequence[Mapping[str, str]],
) -> None:
    errors = []
    for key, updates in (
        ("object_knowledge_updates", object_updates),
        ("procedural_knowledge_updates", procedural_updates),
    ):
        planned = list(plan.get(key) or [])
        if len(planned) != len(updates):
            errors.append(f"{key} implementation count does not match the plan")
            continue
        for item, update in zip(planned, updates):
            target = _identity(str(item.get("target") or ""))
            name = _identity(str(update.get("name") or ""))
            existing = _identity(str(update.get("existing_name") or ""))
            operation = str(item.get("operation") or "")
            if target != name:
                errors.append(f"planned target {target} was implemented as {name}")
            if operation == "create" and existing:
                errors.append(f"new target {target} must have empty existing_name")
            if operation in {"update", "replace"} and existing != target:
                errors.append(f"existing target {target} must name itself in existing_name")
            if not str(update.get("code") or "").strip():
                errors.append(f"target {target} has empty code")
    if errors:
        raise ValueError("knowledge update does not implement the plan: " + "; ".join(errors))


def _parse_plan_updates(value: Any, kind: str) -> list[dict[str, str]]:
    result = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, Mapping):
            continue
        target = str(item.get("target") or "").strip().strip("`")
        target = re.sub(r"^(?:object|procedural)\.", "", target)
        target = target.split(".", 1)[0]
        target = _safe_identifier(target) if kind == "object" else _safe_stem(target)
        result.append(
            {
                "target": target,
                "operation": str(item.get("operation") or "").strip().lower(),
                "lesson": str(item.get("lesson") or "").strip(),
                "evidence": str(item.get("evidence") or "").strip(),
            }
        )
    return result


def _parse_code_updates(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {
            "existing_name": str(item.get("existing_name") or ""),
            "name": str(item.get("name") or ""),
            "code": str(item.get("code") or ""),
        }
        for item in value
        if isinstance(item, Mapping)
    ]


def _safe_stem(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", name)).strip("_").lower()


def _safe_identifier(name: str) -> str:
    match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*", name.strip())
    return match.group(0) if match else ""


def _identity(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name).lower()


def _extract_patch(raw: str) -> str:
    text = str(raw or "")
    start = text.find("*** Begin Patch")
    end = text.rfind("*** End Patch")
    if start < 0 or end < start:
        raise ValueError("knowledge update must contain an apply_patch block")
    return text[start : end + len("*** End Patch")]


def _fingerprint(source: str) -> str:
    try:
        return ast.dump(ast.parse(source or "\n"), include_attributes=False)
    except SyntaxError:
        return source.strip()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
