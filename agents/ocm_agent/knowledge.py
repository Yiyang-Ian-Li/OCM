from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from .knowledge_code import (
    build_signature_index,
    dedupe_top_level_imports,
    extract_definitions,
    merge_top_level_definitions,
    remove_top_level_self_imports,
    remove_top_level_definitions,
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
        if load_path:
            store._load_from(Path(load_path).expanduser())
        if not store.object_knowledge_path.exists():
            store.object_knowledge_path.write_text("", encoding="utf-8")
        if not store.procedural_index_path.exists():
            store.procedural_index_path.write_text("PROCEDURAL_KNOWLEDGE = {}\n", encoding="utf-8")
        store.refresh_indexes()
        return store

    def _load_from(self, source: Path) -> None:
        source_root = source
        if (source / "knowledge").exists():
            source_root = source / "knowledge"
        object_source = source_root / "object_knowledge.py"
        procedural_source = source_root / "procedural_knowledge"
        if object_source.exists():
            self.object_knowledge_path.write_text(object_source.read_text(encoding="utf-8"), encoding="utf-8")
        if procedural_source.exists():
            self.procedural_dir.mkdir(parents=True, exist_ok=True)
            for path in procedural_source.glob("*.py"):
                if path.name == "index.py":
                    continue
                (self.procedural_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    def object_source(self) -> str:
        return self.object_knowledge_path.read_text(encoding="utf-8")

    def object_signatures(self) -> str:
        return self.object_signature_path.read_text(encoding="utf-8")

    def procedural_index(self) -> str:
        return self.procedural_index_path.read_text(encoding="utf-8")

    def has_knowledge(self) -> bool:
        if self.object_source().strip():
            return True
        return any(path.name != "index.py" for path in self.procedural_dir.glob("*.py"))

    def refresh_indexes(self) -> None:
        object_source = dedupe_top_level_imports(remove_top_level_self_imports(self.object_source(), "object_knowledge"))
        self.object_knowledge_path.write_text(object_source, encoding="utf-8")
        self.object_signature_path.write_text(build_signature_index(object_source), encoding="utf-8")
        self.procedural_index_path.write_text(self._build_procedural_index(), encoding="utf-8")

    def inspect(self, names: list[str]) -> str:
        object_names: list[str] = []
        procedural_names: list[str] = []
        for raw_name in names:
            name = raw_name.strip().strip("`")
            if not name:
                continue
            if name.startswith(("object.", "object_knowledge.")):
                object_names.append(name.split(".", 1)[1].split(".", 1)[0])
            elif name.startswith(("procedural.", "procedural_knowledge.")):
                procedural_names.append(name.split(".", 1)[1].split(".", 1)[0])
            else:
                object_names.append(name.split(".", 1)[0])
                procedural_names.append(name.split(".", 1)[0])
        chunks = []
        object_source = extract_definitions(self.object_source(), _dedupe(object_names))
        if object_source.strip():
            chunks.append("# object_knowledge.py\n" + object_source)
        for procedural_name in _dedupe(procedural_names):
            path = self.procedural_dir / f"{_safe_stem(procedural_name)}.py"
            if path.exists() and path.name != "index.py":
                chunks.append(f"# procedural_knowledge/{path.name}\n" + path.read_text(encoding="utf-8"))
        return "\n\n".join(chunks)

    def preview_object_updates(self, updates: list[dict[str, str]]) -> str:
        source = self.object_source()
        for item in updates:
            existing_name = _safe_identifier(str(item.get("existing_name") or ""))
            name = _safe_identifier(str(item.get("name") or ""))
            code = str(item.get("code") or "")
            if not name or not code.strip():
                continue
            code = remove_top_level_self_imports(code, "object_knowledge")
            compile(code, f"object_knowledge/{name}.py", "exec")
            if existing_name and existing_name != name:
                source = remove_top_level_definitions(source, [existing_name])
            source = merge_top_level_definitions(source, code)
        source = dedupe_top_level_imports(remove_top_level_self_imports(source, "object_knowledge"))
        compile(source, "object_knowledge.py", "exec")
        return source

    def write_object_source(self, source: str) -> None:
        source = remove_top_level_self_imports(source, "object_knowledge")
        compile(source, "object_knowledge.py", "exec")
        self.object_knowledge_path.write_text(dedupe_top_level_imports(source.rstrip() + "\n"), encoding="utf-8")
        self.refresh_indexes()

    def commit_knowledge_updates(
        self,
        *,
        object_updates: list[dict[str, str]],
        procedural_updates: list[dict[str, str]],
    ) -> dict[str, list[str]]:
        object_source = self.preview_object_updates(object_updates)
        errors = self.check_candidate_knowledge(procedural_updates, object_source=object_source)
        if errors:
            raise ValueError("knowledge check failed:\n" + "\n\n".join(errors))
        self.write_object_source(object_source)
        written_procedural = self.write_procedural_knowledge(procedural_updates)
        return {
            "object": [_safe_identifier(str(item.get("name") or "")) for item in object_updates if str(item.get("name") or "").strip()],
            "procedural": written_procedural,
        }

    def write_procedural_knowledge(self, updates: list[dict[str, str]]) -> list[str]:
        errors = self.check_candidate_knowledge(updates)
        if errors:
            raise ValueError("procedural knowledge check failed:\n" + "\n\n".join(errors))
        written = []
        for item in updates:
            existing_name = _safe_stem(str(item.get("existing_name") or ""))
            name = _safe_stem(str(item.get("name") or ""))
            code = str(item.get("code") or "")
            if not name or not code.strip():
                continue
            compile(code, f"procedural_knowledge/{name}.py", "exec")
            if existing_name and existing_name != name:
                existing_path = self.procedural_dir / f"{existing_name}.py"
                if existing_path.exists() and existing_path.name != "index.py":
                    existing_path.unlink()
            path = self.procedural_dir / f"{name}.py"
            path.write_text(code.strip() + "\n", encoding="utf-8")
            written.append(name)
        self.refresh_indexes()
        return written

    def check_candidate_knowledge(
        self,
        updates: list[dict[str, str]],
        *,
        object_source: str | None = None,
    ) -> list[str]:
        errors: list[str] = []
        normalized_updates = []
        for item in updates:
            existing_name = _safe_stem(str(item.get("existing_name") or ""))
            name = _safe_stem(str(item.get("name") or ""))
            code = str(item.get("code") or "")
            if not name or not code.strip():
                continue
            try:
                compile(code, f"procedural_knowledge/{name}.py", "exec")
            except Exception:
                errors.append(f"{name}: compile failed\n{traceback.format_exc(limit=8)}")
                continue
            normalized_updates.append({"existing_name": existing_name, "name": name, "code": code.strip() + "\n"})
        if errors:
            return errors
        errors.extend(self._import_candidate_knowledge(normalized_updates, object_source=object_source))
        return errors

    def _import_candidate_knowledge(
        self,
        procedural_updates: list[dict[str, str]],
        *,
        object_source: str | None = None,
    ) -> list[str]:
        errors: list[str] = []
        candidate_names = {str(item.get("name") or "") for item in procedural_updates if str(item.get("name") or "")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            procedural_dir = root / "procedural_knowledge"
            procedural_dir.mkdir(parents=True, exist_ok=True)
            (root / "object_knowledge.py").write_text(
                (object_source if object_source is not None else self.object_source()).rstrip() + "\n",
                encoding="utf-8",
            )
            for existing in self.procedural_dir.glob("*.py"):
                if existing.name != "index.py":
                    (procedural_dir / existing.name).write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
            for item in procedural_updates:
                existing_name = str(item.get("existing_name") or "")
                if existing_name and existing_name != item["name"]:
                    existing_path = procedural_dir / f"{existing_name}.py"
                    if existing_path.exists() and existing_path.name != "index.py":
                        existing_path.unlink()
                (procedural_dir / f"{item['name']}.py").write_text(item["code"], encoding="utf-8")

            old_path = list(sys.path)
            saved_object = sys.modules.get("object_knowledge")
            sys.path.insert(0, str(root))
            try:
                sys.modules.pop("object_knowledge", None)
                self._load_module(root / "object_knowledge.py", "_ocm_import_object_knowledge")
                for path in sorted(procedural_dir.glob("*.py")):
                    if path.name == "index.py":
                        continue
                    module_name = f"_ocm_import_{path.stem}"
                    try:
                        module = self._load_module(path, module_name)
                        if path.stem in candidate_names:
                            verifier = getattr(module, "__verify__", None)
                            if not callable(verifier):
                                errors.append(
                                    f"{path.stem}: missing callable __verify__() for procedural self-consistency check"
                                )
                            else:
                                try:
                                    verifier()
                                except Exception:
                                    errors.append(
                                        f"{path.stem}: __verify__ failed\n{traceback.format_exc(limit=12)}"
                                    )
                    except Exception:
                        errors.append(f"{path.stem}: import failed\n{traceback.format_exc(limit=12)}")
                    finally:
                        sys.modules.pop(module_name, None)
            except Exception:
                errors.append("object_knowledge: import failed\n" + traceback.format_exc(limit=12))
            finally:
                sys.path[:] = old_path
                sys.modules.pop("object_knowledge", None)
                sys.modules.pop("_ocm_import_object_knowledge", None)
                if saved_object is not None:
                    sys.modules["object_knowledge"] = saved_object
        return errors

    def _load_module(self, path: Path, module_name: str):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _build_procedural_index(self) -> str:
        entries = {}
        for path in sorted(self.procedural_dir.glob("*.py")):
            if path.name == "index.py":
                continue
            source = path.read_text(encoding="utf-8")
            entries[path.stem] = {
                "file": path.name,
                "signatures": build_signature_index(source).strip(),
            }
        return "PROCEDURAL_KNOWLEDGE = " + json.dumps(entries, ensure_ascii=True, indent=2) + "\n"


def parse_reflection_plan(raw: str) -> dict[str, Any]:
    payload = _loads_jsonish(raw)
    if not isinstance(payload, dict):
        raise ValueError("reflection plan must be a JSON object")
    return {
        "object_knowledge_updates": _as_list(payload.get("object_knowledge_updates") or payload.get("Object Knowledge Updates")),
        "procedural_knowledge_updates": _as_list(
            payload.get("procedural_knowledge_updates") or payload.get("Procedural Knowledge Updates")
        ),
        "inspect_objects": [str(item) for item in _as_list(payload.get("inspect_objects") or payload.get("Inspect Objects"))],
        "inspect_procedural": [
            str(item) for item in _as_list(payload.get("inspect_procedural") or payload.get("Inspect Procedural"))
        ],
    }


def parse_knowledge_updates(raw: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = _loads_jsonish(raw)
    if not isinstance(payload, dict):
        raise ValueError("knowledge update must be a JSON object")
    return (
        _parse_named_code_updates(payload.get("object_knowledge") or payload.get("object_knowledge_updates")),
        _parse_named_code_updates(payload.get("procedural_knowledge") or payload.get("procedural_knowledge_updates")),
    )


def _parse_named_code_updates(payload: Any) -> list[dict[str, str]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("knowledge updates must be a JSON list")
    updates = []
    for item in payload:
        if isinstance(item, dict):
            updates.append(
                {
                    "existing_name": str(item.get("existing_name") or ""),
                    "name": str(item.get("name") or ""),
                    "code": str(item.get("code") or ""),
                }
            )
    return updates


def _loads_jsonish(raw: str) -> Any:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip())
    stem = re.sub(r"_+", "_", stem).strip("_").lower()
    return stem


def _safe_identifier(name: str) -> str:
    value = name.strip().strip("`")
    match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*", value)
    return match.group(0) if match else ""


def _dedupe(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
