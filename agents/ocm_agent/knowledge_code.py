from __future__ import annotations

import ast
import re

from agents.common import clean_action_text


def build_signature_index(source: str) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        return f"# object_knowledge.py signatures\n\nCould not parse object_knowledge.py: {exc}\n"
    lines = ["# object_knowledge.py signatures", ""]
    module_doc = _docstring_lines(module)
    if module_doc:
        lines.append("- module docstring:")
        lines.extend(f"  {line}" for line in module_doc)
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"- class {node.name}{_base_list(node)}")
            doc = _docstring_lines(node)
            if doc:
                lines.append("  - docstring:")
                lines.extend(f"    {line}" for line in doc)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    lines.append(f"  - def {item.name}{_signature(item)}")
                    method_doc = _docstring_lines(item)
                    if method_doc:
                        lines.append("    - docstring:")
                        lines.extend(f"      {line}" for line in method_doc)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            lines.append(f"- def {node.name}{_signature(node)}")
            doc = _docstring_lines(node)
            if doc:
                lines.append("  - docstring:")
                lines.extend(f"    {line}" for line in doc)
    return "\n".join(lines).strip() + "\n"


def extract_definitions(source: str, names: list[str], *, max_chars: int = 16000) -> str:
    wanted = {name.strip() for name in names if name and name.strip()}
    if not wanted:
        return ""
    try:
        module = ast.parse(source)
    except SyntaxError:
        return ""
    chunks: list[str] = []
    lines = source.splitlines()
    for node in module.body:
        node_name = getattr(node, "name", None)
        if not node_name or node_name not in wanted:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        chunks.append("\n".join(lines[start - 1 : end]))
    text = "\n\n".join(chunks).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def merge_top_level_definitions(current_source: str, update_source: str) -> str:
    current_source = current_source or ""
    update_source = update_source or ""
    if not update_source.strip():
        return current_source
    current_module = ast.parse(current_source or "\n")
    update_module = ast.parse(update_source)
    current_lines = current_source.splitlines()
    update_lines = update_source.splitlines()

    replacements: dict[str, str] = {}
    new_imports: list[str] = []
    ordered_new_names: list[str] = []
    for node in update_module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            snippet = _node_source(update_lines, node)
            if snippet and snippet not in new_imports:
                new_imports.append(snippet)
            continue
        name = _top_level_name(node)
        if not name:
            continue
        snippet = _node_source(update_lines, node)
        if not snippet:
            continue
        replacements[name] = snippet
        if name not in ordered_new_names:
            ordered_new_names.append(name)

    if not replacements and not new_imports:
        return current_source

    consumed: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    existing_imports: set[str] = set()
    for node in current_module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            snippet = _node_source(current_lines, node)
            if snippet:
                existing_imports.add(snippet)
        name = _top_level_name(node)
        if name and name in replacements:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None:
                ranges.append((start - 1, end, replacements[name]))
                consumed.add(name)

    merged_lines = list(current_lines)
    for start, end, replacement in sorted(ranges, reverse=True):
        merged_lines[start:end] = replacement.splitlines()
    merged = "\n".join(merged_lines).rstrip()

    imports_to_add = [line for line in new_imports if line not in existing_imports]
    if imports_to_add:
        merged = _insert_imports(merged, imports_to_add)

    additions = [replacements[name] for name in ordered_new_names if name not in consumed]
    if additions:
        merged = (merged.rstrip() + "\n\n\n" + "\n\n\n".join(additions)).strip()
    return dedupe_top_level_imports(merged.rstrip() + "\n")


def remove_top_level_definitions(source: str, names: list[str]) -> str:
    wanted = {name for name in names if name}
    if not wanted:
        return source
    module = ast.parse(source or "\n")
    lines = (source or "").splitlines()
    ranges = []
    for node in module.body:
        name = _top_level_name(node)
        if name not in wanted:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None:
            ranges.append((start - 1, end))
    merged_lines = list(lines)
    for start, end in sorted(ranges, reverse=True):
        merged_lines[start:end] = []
        while start < len(merged_lines) and not merged_lines[start].strip():
            merged_lines.pop(start)
    return "\n".join(merged_lines).rstrip() + ("\n" if merged_lines else "")


def dedupe_top_level_imports(source: str) -> str:
    if not source.strip():
        return source
    try:
        module = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines()
    seen = set()
    ranges = []
    for node in module.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        snippet = _node_source(lines, node)
        if not snippet:
            continue
        if snippet in seen:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None:
                ranges.append((start - 1, end))
        else:
            seen.add(snippet)
    if not ranges:
        return source.rstrip() + "\n"
    merged_lines = list(lines)
    for start, end in sorted(ranges, reverse=True):
        merged_lines[start:end] = []
        while start < len(merged_lines) and not merged_lines[start].strip():
            merged_lines.pop(start)
    return "\n".join(merged_lines).rstrip() + "\n"


def remove_top_level_self_imports(source: str, module_name: str) -> str:
    if not source.strip():
        return source
    try:
        module = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines()
    ranges = []
    for node in module.body:
        remove = False
        if isinstance(node, ast.ImportFrom):
            remove = node.module == module_name
        elif isinstance(node, ast.Import):
            remove = any(alias.name == module_name for alias in node.names)
        if not remove:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None:
            ranges.append((start - 1, end))
    if not ranges:
        return source.rstrip() + "\n"
    merged_lines = list(lines)
    for start, end in sorted(ranges, reverse=True):
        merged_lines[start:end] = []
        while start < len(merged_lines) and not merged_lines[start].strip():
            merged_lines.pop(start)
    return "\n".join(merged_lines).rstrip() + "\n"


def parse_inspect_request(raw_output: str) -> list[str]:
    text = raw_output or ""
    action_matches = list(re.finditer(r"^\s*Action\s*:?\s+(.+)$", text, flags=re.IGNORECASE | re.MULTILINE))
    if not action_matches:
        action_matches = list(re.finditer(r"\bAction\s*:\s*(.+)", text, flags=re.IGNORECASE))
    if not action_matches:
        return []
    action = clean_action_text(action_matches[-1].group(1))
    match = re.match(r"(?is)^Inspect\s*\[\s*(.*?)\s*\]\s*$", action)
    if not match:
        return []
    value = match.group(1).strip()
    if not value:
        return []
    return [item.strip().strip("`") for item in re.split(r"[,;\n]+", value) if item.strip()]


def _top_level_name(node: ast.AST) -> str:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Assign):
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        return names[0] if len(names) == 1 else ""
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return ""


def _node_source(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return ""
    return "\n".join(lines[start - 1 : end]).strip()


def _insert_imports(source: str, imports: list[str]) -> str:
    lines = source.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("#"):
        insert_at = 1
    if len(lines) > insert_at and lines[insert_at].strip() == "":
        insert_at += 1
    if len(lines) > insert_at and lines[insert_at].startswith("from __future__ import"):
        insert_at += 1
        if len(lines) > insert_at and lines[insert_at].strip() == "":
            insert_at += 1
    lines[insert_at:insert_at] = imports + [""]
    return "\n".join(lines).rstrip() + "\n"


def _base_list(node: ast.ClassDef) -> str:
    if not node.bases:
        return ":"
    bases = []
    for base in node.bases:
        try:
            bases.append(ast.unparse(base))
        except Exception:
            continue
    return f"({', '.join(bases)}):" if bases else ":"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    returns = ""
    if node.returns is not None:
        try:
            returns = " -> " + ast.unparse(node.returns)
        except Exception:
            returns = ""
    return f"({args}){returns}"


def _docstring_lines(node: ast.AST) -> list[str]:
    doc = ast.get_docstring(node) or ""
    lines = []
    for line in doc.strip().splitlines():
        clean = line.strip()
        if clean:
            lines.append(clean)
    return lines
