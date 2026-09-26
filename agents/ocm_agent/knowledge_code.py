from __future__ import annotations

import ast


def build_signature_index(source: str, *, namespace: str = "object") -> str:
    try:
        module = ast.parse(source or "\n")
    except SyntaxError as exc:
        return f"# Available {namespace} knowledge\n\nCould not parse knowledge: {exc}\n"
    lines = [f"# Available {namespace} knowledge", ""]
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            lines.append(f"- `{namespace}.{node.name}`: class {node.name}{_base_list(node)}")
            _append_docstring(lines, node, indent="  ")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
                    lines.append(
                        f"  - `{namespace}.{node.name}.{item.name}`: def {item.name}{_signature(item)}"
                    )
                    _append_docstring(lines, item, indent="    ")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            lines.append(f"- `{namespace}.{node.name}`: def {node.name}{_signature(node)}")
            _append_docstring(lines, node, indent="  ")
    return "\n".join(lines).strip() + "\n"


def extract_definitions(source: str, names: list[str]) -> str:
    wanted = {name.strip().split(".", 1)[0] for name in names if name.strip()}
    if not wanted:
        return ""
    try:
        module = ast.parse(source or "\n")
    except SyntaxError:
        return ""
    lines = source.splitlines()
    chunks = []
    for node in module.body:
        name = _top_level_name(node)
        if name not in wanted:
            continue
        snippet = _node_source(lines, node)
        if snippet:
            chunks.append(snippet)
    return "\n\n".join(chunks)


def top_level_definition_names(source: str) -> set[str]:
    try:
        module = ast.parse(source or "\n")
    except SyntaxError:
        return set()
    return {name for node in module.body if (name := _top_level_name(node))}


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
    ordered_names: list[str] = []
    for node in update_module.body:
        snippet = _node_source(update_lines, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if snippet and snippet not in new_imports:
                new_imports.append(snippet)
            continue
        name = _top_level_name(node)
        if name and snippet:
            replacements[name] = snippet
            if name not in ordered_names:
                ordered_names.append(name)

    consumed: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    existing_imports: set[str] = set()
    for node in current_module.body:
        snippet = _node_source(current_lines, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and snippet:
            existing_imports.add(snippet)
        name = _top_level_name(node)
        if name in replacements:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is not None and end is not None:
                ranges.append((start - 1, end, replacements[name]))
                consumed.add(name)

    merged_lines = list(current_lines)
    for start, end, replacement in sorted(ranges, reverse=True):
        merged_lines[start:end] = replacement.splitlines()
    merged = "\n".join(merged_lines).rstrip()
    imports = [line for line in new_imports if line not in existing_imports]
    if imports:
        merged = _insert_imports(merged, imports)
    additions = [replacements[name] for name in ordered_names if name not in consumed]
    if additions:
        merged = (merged.rstrip() + "\n\n\n" + "\n\n\n".join(additions)).strip()
    return dedupe_top_level_imports(merged.rstrip() + "\n")


def remove_top_level_definitions(source: str, names: list[str]) -> str:
    wanted = {name for name in names if name}
    if not wanted:
        return source
    module = ast.parse(source or "\n")
    lines = source.splitlines()
    ranges = []
    for node in module.body:
        if _top_level_name(node) not in wanted:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None and end is not None:
            ranges.append((start - 1, end))
    for start, end in sorted(ranges, reverse=True):
        lines[start:end] = []
    return "\n".join(lines).strip() + ("\n" if lines else "")


def remove_top_level_self_imports(source: str, module_name: str) -> str:
    if not source.strip():
        return source
    module = ast.parse(source)
    lines = source.splitlines()
    ranges = []
    for node in module.body:
        remove = (
            isinstance(node, ast.ImportFrom)
            and node.module == module_name
        ) or (
            isinstance(node, ast.Import)
            and any(alias.name == module_name for alias in node.names)
        )
        if remove:
            ranges.append((node.lineno - 1, node.end_lineno or node.lineno))
    for start, end in sorted(ranges, reverse=True):
        lines[start:end] = []
    return "\n".join(lines).strip() + ("\n" if lines else "")


def dedupe_top_level_imports(source: str) -> str:
    if not source.strip():
        return source
    module = ast.parse(source)
    lines = source.splitlines()
    seen: set[str] = set()
    ranges = []
    for node in module.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        snippet = _node_source(lines, node)
        if snippet in seen:
            ranges.append((node.lineno - 1, node.end_lineno or node.lineno))
        else:
            seen.add(snippet)
    for start, end in sorted(ranges, reverse=True):
        lines[start:end] = []
    return "\n".join(lines).rstrip() + "\n"


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
    if lines and lines[0].startswith("from __future__ import"):
        insert_at = 1
    lines[insert_at:insert_at] = imports + [""]
    return "\n".join(lines).rstrip() + "\n"


def _base_list(node: ast.ClassDef) -> str:
    bases = [ast.unparse(base) for base in node.bases]
    return f"({', '.join(bases)}):" if bases else ":"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    result = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        result += " -> " + ast.unparse(node.returns)
    return result


def _append_docstring(lines: list[str], node: ast.AST, *, indent: str) -> None:
    doc = ast.get_docstring(node)
    if doc:
        lines.append(indent + "description: " + " ".join(doc.strip().split()))
