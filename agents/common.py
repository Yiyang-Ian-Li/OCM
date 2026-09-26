from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def load_benchmark_instruction(benchmark: str | None) -> str:
    if not benchmark:
        return ""
    name = str(benchmark).strip().lower()
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / name / "instruction.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=16)
def load_benchmark_examples(benchmark: str | None) -> str:
    if str(benchmark or "").strip().lower() != "sciworld":
        return ""
    root = Path(__file__).resolve().parents[1]
    path = root / "baselines" / "WorldEvolver" / "src" / "prompts" / "ReactAgent" / "scienceworld_react.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples") or ""
    if isinstance(examples, list):
        return "\n".join(str(item) for item in examples)
    return str(examples).strip()

def extract_action_candidate(raw_output: str) -> str:
    text = (raw_output or "").strip().replace("```", "")
    if not text:
        return ""

    # Prefer explicit "Action: ..." if provided by prompt format. Use the last
    # match so a single-line "Thought: ... Action: ..." response still parses.
    action_matches = list(re.finditer(r"^\s*Action\s*:?\s+(.+)$", text, flags=re.IGNORECASE | re.MULTILINE))
    if not action_matches:
        action_matches = list(re.finditer(r"\bAction\s*:\s*(.+)", text, flags=re.IGNORECASE))
    if action_matches:
        return clean_action_text(action_matches[-1].group(1))

    # Fallback for models that produce:
    #   Thought: ...
    #   <action>
    # without the explicit Action: label.
    candidate_lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(candidate_lines):
        if re.match(r"^(Thought|Reasoning)\s*:", line, flags=re.IGNORECASE):
            continue
        return clean_action_text(line)
    if candidate_lines:
        return clean_action_text(candidate_lines[-1])
    return ""


def clean_action_text(action: str) -> str:
    text = action.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text
