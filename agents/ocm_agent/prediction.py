from __future__ import annotations

import re


def parse_prediction_provenance(raw_output: str) -> dict[str, object]:
    prediction = _labeled_line(raw_output, "Prediction")
    grounding_raw = _labeled_line(raw_output, "Grounding")
    object_symbols: list[str] = []
    procedural_symbols: list[str] = []
    unresolved_symbols: list[str] = []

    for raw_symbol in re.split(r"[,;]", grounding_raw):
        symbol = raw_symbol.strip().strip("`[]()")
        if not symbol or symbol.lower() in {"none", "n/a", "na"}:
            continue
        normalized = _normalize_symbol(symbol)
        if normalized.startswith("object."):
            object_symbols.append(normalized)
        elif normalized.startswith("procedural."):
            procedural_symbols.append(normalized)
        else:
            unresolved_symbols.append(symbol)

    return {
        "prediction": prediction,
        "grounding_raw": grounding_raw,
        "object_symbols": _dedupe(object_symbols),
        "procedural_symbols": _dedupe(procedural_symbols),
        "unresolved_symbols": _dedupe(unresolved_symbols),
    }


def _labeled_line(text: str, label: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
        text or "",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _normalize_symbol(symbol: str) -> str:
    lowered = symbol.lower()
    if lowered.startswith("object_knowledge."):
        return "object." + symbol.split(".", 1)[1]
    if lowered.startswith("procedural_knowledge."):
        return "procedural." + symbol.split(".", 1)[1]
    return symbol


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
