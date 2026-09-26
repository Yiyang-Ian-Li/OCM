from __future__ import annotations

import json
import re
from typing import Any


def load_jsonish(raw: str) -> Any:
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        embedded = _embedded_json(text)
        if embedded is not None:
            return embedded
        suffix = _missing_delimiters(text)
        if not suffix:
            raise original_error
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            raise original_error


def _embedded_json(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    starts = [index for index, character in enumerate(text) if character in "[{"]
    if not starts:
        return None
    candidate = text[starts[0] :]
    try:
        value, _ = decoder.raw_decode(candidate)
        return value
    except json.JSONDecodeError:
        suffix = _missing_delimiters(candidate)
        if not suffix:
            return None
        try:
            return json.loads(candidate + suffix)
        except json.JSONDecodeError:
            return None


def _missing_delimiters(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack or stack.pop() != pairs[character]:
                return ""
    if in_string:
        return ""
    closing = {"{": "}", "[": "]"}
    return "".join(closing[value] for value in reversed(stack))
