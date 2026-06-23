"""Shared conversation history helpers."""

from __future__ import annotations

# 0 = keep full chat history within one session
MAX_HISTORY_TURNS = 0


def trim_turns(turns: list[dict]) -> list[dict]:
    if MAX_HISTORY_TURNS <= 0:
        return list(turns or [])
    return list(turns or [])[-MAX_HISTORY_TURNS:]
