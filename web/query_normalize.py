"""Lightweight user-query normalization for typos and oral Chinese."""

from __future__ import annotations

import re


# Order matters: longer / more specific patterns first.
PHRASE_FIXES: tuple[tuple[str, str], ...] = (
    (r"迟饭", "吃饭"),
    (r"拖芡", "拖欠"),
    (r"仲栽", "仲裁"),
    (r"劳东法", "劳动法"),
    (r"劳东", "劳动"),
    (r"合通法", "合同法"),
    (r"合通", "合同"),
    (r"辞聘", "辞退"),
    (r"补尝", "补偿"),
    (r"陪偿", "赔偿"),
    (r"公资", "工资"),
    (r"薪姿", "薪资"),
    (r"加斑", "加班"),
    (r"解聘", "辞退"),
    (r"工伤险", "工伤保险"),
    (r"调解仲栽", "调解仲裁"),
    (r"没签和同", "没签合同"),
    (r"没签合通", "没签合同"),
)


def normalize_user_query(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return text
    for pattern, replacement in PHRASE_FIXES:
        text = re.sub(pattern, replacement, text)
    return text


def query_changed(original: str, normalized: str) -> bool:
    return (original or "").strip() != (normalized or "").strip()
