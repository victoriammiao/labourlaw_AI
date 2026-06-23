#!/usr/bin/env python3
"""v3 劳动法数据筛选与规范化 — 共享常量与工具函数。"""

from __future__ import annotations

import re
from typing import Iterable

# 核心劳动法主题词（命中任一即可视为劳动法相关）
CORE_KEYWORDS: tuple[str, ...] = (
    "劳动合同",
    "劳动法",
    "劳动争议",
    "工伤",
    "社保",
    "社会保险",
    "辞退",
    "解雇",
    "加班",
    "工资",
    "仲裁",
    "派遣",
    "竞业",
    "试用期",
    "经济补偿",
    "赔偿金",
    "带薪年假",
    "年休假",
    "产假",
    "哺乳期",
    "劳务派遣",
    "非全日制",
    "离职证明",
    "双倍工资",
    "调岗",
    "降薪",
    "经济性裁员",
)

# 劳动法条白名单（引用法条时优先保留）
LABOR_LAW_WHITELIST: tuple[str, ...] = (
    "劳动合同法",
    "劳动法",
    "劳动争议调解仲裁法",
    "工伤保险条例",
    "社会保险法",
    "职工带薪年休假条例",
    "劳动合同法实施条例",
    "就业促进法",
    "职业病防治法",
    "工会法",
    "安全生产法",
    "妇女权益保障法",
)

# 明显非劳动法领域，引用占主导时丢弃
EXCLUDED_LAW_BLACKLIST: tuple[str, ...] = (
    "海商法",
    "证券法",
    "合伙企业法",
    "民事诉讼法",
    "海事诉讼特别程序法",
    "企业破产法",
    "刑法",
    "民法典",
    "婚姻法",
    "公司法",
    "行政诉讼法",
)

LAW_NAME_RE = re.compile(r"《([^》]+)》")
QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<问题>[：:]\s*\n?(.+)", re.DOTALL),
    re.compile(r"【问题】[：:]\s*\n?(.+)", re.DOTALL),
    re.compile(r"(?:^|\n)问题[：:]\s*\n?(.+)", re.DOTALL),
)
MULTI_SECTION_RE = re.compile(r"[一二三四五六七八九十]+[、.)．]")
NUMBERED_LIST_RE = re.compile(r"^\s*1[、.)．]\s*")
NORMALIZE_RE = re.compile(r"[\s\u3000，。！？、；：""''（）()\[\]【】]+")


def normalize_text(text: str) -> str:
    return NORMALIZE_RE.sub("", text.strip().lower())


def extract_law_names(text: str) -> list[str]:
    return LAW_NAME_RE.findall(text)


def _law_matches(name: str, patterns: Iterable[str]) -> bool:
    return any(p in name for p in patterns)


def count_labor_and_blacklist_laws(text: str) -> tuple[int, int]:
    labor = blacklist = 0
    for name in extract_law_names(text):
        if _law_matches(name, LABOR_LAW_WHITELIST):
            labor += 1
        if _law_matches(name, EXCLUDED_LAW_BLACKLIST):
            blacklist += 1
    return labor, blacklist


def has_core_keyword(text: str) -> bool:
    return any(k in text for k in CORE_KEYWORDS)


def is_strict_labor_row(row: dict, *, triplet: bool = False) -> tuple[bool, str]:
    """返回 (是否保留, 原因标签)。"""
    inp = str(row.get("input", "")).strip()
    out = str(row.get("output", "")).strip()
    refs = row.get("reference", [])
    ref_text = "\n".join(refs) if isinstance(refs, list) else str(refs or "")

    combined = f"{inp}\n{out}\n{ref_text}"
    if not inp or not out:
        return False, "empty"

    if has_core_keyword(inp) or has_core_keyword(out):
        return True, "core_keyword"

    labor, blacklist = count_labor_and_blacklist_laws(combined)
    if labor >= 1 and labor > blacklist:
        return True, "labor_law_cite"

    if triplet and isinstance(refs, list) and refs:
        ref_labor = sum(1 for r in refs if _law_matches(str(r), LABOR_LAW_WHITELIST))
        ref_black = sum(1 for r in refs if _law_matches(str(r), EXCLUDED_LAW_BLACKLIST))
        if ref_labor >= 1 and ref_labor >= ref_black:
            return True, "triplet_ref"

    if blacklist >= 2 and labor == 0:
        return False, "blacklist_law"

    if labor == 0 and not has_core_keyword(combined):
        return False, "not_labor"

    return False, "weak_labor"


def extract_triplet_question(input_text: str) -> str | None:
    text = input_text.strip()
    for pat in QUESTION_PATTERNS:
        m = pat.search(text)
        if m:
            question = m.group(1).strip()
            question = question.split("\n")[0].strip()
            if len(question) >= 4:
                return question
    # 兜底：取最后一段非空行（triplet 常把问题放末尾）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        if len(tail) >= 4 and "《" not in tail[:20]:
            return tail
    return None


def truncate_output(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    for sep in ("。", "；", "\n", "，"):
        pos = cut.rfind(sep)
        if pos > max_chars // 2:
            return cut[: pos + 1].strip()
    return cut.strip() + "……"
