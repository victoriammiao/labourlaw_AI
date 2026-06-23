"""Finite workflow orchestration for labor law RAG QA."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from web.model_runtime import generate_text, stream_text
from config import WEB_SEARCH_ENABLED
from web.legal_agent import run_legal_agent
from web.prompts import build_answer_prompt, build_attachment_answer_prompt, build_planner_prompt
from web.rag_client import (
    ask_rag_api,
    build_boundary_notice,
    build_no_evidence_message,
    format_rag_only_answer,
    has_relevant_evidence,
    is_out_of_knowledge_scope,
)


from web.attachment_context import (
    format_planner_attachment_hint,
    is_attachment_focused_query,
    select_attachment_context,
    should_use_attachment_first_path,
)
from web.query_normalize import normalize_user_query, query_changed
from web.history_utils import trim_turns
from web.references import UsedSources, extract_agent_sources


@dataclass
class WorkflowDecision:
    intent: str
    rewritten_query: str
    need_retrieve: bool
    direct_answer: str = ""
    reason: str = ""


def new_state() -> dict:
    return {
        "turns": [],
        "last_rewritten_query": "",
    }


def ensure_state(state) -> dict:
    if isinstance(state, dict):
        state.setdefault("turns", [])
        state.setdefault("last_rewritten_query", "")
        return state
    return new_state()


def remember_turn(state: dict, query: str, response: str, decision: WorkflowDecision) -> dict:
    if decision.rewritten_query:
        state["last_rewritten_query"] = decision.rewritten_query
    state["turns"].append(
        {
            "query": query,
            "response": response,
            "intent": decision.intent,
            "rewritten_query": decision.rewritten_query,
            "need_retrieve": decision.need_retrieve,
        }
    )
    state["turns"] = trim_turns(state["turns"])
    return state


def _extract_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _decision_from_dict(data: dict, query: str) -> WorkflowDecision:
    intent = str(data.get("intent") or "labor_question").strip()
    rewritten_query = str(data.get("rewritten_query") or query).strip()
    direct_answer = str(data.get("direct_answer") or "").strip()
    reason = str(data.get("reason") or "").strip()
    need_retrieve = bool(data.get("need_retrieve"))

    if intent == "non_labor":
        return WorkflowDecision(
            intent="web_search",
            rewritten_query=rewritten_query,
            need_retrieve=True,
            direct_answer="",
            reason=reason or "非劳动法问题，自动联网搜索",
        )

    if intent in {"greeting", "identity", "need_clarify"}:
        need_retrieve = False
    if not need_retrieve and not direct_answer:
        direct_answer = "你可以再补充一点具体情况，我会尽量结合劳动法资料帮你分析。"
    if intent == "identity":
        direct_answer = "我是劳动法律咨询助手，会先检索本地劳动法知识库，再结合资料给出初步参考。"
    return WorkflowDecision(intent, rewritten_query, need_retrieve, direct_answer, reason)


def _apply_attachment_overrides(
    decision: WorkflowDecision,
    query: str,
    attachment_context: str,
) -> WorkflowDecision:
    if not attachment_context:
        return decision

    if decision.intent in {"greeting", "identity", "empty"}:
        return decision

    if should_use_attachment_first_path(query, attachment_context):
        return WorkflowDecision(
            intent="attachment_qa",
            rewritten_query=decision.rewritten_query or query,
            need_retrieve=True,
            direct_answer="",
            reason=(decision.reason or "") + "；优先解读上传文件",
        )

    if decision.direct_answer or not decision.need_retrieve:
        intent = decision.intent
        if intent in {"legal_document", "need_clarify", "followup"}:
            intent = "labor_question"
        return WorkflowDecision(
            intent=intent,
            rewritten_query=decision.rewritten_query or query,
            need_retrieve=True,
            direct_answer="",
            reason=(decision.reason or "") + "；会话含附件，需结合文件回答",
        )

    return decision


def _fallback_decision(query: str, state: dict) -> WorkflowDecision:
    stripped = query.strip()
    if not stripped:
        return WorkflowDecision("empty", stripped, False, "请输入你的劳动法问题。")
    if stripped in {"你好", "您好", "hello", "hi", "嗨"}:
        return WorkflowDecision(
            "greeting",
            stripped,
            False,
            "你好，我是劳动法律咨询助手。你可以问我劳动合同、工资、加班、辞退、工伤、仲裁等问题。",
        )
    if len(stripped) <= 4 and state.get("last_rewritten_query"):
        rewritten = f"结合上一轮问题“{state['last_rewritten_query']}”继续回答：{stripped}"
        return WorkflowDecision("followup", rewritten, True, reason="fallback follow-up")
    return WorkflowDecision("labor_question", stripped, True, reason="fallback retrieval")


def plan_decision(
    query: str,
    state: dict,
    model=None,
    tokenizer=None,
    attachments: list[dict] | None = None,
    use_lora: bool = True,
) -> WorkflowDecision:
    if model is None or tokenizer is None:
        return _fallback_decision(query, state)

    attachment_hint = format_planner_attachment_hint(attachments)
    prompt = build_planner_prompt(query, state.get("turns", []), attachment_hint)
    try:
        raw = generate_text(model, tokenizer, prompt, max_new_tokens=360, use_lora=use_lora)
    except Exception as exc:
        print(f"Workflow planner failed: {exc}")
        return _fallback_decision(query, state)

    data = _extract_json_object(raw)
    if not data:
        print(f"Workflow planner returned non-JSON: {raw}")
        return _fallback_decision(query, state)
    return _decision_from_dict(data, query)


def _attachment_items(
    attachments: list[dict] | None,
    attachment_context: str,
    query: str,
    decision: WorkflowDecision,
) -> list[dict] | None:
    if not attachment_context or not attachments:
        return None
    if decision.intent == "attachment_qa":
        return list(attachments)
    if is_attachment_focused_query(query):
        return list(attachments)
    if decision.intent == "legal_document" and attachment_context:
        return list(attachments)
    return None


def prepare_response(
    query: str,
    state: dict,
    args,
    model=None,
    tokenizer=None,
    attachments: list[dict] | None = None,
    use_lora: bool = True,
):
    original_query = (query or "").strip()
    normalized_query = normalize_user_query(original_query)
    if query_changed(original_query, normalized_query):
        print(f"Query normalized: {original_query!r} -> {normalized_query!r}")

    attachment_context = select_attachment_context(attachments, normalized_query)
    if attachment_context:
        print(f"Attachment context chars: {len(attachment_context)}")
        names = "、".join(item.get("name", "?") for item in (attachments or []))
        print(f"Attachment files: {names}")

    decision = plan_decision(
        normalized_query, state, model, tokenizer, attachments, use_lora=use_lora
    )
    decision = _apply_attachment_overrides(decision, normalized_query, attachment_context)
    print(f"Workflow: {decision}")

    if decision.direct_answer:
        return decision, None, decision.direct_answer, None, UsedSources()

    if args.disable_rag:
        return decision, None, "RAG 已禁用。请启用 RAG 后再进行劳动法咨询。", None, UsedSources()

    if decision.intent == "attachment_qa":
        rag_data = ask_rag_api(decision.rewritten_query, args.rag_api_url, args.rag_top_k)
        rag_used = bool(rag_data and has_relevant_evidence(rag_data, normalized_query))
        context = rag_data.get("context", "") if rag_used else ""
        prompt = build_attachment_answer_prompt(
            original_query=original_query,
            attachment_context=attachment_context,
            turns=state.get("turns", []),
            context=context,
        )
        used = UsedSources(
            rag_data=rag_data if rag_used else None,
            attachments=_attachment_items(attachments, attachment_context, normalized_query, decision),
        )
        return decision, rag_data, "", prompt, used

    if decision.intent in {"legal_document", "web_search", "non_labor"}:
        try:
            agent_result = run_legal_agent(
                normalized_query,
                turns=state.get("turns", []),
                intent=decision.intent,
                original_query=original_query,
                attachment_context=attachment_context,
            )
        except Exception as exc:
            return decision, None, f"LangChain 工具调用失败：{exc}", None, UsedSources()
        answer = agent_result.get("answer") or "Agent 没有返回可用回答，请重试或补充更多信息。"
        notice = build_boundary_notice(normalized_query)
        if notice:
            answer = f"{notice}{answer}"
        agent_sources = extract_agent_sources(
            agent_result.get("raw"),
            rag_api_url=args.rag_api_url,
            rag_top_k=args.rag_top_k,
        )
        if _attachment_items(attachments, attachment_context, normalized_query, decision):
            agent_sources.attachments = _attachment_items(
                attachments, attachment_context, normalized_query, decision
            )
        return decision, None, answer, None, agent_sources

    rag_data = ask_rag_api(decision.rewritten_query, args.rag_api_url, args.rag_top_k)
    if rag_data and rag_data.get("results"):
        sources = [
            f"{item.get('rank')}:{item.get('source')}:{item.get('distance'):.4f}"
            for item in rag_data.get("results", [])
        ]
        print("RAG:", decision.rewritten_query, sources)
    rag_used = bool(rag_data and has_relevant_evidence(rag_data, normalized_query))
    attachment_items = _attachment_items(attachments, attachment_context, normalized_query, decision)

    if not rag_used and not attachment_context:
        answer = build_no_evidence_message(normalized_query)
        used = UsedSources()
        if WEB_SEARCH_ENABLED and is_out_of_knowledge_scope(normalized_query):
            try:
                agent_result = run_legal_agent(
                    normalized_query,
                    turns=state.get("turns", []),
                    intent="web_search",
                    original_query=original_query,
                    attachment_context=attachment_context,
                )
                web_answer = agent_result.get("answer") or ""
                if web_answer:
                    answer = f"{answer}\n\n---\n\n**联网搜索结果：**\n\n{web_answer}"
                used = extract_agent_sources(
                    agent_result.get("raw"),
                    rag_api_url=args.rag_api_url,
                    rag_top_k=args.rag_top_k,
                )
            except Exception as exc:
                answer = f"{answer}\n\n联网搜索失败：{exc}"
        elif WEB_SEARCH_ENABLED:
            answer = f"{answer}\n\n如需联网补充查询，请换一种更具体的问法，或说明需要最新政策/地区信息。"
        return decision, rag_data, answer, None, used

    if args.rag_only:
        return (
            decision,
            rag_data,
            format_rag_only_answer(decision, rag_data),
            None,
            UsedSources(rag_data=rag_data if rag_used else None),
        )

    prompt = build_answer_prompt(
        original_query=original_query,
        rewritten_query=decision.rewritten_query,
        context=rag_data["context"] if rag_used and rag_data else "",
        turns=state.get("turns", []),
        attachment_context=attachment_context,
    )
    used = UsedSources(
        rag_data=rag_data if rag_used else None,
        attachments=attachment_items,
    )
    return decision, rag_data, "", prompt, used


def stream_answer(model, tokenizer, prompt: str, use_lora: bool = True):
    yield from stream_text(model, tokenizer, prompt, use_lora=use_lora)
