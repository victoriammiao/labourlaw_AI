"""LangChain DeepSeek agent for tool-calling legal document workflows."""

from __future__ import annotations

from functools import lru_cache

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, WEB_SEARCH_ENABLED
from web.document_prompts import LEGAL_AGENT_SYSTEM_PROMPT
from web.legal_tools import build_tools


from web.history_utils import trim_turns


def _format_history(turns: list[dict] | None) -> str:
    if not turns:
        return "无"
    lines = []
    for item in trim_turns(turns):
        lines.append(f"用户：{item.get('query', '')}")
        response = item.get("response", "")
        if response:
            lines.append(f"助手：{response}")
    return "\n".join(lines)


def _extract_answer(result: dict) -> str:
    from langchain_core.messages import AIMessage

    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


@lru_cache(maxsize=1)
def get_legal_agent():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is required. Put it in .env.")

    try:
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain agent dependencies are missing. Install requirements.txt again."
        ) from exc

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.2,
    )
    tools = build_tools()
    return create_agent(
        llm,
        tools=tools,
        system_prompt=LEGAL_AGENT_SYSTEM_PROMPT,
        debug=True,
    )


def run_legal_agent(
    user_input: str,
    turns: list[dict] | None = None,
    *,
    intent: str = "",
    original_query: str = "",
    attachment_context: str = "",
) -> dict:
    if intent in {"web_search", "non_labor"} and not WEB_SEARCH_ENABLED:
        return {
            "answer": (
                "联网搜索尚未启用。请在项目根目录创建 `.env`，设置 "
                "`WEB_SEARCH_ENABLED=true` 和 `TAVILY_API_KEY=...` 后重试。"
            ),
            "raw": None,
        }

    from langchain_core.messages import HumanMessage

    agent = get_legal_agent()
    history = _format_history(turns)
    raw = (original_query or user_input).strip()
    if attachment_context.strip():
        content = (
            "用户已上传文件，以下内容已从文件中解析提取。"
            "请优先结合这些内容回答；如信息不足，再说明需要补充哪些事实。\n\n"
            f"{attachment_context.strip()}\n\n"
            f"最近对话：\n{history}\n\n"
        )
        if raw and raw != user_input.strip():
            content += (
                f"用户原始输入：\n{raw}\n\n"
                f"规范化理解：\n{user_input}\n\n"
                "请按规范化理解回答用户。"
            )
        else:
            content += f"用户需求：\n{user_input}"
    elif raw and raw != user_input.strip():
        content = (
            f"最近对话：\n{history}\n\n"
            f"用户原始输入：\n{raw}\n\n"
            f"规范化理解：\n{user_input}\n\n"
            "请按规范化理解回答用户。"
        )
    else:
        content = f"最近对话：\n{history}\n\n用户需求：\n{user_input}"
    result = agent.invoke({"messages": [HumanMessage(content=content)]})
    return {
        "answer": _extract_answer(result),
        "raw": result,
    }
