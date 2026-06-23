"""In-memory multi-chat session helpers for the Gradio UI."""

from __future__ import annotations

from datetime import datetime
import uuid

from web.workflow import new_state


def _now() -> str:
    return datetime.now().strftime("%m-%d %H:%M")


def _session_title(first_query: str) -> str:
    text = (first_query or "").strip().replace("\n", " ")
    if not text:
        return "新对话"
    return text if len(text) <= 40 else f"{text[:40]}..."


def new_session_store() -> dict:
    session = _blank_session()
    session_id = session["id"]
    return {
        "active_id": session_id,
        "order": [session_id],
        "sessions": {session_id: session},
    }


def _blank_session(title: str = "新对话") -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "title_user_set": False,
        "updated_at": _now(),
        "messages": [],
        "workflow": new_state(),
        "attachments": [],
    }


def _next_blank_title(store: dict) -> str:
    existing = {session["title"] for session in store["sessions"].values()}
    base = "新对话"
    if base not in existing:
        return base
    index = 2
    while f"{base} ({index})" in existing:
        index += 1
    return f"{base} ({index})"


def get_active_session(store: dict | None) -> dict:
    store = ensure_store(store)
    return store["sessions"][store["active_id"]]


def ensure_store(store) -> dict:
    if not isinstance(store, dict) or "sessions" not in store:
        return new_session_store()
    store.setdefault("order", list(store["sessions"].keys()))
    store.setdefault("active_id", store["order"][0] if store["order"] else _blank_session()["id"])
    return store


def sync_active_session(store: dict, messages: list, workflow: dict, *, first_query: str = "") -> dict:
    store = ensure_store(store)
    session = store["sessions"][store["active_id"]]
    session["messages"] = messages
    session["workflow"] = workflow
    session.setdefault("attachments", [])
    session.setdefault("title_user_set", False)
    session["updated_at"] = _now()
    turn_count = len(workflow.get("turns", []))
    if first_query and not session["title_user_set"] and turn_count <= 1:
        session["title"] = _session_title(first_query)
    if store["active_id"] in store["order"]:
        store["order"].remove(store["active_id"])
    store["order"].insert(0, store["active_id"])
    return store


def get_active_title(store: dict) -> str:
    return get_active_session(store).get("title", "新对话")


def rename_active_session(store: dict, title: str) -> dict:
    store = ensure_store(store)
    session = store["sessions"][store["active_id"]]
    cleaned = (title or "").strip() or "新对话"
    session["title"] = cleaned if len(cleaned) <= 40 else f"{cleaned[:40]}..."
    session["title_user_set"] = True
    session["updated_at"] = _now()
    return store


def create_new_chat(store: dict, current_messages: list, current_workflow: dict) -> dict:
    store = ensure_store(store)
    sync_active_session(store, current_messages, current_workflow)
    session = _blank_session(title=_next_blank_title(store))
    store["sessions"][session["id"]] = session
    store["active_id"] = session["id"]
    store["order"].insert(0, session["id"])
    return store


def select_chat_by_index(store: dict, index: int, current_messages: list, current_workflow: dict) -> dict:
    store = ensure_store(store)
    sync_active_session(store, current_messages, current_workflow)
    if index is None or index < 0 or index >= len(store["order"]):
        return store
    store["active_id"] = store["order"][index]
    return store


def delete_chat_by_index(store: dict, index: int, current_messages: list, current_workflow: dict) -> dict:
    store = ensure_store(store)
    sync_active_session(store, current_messages, current_workflow)
    if index is None or index < 0 or index >= len(store["order"]):
        return store

    session_id = store["order"][index]
    was_active = session_id == store["active_id"]
    del store["sessions"][session_id]
    store["order"].pop(index)

    if not store["order"]:
        session = _blank_session()
        store["sessions"][session["id"]] = session
        store["active_id"] = session["id"]
        store["order"] = [session["id"]]
    elif was_active:
        next_index = min(index, len(store["order"]) - 1)
        store["active_id"] = store["order"][next_index]
    return store


def get_active_attachments(store: dict) -> list[dict]:
    session = get_active_session(store)
    session.setdefault("attachments", [])
    return list(session.get("attachments", []))


def add_attachments_to_session(store: dict, new_items: list[dict]) -> tuple[dict, list[str]]:
    from web.document_parser import MAX_FILES_PER_SESSION, MAX_SESSION_ATTACHMENT_CHARS

    store = ensure_store(store)
    session = store["sessions"][store["active_id"]]
    attachments = list(session.get("attachments", []))
    messages: list[str] = []

    for item in new_items:
        if any(existing.get("name") == item.get("name") for existing in attachments):
            messages.append(f"{item.get('name')} 已存在，已跳过重复文件")
            continue
        if len(attachments) >= MAX_FILES_PER_SESSION:
            messages.append(f"每个对话最多附加 {MAX_FILES_PER_SESSION} 个文件，{item.get('name')} 未添加")
            break
        total_chars = sum(existing.get("chars", 0) for existing in attachments) + item.get("chars", 0)
        if total_chars > MAX_SESSION_ATTACHMENT_CHARS:
            messages.append(
                f"当前对话附件总字数已达上限（{MAX_SESSION_ATTACHMENT_CHARS}），{item.get('name')} 未添加"
            )
            break
        attachments.append(item)
        messages.append(f"已添加 {item.get('name')}（{item.get('chars', 0)} 字）")

    session["attachments"] = attachments
    session["updated_at"] = _now()
    return store, messages


def clear_active_attachments(store: dict) -> dict:
    store = ensure_store(store)
    session = store["sessions"][store["active_id"]]
    session["attachments"] = []
    session["updated_at"] = _now()
    return store


def load_active_chat(store: dict) -> tuple[list, dict, list]:
    session = get_active_session(store)
    return (
        session.get("messages", []),
        session.get("workflow", new_state()),
        session.get("attachments", []),
    )


DELETE_COL_INDEX = 3


def build_history_table(store: dict) -> list[list]:
    store = ensure_store(store)
    rows = []
    for session_id in store["order"]:
        session = store["sessions"][session_id]
        turn_count = len(session.get("workflow", {}).get("turns", []))
        active_mark = "● " if session_id == store["active_id"] else ""
        rows.append(
            [
                f"{active_mark}{session['title']}",
                str(turn_count),
                session["updated_at"],
                "删除",
            ]
        )
    return rows


def history_table_update(store: dict):
    import gradio as gr

    return gr.update(value=build_history_table(store))


def active_title_update(store: dict):
    import gradio as gr

    return gr.update(value=get_active_title(store))
