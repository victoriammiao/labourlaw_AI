"""Gradio UI wiring for the labor law workflow frontend."""

from __future__ import annotations

import gradio as gr

from web.attachment_context import format_attachment_status, is_attachment_focused_query
from web.references import format_used_sources
from web.chat_sessions import (
    DELETE_COL_INDEX,
    active_title_update,
    add_attachments_to_session,
    build_history_table,
    clear_active_attachments,
    create_new_chat,
    delete_chat_by_index,
    get_active_attachments,
    get_active_title,
    history_table_update,
    load_active_chat,
    new_session_store,
    rename_active_session,
    select_chat_by_index,
    sync_active_session,
)
from web.document_parser import parse_uploaded_files
from web.model_runtime import is_peft_model, release_cuda_cache
from web.workflow import ensure_state, new_state, prepare_response, remember_turn, stream_answer


APP_CSS = """
html,
body {
  width: 100%;
  height: auto !important;
  min-width: 1100px;
  min-height: 100%;
  margin: 0;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  background:
    radial-gradient(circle at top left, rgba(39, 105, 255, 0.16), transparent 30%),
    radial-gradient(circle at top right, rgba(21, 184, 166, 0.14), transparent 28%),
    #f7f8fb;
}
#root,
#root > div,
.gradio-app,
.app,
main {
  width: 100%;
  height: auto !important;
  min-height: 100%;
  overflow: visible !important;
}
/* AutoDL / 嵌入页 iframe 高度容器：固定视口高度 + 内部纵向滚动 */
[data-iframe-height],
.iframe-height,
.iframe-auto-height {
  display: block;
  width: 100% !important;
  height: 100vh !important;
  max-height: 100vh !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
  -webkit-overflow-scrolling: touch;
  box-sizing: border-box;
}
[data-iframe-height] iframe,
.iframe-height iframe {
  display: block;
  width: 100% !important;
  min-height: 100% !important;
}
html::-webkit-scrollbar,
body::-webkit-scrollbar,
[data-iframe-height]::-webkit-scrollbar {
  width: 12px;
}
html::-webkit-scrollbar-track,
body::-webkit-scrollbar-track,
[data-iframe-height]::-webkit-scrollbar-track {
  background: rgba(15, 23, 42, 0.06);
  border-radius: 999px;
}
html::-webkit-scrollbar-thumb,
body::-webkit-scrollbar-thumb,
[data-iframe-height]::-webkit-scrollbar-thumb {
  background: rgba(100, 116, 139, 0.55);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
html::-webkit-scrollbar-thumb:hover,
body::-webkit-scrollbar-thumb:hover,
[data-iframe-height]::-webkit-scrollbar-thumb:hover {
  background: rgba(71, 85, 105, 0.75);
  background-clip: content-box;
}
.gradio-container {
  width: 92% !important;
  max-width: 1680px !important;
  min-width: 1100px !important;
  height: auto !important;
  max-height: none !important;
  min-height: calc(100vh - 24px) !important;
  margin: 0 auto !important;
  padding: 12px 0 72px !important;
  box-sizing: border-box;
  overflow: visible !important;
}
.app-shell {
  padding: 14px 22px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 22px 70px rgba(15, 23, 42, 0.10);
  backdrop-filter: blur(14px);
}
.hero-title {
  margin: 0;
  font-size: 26px;
  font-weight: 760;
  letter-spacing: -0.03em;
  color: #0f172a;
}
.hero-subtitle {
  margin-top: 8px;
  color: #64748b;
  font-size: 14px;
}
.status-pill {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  margin-top: 14px;
  padding: 6px 11px;
  border-radius: 999px;
  background: #eef6ff;
  color: #155e75;
  font-size: 12px;
}
.sidebar-panel {
  padding: 14px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
  height: 100%;
  box-sizing: border-box;
}
.sidebar-title {
  font-size: 14px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 10px;
}
.chat-main-panel {
  padding: 14px;
  border: 1px solid rgba(15, 23, 42, 0.08);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: visible;
}
.action-row {
  position: sticky;
  bottom: 0;
  z-index: 20;
  background: rgba(255, 255, 255, 0.96);
  border-top: 1px solid rgba(15, 23, 42, 0.08);
  padding: 10px 0 4px;
  margin-top: 4px;
}
.action-row button {
  min-height: 42px;
}
.reference-card {
  margin-top: 12px;
  padding: 12px 14px;
  border: 1px solid rgba(37, 99, 235, 0.15);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(239, 246, 255, 0.86), rgba(248, 250, 252, 0.92));
}
.reference-card h4 {
  margin-top: 0;
}
#chatbot {
  height: 360px !important;
  min-height: 280px !important;
  max-height: 42vh !important;
  flex-shrink: 0;
}
#chatbot .wrap,
#chatbot .bubble-wrap {
  height: 100% !important;
  max-height: 100% !important;
  overflow-y: auto !important;
}
.sidebar-panel .dataframe {
  max-height: 52vh;
  overflow-y: auto !important;
}
footer {
  display: none !important;
}
.history-table table tbody tr {
  cursor: pointer;
  transition: background 0.15s ease;
}
.history-table table tbody tr:hover {
  background: rgba(37, 99, 235, 0.08) !important;
}
.history-table table tbody tr td:last-child {
  color: #dc2626;
  font-weight: 600;
  text-align: center;
}
.history-table table tbody tr td:last-child:hover {
  background: rgba(220, 38, 38, 0.12) !important;
}
.history-table table thead th:last-child {
  text-align: center;
}
"""

SCROLL_FIX_HEAD = """
<script>
(function () {
  function applyScroll(root) {
    if (!root) return;
    root.querySelectorAll("[data-iframe-height]").forEach(function (el) {
      el.style.setProperty("overflow-y", "auto", "important");
      el.style.setProperty("overflow-x", "hidden", "important");
      el.style.setProperty("max-height", "100vh", "important");
      el.style.setProperty("height", "100vh", "important");
    });
    if (root.documentElement) {
      root.documentElement.style.overflowY = "auto";
      root.documentElement.style.height = "auto";
    }
    if (root.body) {
      root.body.style.overflowY = "auto";
      root.body.style.height = "auto";
      root.body.style.minHeight = "100%";
    }
  }
  function run() {
    applyScroll(document);
    try {
      if (window.parent && window.parent !== window) {
        applyScroll(window.parent.document);
      }
    } catch (err) {}
  }
  run();
  window.addEventListener("load", run);
  window.addEventListener("resize", run);
  if (window.MutationObserver) {
    new MutationObserver(run).observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["style", "class", "data-iframe-height"],
    });
  }
})();
</script>
"""


def launch_demo(args, model, tokenizer) -> None:
    lora_available = model is not None and is_peft_model(model)

    def _use_lora_from_mode(model_mode: str) -> bool:
        return lora_available and model_mode == "lora"

    def _sidebar_updates(sessions_state):
        return history_table_update(sessions_state), active_title_update(sessions_state)

    def _attachment_status_update(sessions_state):
        return format_attachment_status(get_active_attachments(sessions_state))

    def _finalize_turn(chatbot, workflow_state, sessions_state, query, response, decision):
        workflow_state = remember_turn(workflow_state, query, response, decision)
        sessions_state = sync_active_session(
            sessions_state,
            chatbot,
            workflow_state,
            first_query=query,
        )
        table, title = _sidebar_updates(sessions_state)
        return chatbot, workflow_state, sessions_state, table, title, _attachment_status_update(sessions_state)

    def predict(_query, _chatbot, _workflow_state, _sessions_state, _model_mode):
        state = ensure_state(_workflow_state)
        query = (_query or "").strip()
        use_lora = _use_lora_from_mode(_model_mode)
        model_label = "微调模型 (LoRA v2)" if use_lora else "基座模型 (未微调)"
        print(f"Model mode: {model_label}")
        attachments = get_active_attachments(_sessions_state)
        if is_attachment_focused_query(query) and not attachments:
            _chatbot.append({"role": "user", "content": query})
            hint = (
                "你问的是上传文件相关的问题，但当前对话还没有已解析的附件。\n\n"
                "请先在下方「上传文件」区域选择 PDF / Word / TXT，"
                "选好后会自动解析；看到「已附加文件」列表后再提问。"
            )
            _chatbot.append({"role": "assistant", "content": hint})
            table, title = _sidebar_updates(_sessions_state)
            yield _chatbot, state, _sessions_state, table, title, _attachment_status_update(_sessions_state)
            return

        print(f"User: {query}")
        _chatbot.append({"role": "user", "content": query})
        _chatbot.append({"role": "assistant", "content": "正在理解你的问题，并检索相关资料..."})
        table, title = _sidebar_updates(_sessions_state)
        attachment_status = _attachment_status_update(_sessions_state)
        yield _chatbot, state, _sessions_state, table, title, attachment_status

        decision, rag_data, response, prompt, used_sources = prepare_response(
            query,
            state,
            args,
            model,
            tokenizer,
            attachments=attachments,
            use_lora=use_lora,
        )
        references = "" if args.rag_only else format_used_sources(used_sources)
        model_tag = (
            f"\n\n---\n\n*本次回答模型：{model_label}*"
            if lora_available
            else ""
        )
        if prompt is None:
            _chatbot[-1] = {"role": "assistant", "content": f"{response}{references}{model_tag}"}
            chatbot, state, sessions_state, table, title, attachment_status = _finalize_turn(
                _chatbot, state, _sessions_state, query, response, decision
            )
            yield chatbot, state, sessions_state, table, title, attachment_status
            return

        _chatbot[-1] = {"role": "assistant", "content": ""}
        yield _chatbot, state, _sessions_state, table, title, attachment_status

        for new_text in stream_answer(model, tokenizer, prompt, use_lora=use_lora):
            response += new_text
            _chatbot[-1] = {"role": "assistant", "content": response}
            yield _chatbot, state, _sessions_state, table, title, attachment_status

        _chatbot[-1] = {"role": "assistant", "content": f"{response}{references}{model_tag}"}
        chatbot, state, sessions_state, table, title, attachment_status = _finalize_turn(
            _chatbot, state, _sessions_state, query, response, decision
        )
        print(f"Assistant: {response}")
        yield chatbot, state, sessions_state, table, title, attachment_status

    def reset_user_input():
        return gr.update(value="")

    def on_new_chat(chatbot, workflow_state, sessions_state):
        sessions_state = create_new_chat(sessions_state, chatbot, workflow_state)
        chatbot, workflow_state, _attachments = load_active_chat(sessions_state)
        release_cuda_cache()
        table, title = _sidebar_updates(sessions_state)
        print(f"New chat created. sessions={len(sessions_state['order'])}")
        return chatbot, workflow_state, sessions_state, table, title, _attachment_status_update(sessions_state)

    def on_table_action(evt: gr.SelectData, chatbot, workflow_state, sessions_state):
        row = evt.index[0]
        col = evt.index[1] if len(evt.index) > 1 else 0
        if col == DELETE_COL_INDEX:
            sessions_state = delete_chat_by_index(
                sessions_state,
                row,
                chatbot,
                workflow_state,
            )
        else:
            sessions_state = select_chat_by_index(
                sessions_state,
                row,
                chatbot,
                workflow_state,
            )
        chatbot, workflow_state, _attachments = load_active_chat(sessions_state)
        table, title = _sidebar_updates(sessions_state)
        return chatbot, workflow_state, sessions_state, table, title, _attachment_status_update(sessions_state)

    def on_upload_files(files, sessions_state):
        if not files:
            return sessions_state, _attachment_status_update(sessions_state), gr.update()
        before = len(get_active_attachments(sessions_state))
        parsed, errors = parse_uploaded_files(files)
        sessions_state, notes = add_attachments_to_session(sessions_state, parsed)
        after = get_active_attachments(sessions_state)
        print(f"Upload batch: selected={len(files or [])}, parsed={len(parsed)}, session {before}->{len(after)} files")
        for item in after:
            print(f"  attachment: {item.get('name')} ({item.get('chars', 0)} chars)")
        status = _attachment_status_update(sessions_state)
        if parsed:
            status = f"**本次解析 {len(parsed)} 个文件。**\n\n" + status
        if errors or notes:
            status = status + "\n\n" + "\n".join(f"- {line}" for line in errors + notes)
        return sessions_state, status, gr.update()

    def on_clear_attachments(sessions_state):
        sessions_state = clear_active_attachments(sessions_state)
        return sessions_state, _attachment_status_update(sessions_state)

    def on_save_title(title, sessions_state):
        sessions_state = rename_active_session(sessions_state, title)
        table, title_update = _sidebar_updates(sessions_state)
        return sessions_state, table, title_update

    initial_store = new_session_store()

    with gr.Blocks(fill_height=False) as demo:
        gr.Markdown(
            """
<div class="app-shell">
  <div class="hero-title">劳动法律咨询 AI 顾问</div>
  <div class="hero-subtitle">支持多会话管理。每个对话内会保留完整上下文，用于路由、RAG 与 Agent 回答。</div>
  <div class="status-pill">Multi-Chat · RAG Workflow · Qwen · DeepSeek Agent</div>
</div>
"""
        )

        if lora_available:
            gr.Markdown(
                """
<div style="margin:10px 0 0;padding:10px 14px;border-radius:12px;background:#f0fdf4;border:1px solid rgba(22,163,74,0.2);color:#166534;font-size:13px;">
已加载 LoRA 微调权重，可在下方切换「基座模型」与「微调模型」对比回答效果。
</div>
"""
            )

        sessions_state = gr.State(initial_store)
        workflow_state = gr.State(new_state())

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=260):
                with gr.Group(elem_classes="sidebar-panel"):
                    gr.Markdown('<div class="sidebar-title">会话管理</div>')
                    new_chat_btn = gr.Button("＋ 新建对话", variant="primary")
                    gr.Markdown(
                        '<div style="color:#64748b;font-size:12px;margin:8px 0;">'
                        "点击某行切换对话；点击右侧「删除」列可删除该对话</div>"
                    )
                    history_table = gr.Dataframe(
                        headers=["标题", "轮数", "更新时间", "操作"],
                        datatype=["str", "str", "str", "str"],
                        value=build_history_table(initial_store),
                        interactive=False,
                        wrap=True,
                        label="历史对话",
                        elem_classes="history-table",
                    )

            with gr.Column(scale=8):
                with gr.Group(elem_classes="chat-main-panel"):
                    with gr.Row():
                        chat_title = gr.Textbox(
                            label="对话标题",
                            value=get_active_title(initial_store),
                            placeholder="发送第一个问题后会自动命名，也可手动修改",
                            scale=5,
                        )
                        save_title_btn = gr.Button("保存标题", scale=1, variant="secondary")
                    chatbot = gr.Chatbot(
                        label="当前对话",
                        elem_id="chatbot",
                        height=360,
                        avatar_images=(None, None),
                    )
                    query = gr.Textbox(
                        lines=2,
                        label="请输入问题",
                        placeholder="例如：公司拖欠工资怎么办？被辞退有没有补偿？",
                    )
                    if lora_available:
                        model_mode = gr.Radio(
                            choices=[
                                ("微调模型 (LoRA v2)", "lora"),
                                ("基座模型 (未微调)", "base"),
                            ],
                            value="lora",
                            label="回答模型",
                            info="同一套 RAG 检索结果下，对比微调前后生成效果",
                        )
                    else:
                        model_mode = gr.State("base")
                    with gr.Row(elem_classes="action-row"):
                        submit_btn = gr.Button("发送", variant="primary", scale=2)
                        new_chat_btn_secondary = gr.Button("新建对话", variant="secondary", scale=1)
                    with gr.Accordion("上传文件（PDF / Word / TXT）", open=True):
                        file_upload = gr.File(
                            label="选择文件（选中后自动解析并附加）",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            type="filepath",
                        )
                        with gr.Row():
                            upload_btn = gr.Button("重新解析所选文件", variant="secondary")
                            clear_files_btn = gr.Button("清空附件", variant="secondary")
                        attachment_status = gr.Markdown(
                            value=format_attachment_status([]),
                        )

        submit_btn.click(
            predict,
            [query, chatbot, workflow_state, sessions_state, model_mode],
            [chatbot, workflow_state, sessions_state, history_table, chat_title, attachment_status],
            show_progress=True,
        )
        submit_btn.click(reset_user_input, [], [query])

        new_chat_btn.click(
            on_new_chat,
            [chatbot, workflow_state, sessions_state],
            [chatbot, workflow_state, sessions_state, history_table, chat_title, attachment_status],
            show_progress=True,
        )
        new_chat_btn_secondary.click(
            on_new_chat,
            [chatbot, workflow_state, sessions_state],
            [chatbot, workflow_state, sessions_state, history_table, chat_title, attachment_status],
            show_progress=True,
        )
        history_table.select(
            on_table_action,
            [chatbot, workflow_state, sessions_state],
            [chatbot, workflow_state, sessions_state, history_table, chat_title, attachment_status],
        )
        upload_btn.click(
            on_upload_files,
            [file_upload, sessions_state],
            [sessions_state, attachment_status, file_upload],
        )
        file_upload.change(
            on_upload_files,
            [file_upload, sessions_state],
            [sessions_state, attachment_status, file_upload],
        )
        clear_files_btn.click(
            on_clear_attachments,
            [sessions_state],
            [sessions_state, attachment_status],
        )
        save_title_btn.click(
            on_save_title,
            [chat_title, sessions_state],
            [sessions_state, history_table, chat_title],
        )
        chat_title.submit(
            on_save_title,
            [chat_title, sessions_state],
            [sessions_state, history_table, chat_title],
        )

        gr.Markdown(
            """
<div style="color:#64748b;font-size:12px;text-align:center;margin-top:8px;">
本系统仅提供劳动法信息查询和初步参考，不构成正式法律意见。复杂个案请咨询专业律师或劳动仲裁机构。
</div>
"""
        )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
        css=APP_CSS,
        head=SCROLL_FIX_HEAD,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
