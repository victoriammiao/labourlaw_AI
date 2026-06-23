"""Parse uploaded user documents for chat attachment context."""

from __future__ import annotations

import os


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
MAX_SINGLE_FILE_CHARS = 8000
MAX_SESSION_ATTACHMENT_CHARS = 45000
MAX_FILES_PER_SESSION = 5


def _truncate(text: str, limit: int = MAX_SINGLE_FILE_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n...(内容过长，已截断)..."


def _parse_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _parse_docx(path: str) -> str:
    from docx import Document

    doc = Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text).strip()


def _parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def parse_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持的文件类型：{ext}")

    if ext == ".pdf":
        content = _parse_pdf(path)
    elif ext in {".docx", ".doc"}:
        if ext == ".doc":
            raise ValueError("旧版 .doc 请先另存为 .docx 或 PDF 后再上传")
        content = _parse_docx(path)
    else:
        content = _parse_text(path)

    if not content:
        raise ValueError(f"未能从 {name} 提取到文本，可能是扫描件或空文件")

    content = _truncate(content)
    return {
        "name": name,
        "type": ext.lstrip("."),
        "content": content,
        "chars": len(content),
    }


def parse_uploaded_files(files: list | None) -> tuple[list[dict], list[str]]:
    parsed: list[dict] = []
    errors: list[str] = []
    for item in files or []:
        path = getattr(item, "name", None) or (item if isinstance(item, str) else None)
        if not path:
            continue
        try:
            parsed.append(parse_file(path))
        except Exception as exc:
            errors.append(f"{os.path.basename(path)}：{exc}")
    return parsed, errors
