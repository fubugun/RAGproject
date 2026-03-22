import re
from typing import Iterable


def _split_paragraphs(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if not buf.strip():
            buf = ""
            return
        s = buf.strip()
        if len(s) <= chunk_size:
            chunks.append(s)
            buf = ""
            return
        start = 0
        while start < len(s):
            end = min(start + chunk_size, len(s))
            piece = s[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(s):
                break
            start = end - chunk_overlap
            if start < 0:
                start = 0

    for para in paragraphs:
        if len(buf) + len(para) + 2 <= chunk_size:
            buf = f"{buf}\n\n{para}".strip() if buf else para
            continue
        flush_buf()
        if len(para) <= chunk_size:
            buf = para
        else:
            for sub in _window_chunks(para, chunk_size, chunk_overlap):
                chunks.append(sub)
            buf = ""

    flush_buf()
    return chunks


def _window_chunks(s: str, size: int, overlap: int) -> Iterable[str]:
    start = 0
    n = len(s)
    while start < n:
        end = min(start + size, n)
        piece = s[start:end].strip()
        if piece:
            yield piece
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
