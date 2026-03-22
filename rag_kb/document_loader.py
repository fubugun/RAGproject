from pathlib import Path

from pypdf import PdfReader


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return load_text_file(path)
    raise ValueError(f"不支持的文件类型: {suffix}")
