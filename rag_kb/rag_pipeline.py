from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from rag_kb.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_CHAT_MODEL,
)
from rag_kb.vector_store import RetrievedChunk, VectorStore


SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。请仅根据下方「参考资料」回答问题。
若参考资料不足以回答，请明确说明「根据当前知识库无法回答该问题」，不要编造事实。
回答使用简体中文，条理清晰。"""


def build_user_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    if not contexts:
        return (
            f"用户问题：{question}\n\n"
            "参考资料：（无相关片段，相似度未达阈值或知识库为空）\n"
            "请直接说明无法从知识库中找到依据，不要猜测。"
        )
    blocks = []
    for i, c in enumerate(contexts, start=1):
        src = f"（来源：{c.source}）" if c.source else ""
        blocks.append(f"[{i}]{src}\n{c.text}")
    joined = "\n\n---\n\n".join(blocks)
    return f"用户问题：{question}\n\n参考资料：\n{joined}\n\n请基于参考资料作答。"


@dataclass
class RAGResult:
    answer: str
    contexts: list[RetrievedChunk]
    used_context: bool
    raw_messages: list[dict[str, Any]]


def run_rag(
    store: VectorStore,
    question: str,
    top_k: int = 5,
    similarity_threshold: float = 0.25,
    chat_model: str | None = None,
    embedding_model: str | None = None,
) -> RAGResult:
    contexts = store.search(
        question,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        model_name=embedding_model,
    )
    used = len(contexts) > 0
    user_content = build_user_prompt(question, contexts)

    if not OPENAI_API_KEY:
        return RAGResult(
            answer="未配置 OPENAI_API_KEY，无法调用大模型。请在环境变量或 .env 中设置。",
            contexts=contexts,
            used_context=used,
            raw_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    model = chat_model or OPENAI_CHAT_MODEL
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    answer = (resp.choices[0].message.content or "").strip()
    return RAGResult(
        answer=answer,
        contexts=contexts,
        used_context=used,
        raw_messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
