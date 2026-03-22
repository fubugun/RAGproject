from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG 知识库问答", layout="wide")

import tempfile
from pathlib import Path

from rag_kb.chunking import chunk_text
from rag_kb.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OPENAI_CHAT_MODEL,
    STORE_DIR,
)
from rag_kb.document_loader import load_document
from rag_kb.ragas_eval import (
    DEFAULT_EVAL_JSONL,
    RAGAS_METRIC_KEYS,
    RAGAS_METRIC_LABELS_ZH,
    extract_ragas_aggregate_scores,
    format_ragas_output,
    print_ragas_report,
    run_ragas_evaluation,
)

STORE_FILE_MARKER = STORE_DIR / "chunks.jsonl"


def _hr() -> None:
    st.markdown("---")


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _has_modern_chat() -> bool:
    return hasattr(st, "chat_input") and hasattr(st, "chat_message")


def get_store():
    from rag_kb.vector_store import VectorStore

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = (
            VectorStore.load(STORE_DIR)
            if STORE_FILE_MARKER.exists()
            else VectorStore()
        )
    return st.session_state.vector_store


def persist_store(store) -> None:
    store.save(STORE_DIR)


def _render_context_expander(label: str, contexts: list) -> None:
    with st.expander(label):
        if not contexts:
            st.info("无片段超过当前相似度阈值。")
            return
        for i, c in enumerate(contexts, 1):
            src = c.get("source") if isinstance(c, dict) else c.source
            score = c.get("score") if isinstance(c, dict) else c.score
            text = c.get("text") if isinstance(c, dict) else c.text
            st.markdown(f"**[{i}]** 分数 `{score:.3f}` · {src or '未知来源'}")
            st.text(str(text)[:2000])


def _append_and_answer(store, q: str, top_k: int, sim_threshold: float) -> None:
    from rag_kb.rag_pipeline import run_rag

    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner("处理中…"):
        result = run_rag(
            store,
            q,
            top_k=top_k,
            similarity_threshold=sim_threshold,
        )
    ctx_serializable = [
        {"text": c.text, "score": c.score, "source": c.source}
        for c in result.contexts
    ]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "meta": {"contexts": ctx_serializable},
        }
    )


def main() -> None:
    st.title("RAG 知识库问答")

    if "ragas_out" not in st.session_state:
        st.session_state["ragas_out"] = None
    if "ragas_err" not in st.session_state:
        st.session_state["ragas_err"] = None
    if "ragas_scores" not in st.session_state:
        st.session_state["ragas_scores"] = None

    store = get_store()

    with st.sidebar:
        st.subheader("知识库")
        st.text(f"已索引块数: {len(store)}")
        uploaded = st.file_uploader(
            "上传文档（.pdf / .txt / .md）",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        chunk_size = st.number_input(
            "分块大小（字符）", min_value=100, max_value=4000, value=CHUNK_SIZE
        )
        chunk_overlap = st.number_input(
            "分块重叠（字符）", min_value=0, max_value=500, value=CHUNK_OVERLAP
        )
        if st.button("将上传文件加入知识库"):
            if not uploaded:
                st.warning("请先选择文件。")
            else:
                total = 0
                with st.spinner("索引中…"):
                    for f in uploaded:
                        suffix = Path(f.name).suffix or ".txt"
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(f.getvalue())
                            tmp_path = Path(tmp.name)
                        try:
                            text = load_document(tmp_path)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                        chunks = chunk_text(
                            text,
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                        )
                        sources = [f.name] * len(chunks)
                        n = store.add_chunks(chunks, sources=sources)
                        total += n
                persist_store(store)
                st.success(f"已写入 {total} 个文本块。")
                _rerun()

        if st.button("清空知识库（不可恢复）"):
            store.clear()
            persist_store(store)
            st.session_state.pop("messages", None)
            st.success("已清空。")
            _rerun()

        _hr()
        st.subheader("检索参数")
        top_k = st.slider("Top-K", 1, 20, DEFAULT_TOP_K)
        sim_threshold = st.slider(
            "相似度阈值（余弦，越大越严）",
            0.0,
            1.0,
            float(DEFAULT_SIMILARITY_THRESHOLD),
            0.01,
        )
        _hr()
        st.markdown(
            f"**嵌入模型**: `{EMBEDDING_MODEL}`  \n**对话模型**: `{OPENAI_CHAT_MODEL}`"
        )

        _hr()
        st.subheader("RAGAS 评测")
        eval_path_in = st.text_input(
            "评测集 JSONL",
            value=str(DEFAULT_EVAL_JSONL),
        )
        if st.button("运行 RAGAS 评测"):
            path = Path(eval_path_in.strip() or str(DEFAULT_EVAL_JSONL))
            try:
                with st.spinner("RAGAS…"):
                    res, err = run_ragas_evaluation(
                        path,
                        top_k=top_k,
                        threshold=sim_threshold,
                    )
                if err:
                    st.session_state["ragas_out"] = None
                    st.session_state["ragas_scores"] = None
                    st.session_state["ragas_err"] = err
                else:
                    st.session_state["ragas_err"] = None
                    st.session_state["ragas_scores"] = extract_ragas_aggregate_scores(res)
                    st.session_state["ragas_out"] = format_ragas_output(res) or str(res)
                    print_ragas_report(res)
            except Exception as e:
                st.session_state["ragas_out"] = None
                st.session_state["ragas_scores"] = None
                st.session_state["ragas_err"] = f"评测异常: {e}"
            _rerun()

    if st.session_state.get("ragas_err"):
        st.error(st.session_state["ragas_err"])
    if st.session_state.get("ragas_scores"):
        st.subheader("RAGAS")
        cols = st.columns(4)
        for i, key in enumerate(RAGAS_METRIC_KEYS):
            v = st.session_state["ragas_scores"].get(key)
            label = RAGAS_METRIC_LABELS_ZH.get(key, key)
            if v is not None and v == v:
                cols[i].metric(label, f"{float(v):.3f}")
            else:
                cols[i].metric(label, "—")
        detail = st.session_state.get("ragas_out")
        if detail:
            with st.expander("明细"):
                st.code(detail, language=None)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if _has_modern_chat():
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and m.get("meta"):
                    _render_context_expander(
                        "检索到的片段", m["meta"].get("contexts") or []
                    )

        q = st.chat_input("输入问题…")
        if q:
            from rag_kb.rag_pipeline import run_rag

            st.session_state.messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)

            with st.chat_message("assistant"):
                with st.spinner("处理中…"):
                    result = run_rag(
                        store,
                        q,
                        top_k=top_k,
                        similarity_threshold=sim_threshold,
                    )
                st.markdown(result.answer)
                ctx_serializable = [
                    {"text": c.text, "score": c.score, "source": c.source}
                    for c in result.contexts
                ]
                _render_context_expander("检索到的片段", ctx_serializable)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "meta": {"contexts": ctx_serializable},
                }
            )
    else:
        for m in st.session_state.messages:
            if m["role"] == "user":
                st.markdown(f"**你**：{m['content']}")
            else:
                st.markdown(f"**助手**：{m['content']}")
                if m.get("meta"):
                    _render_context_expander(
                        "检索到的片段", m["meta"].get("contexts") or []
                    )

        with st.form("legacy_ask"):
            q_in = st.text_input("输入问题", placeholder="输入后点击发送")
            send = st.form_submit_button("发送")
        if send and q_in and q_in.strip():
            _append_and_answer(store, q_in.strip(), top_k, sim_threshold)
            _rerun()


if __name__ == "__main__":
    main()
