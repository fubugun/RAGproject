from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

RAGAS_METRIC_KEYS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

RAGAS_METRIC_LABELS_ZH = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确度",
    "context_recall": "上下文召回",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_JSONL = PROJECT_ROOT / "data" / "eval_samples.jsonl"


def run_ragas_evaluation(
    dataset_path: Path,
    top_k: int = 5,
    threshold: float = 0.25,
) -> tuple[Any | None, str | None]:
    from datasets import Dataset

    from rag_kb.config import (
        EMBEDDING_MODEL,
        HF_ENDPOINT,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        OPENAI_CHAT_MODEL,
        STORE_DIR,
    )
    from rag_kb.rag_pipeline import run_rag
    from rag_kb.vector_store import VectorStore

    if not OPENAI_API_KEY:
        return None, "未配置 OPENAI_API_KEY"

    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        return None, f"缺少 RAGAS 依赖: {e}"

    if HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        return None, f"未找到数据集: {dataset_path}"

    store = VectorStore.load(STORE_DIR)
    rows: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return None, "评测集为空或没有有效行"

    questions: list[str] = []
    answers: list[str] = []
    contexts_list: list[list[str]] = []
    ground_truth: list[str] = []

    for row in rows:
        q = row["question"]
        gt = str(row.get("ground_truth", "") or "")
        res = run_rag(
            store,
            q,
            top_k=top_k,
            similarity_threshold=threshold,
        )
        ctx_texts = [c.text for c in res.contexts]
        if not ctx_texts:
            ctx_texts = ["（无检索上下文）"]

        questions.append(q)
        answers.append(res.answer)
        contexts_list.append(ctx_texts)
        ground_truth.append(gt)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truth,
        }
    )

    llm = ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.2,
    )

    emb_kw: dict[str, Any] = {"llm": llm}
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        hf_name = EMBEDDING_MODEL
        if hf_name.startswith("sentence-transformers/"):
            hf_name = hf_name.split("/", 1)[1]
        emb_kw["embeddings"] = HuggingFaceEmbeddings(model_name=hf_name)
    except Exception:
        pass

    try:
        result = evaluate(
            ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            **emb_kw,
        )
    except Exception as e:
        return None, f"RAGAS evaluate 失败: {e}"

    return result, None


def extract_ragas_aggregate_scores(result: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: None for k in RAGAS_METRIC_KEYS}
    if result is None:
        return out

    def _ok(x: Any) -> bool:
        try:
            f = float(x)
            return not math.isnan(f)
        except (TypeError, ValueError):
            return False

    try:
        sc = getattr(result, "scores", None)
        if isinstance(sc, dict):
            for k in RAGAS_METRIC_KEYS:
                v = sc.get(k)
                if _ok(v):
                    out[k] = float(v)
            if any(v is not None for v in out.values()):
                return out
    except Exception:
        pass

    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for k in RAGAS_METRIC_KEYS:
                if k not in df.columns:
                    continue
                m = df[k].mean()
                if _ok(m):
                    out[k] = float(m)
    except Exception:
        pass

    return out


def format_ragas_summary_lines(scores: dict[str, float | None]) -> str:
    lines = []
    for k in RAGAS_METRIC_KEYS:
        label = RAGAS_METRIC_LABELS_ZH.get(k, k)
        v = scores.get(k)
        if v is not None and _ok_float(v):
            lines.append(f"{label}: {v:.4f}")
        else:
            lines.append(f"{label}: —")
    return "\n".join(lines)


def _ok_float(v: Any) -> bool:
    try:
        f = float(v)
        return not math.isnan(f)
    except (TypeError, ValueError):
        return False


def format_ragas_output(result: Any) -> str:
    if result is None:
        return ""
    try:
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            s = df.to_string(index=False)
            if s.strip():
                return s
    except Exception:
        pass
    try:
        scores = getattr(result, "scores", None)
        if isinstance(scores, dict) and scores:
            return json.dumps(scores, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass
    out = str(result).strip()
    return out if out else repr(result)


def print_ragas_report(result: Any) -> None:
    scores = extract_ragas_aggregate_scores(result)
    print(format_ragas_summary_lines(scores))
    print(format_ragas_output(result))
