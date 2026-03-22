from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset  # noqa: E402

from rag_kb.config import OPENAI_API_KEY, STORE_DIR  # noqa: E402
from rag_kb.rag_pipeline import run_rag  # noqa: E402
from rag_kb.vector_store import VectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGAS 评测，需: pip install -r requirements-ragas.txt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval_samples.jsonl",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("请设置 OPENAI_API_KEY")
        sys.exit(1)

    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        print("请先安装: pip install -r requirements-ragas.txt")
        sys.exit(1)

    if not args.dataset.exists():
        print(f"未找到数据集: {args.dataset}")
        sys.exit(1)

    store = VectorStore.load(STORE_DIR)
    rows: list[dict] = []
    with args.dataset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

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
            top_k=args.top_k,
            similarity_threshold=args.threshold,
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

    result = evaluate(
        ds,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    print(result)


if __name__ == "__main__":
    main()
