from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_kb.config import OPENAI_API_KEY  # noqa: E402
from rag_kb.ragas_eval import (  # noqa: E402
    DEFAULT_EVAL_JSONL,
    print_ragas_report,
    run_ragas_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_JSONL,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("请设置 OPENAI_API_KEY")
        sys.exit(1)

    result, err = run_ragas_evaluation(
        args.dataset,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    if err:
        print(err)
        sys.exit(1)
    print_ragas_report(result)


if __name__ == "__main__":
    main()
