from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_kb.config import EMBEDDING_MODEL, HF_ENDPOINT


def _apply_hf_hub_endpoint() -> None:
    if HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_ENDPOINT


@lru_cache(maxsize=1)
def get_model(model_name: str | None = None) -> SentenceTransformer:
    _apply_hf_hub_endpoint()
    name = model_name or EMBEDDING_MODEL
    try:
        return SentenceTransformer(name)
    except OSError as e:
        raise OSError(
            f"{e}\n\n可在 .env 设置 HF_ENDPOINT=https://hf-mirror.com，"
            "或将 EMBEDDING_MODEL 设为本地模型目录。"
        ) from e


def encode_texts(
    texts: Sequence[str],
    model_name: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    model = get_model(model_name)
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=len(texts) > 16,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms
