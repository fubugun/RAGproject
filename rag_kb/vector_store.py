from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] | None = None


class VectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._sources: list[str | None] = []
        self._meta: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids.clear()
        self._texts.clear()
        self._sources.clear()
        self._meta.clear()
        self._matrix = None

    def add_chunks(
        self,
        texts: list[str],
        sources: list[str | None] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        model_name: str | None = None,
    ) -> int:
        if not texts:
            return 0
        n0 = len(self._ids)
        new_ids = [str(uuid.uuid4()) for _ in texts]
        src_list = sources if sources is not None else [None] * len(texts)
        meta_list = metadatas if metadatas is not None else [{} for _ in texts]
        from rag_kb.embeddings import encode_texts

        emb = encode_texts(texts, model_name=model_name)

        if self._matrix is None or len(self._matrix) == 0:
            self._matrix = emb
        else:
            self._matrix = np.vstack([self._matrix, emb])

        self._ids.extend(new_ids)
        self._texts.extend(texts)
        self._sources.extend(src_list)
        self._meta.extend(meta_list)
        return len(self._ids) - n0

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        model_name: str | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip() or self._matrix is None or len(self._matrix) == 0:
            return []

        from rag_kb.embeddings import encode_texts

        q = encode_texts([query], model_name=model_name)[0]
        scores = self._matrix @ q

        indexed = list(enumerate(scores.tolist()))
        indexed.sort(key=lambda x: x[1], reverse=True)

        out: list[RetrievedChunk] = []
        for idx, score in indexed:
            if score < similarity_threshold:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=self._ids[idx],
                    text=self._texts[idx],
                    score=float(score),
                    source=self._sources[idx],
                    metadata=self._meta[idx] or None,
                )
            )
            if len(out) >= top_k:
                break
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "chunks.jsonl"
        emb_path = directory / "embeddings.npy"

        with meta_path.open("w", encoding="utf-8") as f:
            for i in range(len(self._ids)):
                row = {
                    "id": self._ids[i],
                    "text": self._texts[i],
                    "source": self._sources[i],
                    "metadata": self._meta[i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if self._matrix is not None:
            np.save(emb_path, self._matrix)
        else:
            if emb_path.exists():
                emb_path.unlink()

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        store = cls()
        meta_path = directory / "chunks.jsonl"
        emb_path = directory / "embeddings.npy"
        if not meta_path.exists():
            return store

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                store._ids.append(row["id"])
                store._texts.append(row["text"])
                store._sources.append(row.get("source"))
                store._meta.append(row.get("metadata") or {})

        if emb_path.exists():
            store._matrix = np.load(emb_path)
        return store
