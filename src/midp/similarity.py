"""Decide a document's type by looking at similar, already typed examples.

The image is transcribed to text, the text is embedded as a vector, and its
k nearest examples vote on the type. The examples are stored as vectors with
their type, so a new document is compared with them, not described to a model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from midp.classifier import Classification

Transcribe = Callable[[Any], str]
Embed = Callable[[list[str]], np.ndarray]


class SimilarityClassifier:
    """Votes among the k most similar examples. Returns the winning type and its best similarity."""

    def __init__(self, transcribe: Transcribe, embed: Embed, examples: list[tuple[str, Any]], k: int = 3) -> None:
        self._transcribe = transcribe
        self._embed = embed
        self._labels = [label for label, _ in examples]
        self._vectors = embed([self._transcribe(image) for _, image in examples])
        self._k = k

    def _score(self, image: Any) -> tuple[str, np.ndarray, np.ndarray]:
        text = self._transcribe(image)
        vector = self._embed([text])[0]
        return text, vector, self._vectors @ vector

    def inspect(self, image: Any) -> dict[str, Any]:
        """Everything about one decision: text, vector, best similarity per type, margin, winner."""
        text, vector, similarities = self._score(image)
        top = np.argsort(-similarities)[: self._k]
        votes = Counter(self._labels[i] for i in top)
        best_count = max(votes.values())
        winners = [label for label, count in votes.items() if count == best_count]
        # on a tie, the winner is the type of the single closest example among the tied types
        winner = max(winners, key=lambda label: max(similarities[i] for i in top if self._labels[i] == label))
        best_per_type = {
            label: float(max(similarities[i] for i, l in enumerate(self._labels) if l == label))
            for label in dict.fromkeys(self._labels)
        }
        ranked = sorted(best_per_type.values(), reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else None
        return {
            "document_type": winner,
            "confidence": float(similarities[top[0]]),
            "text": text,
            "vector": vector.tolist(),
            "best_per_type": best_per_type,
            "margin": margin,
        }

    def classify(self, image: Any) -> Classification:
        decision = self.inspect(image)
        return Classification(document_type=decision["document_type"], confidence=decision["confidence"])

EMBEDDING_MODEL = "gemini-embedding-001"


def gemini_embedder() -> Embed:
    """Embeddings from Gemini, normalised to unit length so a dot product is the cosine similarity."""
    import os

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def embed(texts: list[str]) -> np.ndarray:
        out = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vectors = np.array([e.values for e in out.embeddings])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return embed


def load_similarity_config(path) -> dict:
    """The method's settings: k, how many examples vote."""
    from pathlib import Path

    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {"k": int(config["k"])}
