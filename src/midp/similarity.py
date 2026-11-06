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

    def classify(self, image: Any) -> Classification:
        vector = self._embed([self._transcribe(image)])[0]
        similarities = self._vectors @ vector
        top = np.argsort(-similarities)[: self._k]
        votes = Counter(self._labels[i] for i in top)
        best_count = max(votes.values())
        winners = [label for label, count in votes.items() if count == best_count]
        # on a tie, the winner is the type of the single closest example among the tied types
        winner = max(winners, key=lambda label: max(similarities[i] for i in top if self._labels[i] == label))
        return Classification(document_type=winner, confidence=float(similarities[top[0]]))


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
