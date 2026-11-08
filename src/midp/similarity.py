"""Decide a document's type by looking at similar, already typed examples.

The image is transcribed to text, the text is embedded as a vector, and its
k nearest examples vote on the type. The examples are stored in the examples
table (pgvector) with their type, text and vector, so a new document is compared
with them, not described to a model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from midp.classifier import Classification

Transcribe = Callable[[Any], str]
Embed = Callable[[list[str]], np.ndarray]
Nearest = Callable[[list[float], int], list[tuple[str, float]]]


def vote(neighbours: list[tuple[str, float]]) -> tuple[str, float]:
    """The type most neighbours have. On a tie, the type of the single closest neighbour among the tied types.

    neighbours are (type, similarity), most similar first. Returns the winning type and the best similarity.
    """
    if not neighbours:
        raise ValueError("no examples to vote with")
    votes = Counter(label for label, _ in neighbours)
    best_count = max(votes.values())
    tied = [label for label, count in votes.items() if count == best_count]
    winner = max(tied, key=lambda label: max(s for label_, s in neighbours if label_ == label))
    return winner, neighbours[0][1]


class SimilarityClassifier:
    """The same vote over the k nearest examples, found by `nearest` (the examples table)."""

    def __init__(self, transcribe: Transcribe, embed: Embed, nearest: Nearest, k: int = 3) -> None:
        self._transcribe = transcribe
        self._embed = embed
        self._nearest = nearest
        self._k = k

    def inspect(self, image: Any) -> dict[str, Any]:
        """One decision with what produced it: the transcribed text and its vector."""
        text = self._transcribe(image)
        vector = self._embed([text])[0]
        document_type, confidence = vote(self._nearest(vector.tolist(), self._k))
        return {"document_type": document_type, "confidence": confidence, "text": text, "vector": vector.tolist()}

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
