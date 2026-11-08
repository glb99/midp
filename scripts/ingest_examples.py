"""Store the typed examples for similarity classification in the examples table.

    uv run python -m scripts.ingest_examples
    uv run python -m scripts.ingest_examples --replace

Each example image is transcribed as plain text with the model, embedded, and
stored as a row with its type, its text and its vector. Runs against
MIDP_DATABASE_URL.

Without --replace it refuses to run if the table already has rows, so it
cannot store duplicates. With --replace it deletes every stored example first.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from sqlmodel import Session, delete, func, select

from evals.classify import TRANSCRIBE, example_set
from midp.db import ExampleRow, make_engine
from midp.gemini import DEFAULT_MODEL, gemini_model
from midp.predictors import _png_bytes
from midp.repository import ExampleRepository
from midp.similarity import gemini_embedder

BATCH = 20


def main() -> None:
    parser = argparse.ArgumentParser(prog="scripts.ingest_examples")
    parser.add_argument("--replace", action="store_true", help="delete the stored examples first")
    args = parser.parse_args()

    url = os.environ.get("MIDP_DATABASE_URL")
    if not url:
        raise SystemExit("MIDP_DATABASE_URL is not set")

    engine = make_engine(url)
    with Session(engine) as session:
        existing = session.exec(select(func.count()).select_from(ExampleRow)).one()
    if existing and not args.replace:
        raise SystemExit(f"the examples table already has {existing} rows; nothing was stored")

    examples = example_set()
    model = gemini_model(DEFAULT_MODEL)
    embed = gemini_embedder()

    def transcribe(item: tuple[str, object]) -> str:
        return model(_png_bytes(item[1]), TRANSCRIBE, mime_type="text/plain")

    with ThreadPoolExecutor(max_workers=16) as pool:
        texts = list(pool.map(transcribe, examples))

    with Session(engine) as session:
        if existing:
            session.exec(delete(ExampleRow))
            session.commit()
        repository = ExampleRepository(session)
        for start in range(0, len(examples), BATCH):
            vectors = embed(texts[start : start + BATCH])
            for offset, vector in enumerate(vectors):
                document_type = examples[start + offset][0]
                repository.add(
                    document_type=document_type,
                    text=texts[start + offset],
                    embedding=vector.tolist(),
                )

    counts: dict[str, int] = {}
    for document_type, _ in examples:
        counts[document_type] = counts.get(document_type, 0) + 1
    print(f"stored {len(examples)} examples: {counts}")


if __name__ == "__main__":
    main()
