"""HTTP API: send a document image, get its type and fields back, and read them later.

    uv run uvicorn midp.api:app --reload

Each image is first classified against the registry of document types, then
extracted with that type's config. The result, the image and any corrections
are stored in the database. An image of an unknown type is stored
without fields, so a reviewer can still see it.

Known limitation: a delivery note with no title can be read as an invoice.
Its `document_type` is then "invoice", which is provisional. The invoice
checks flag it only when a required field is missing; otherwise the fields
returned for it are not real (the model fills in prices the document does
not show). Do not treat such a result as validated. No similarity or margin
rule flags it: those were measured and do not separate the errors.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from decimal import Decimal
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlmodel import Session

from midp.classifier import TypeClassifier
from midp.config import load_document_types, load_type_config
from midp.db import make_engine
from midp.repository import DocumentRepository, apply_corrections
from midp.review import review_document

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES_FILE = REPO_ROOT / "configs" / "document_types.yaml"
DATABASE_URL = os.environ.get("MIDP_DATABASE_URL", f"sqlite:///{REPO_ROOT / 'data' / 'midp.sqlite3'}")
ACCEPTED_TYPES = {"image/png", "image/jpeg"}

app = FastAPI(title="midp")


@lru_cache
def get_model() -> Any:
    from midp.gemini import gemini_model

    return gemini_model()


@lru_cache
def get_types() -> list[dict]:
    return load_document_types(TYPES_FILE)


@lru_cache
def get_classifier() -> Any:
    return TypeClassifier(model=get_model(), types=get_types())


@lru_cache
def get_predictors() -> dict[str, Any]:
    """One extraction predictor per document type, each built from that type's config."""
    from midp.gemini import DEFAULT_MODEL
    from midp.predictors import MultimodalPredictor

    return {
        entry["name"]: MultimodalPredictor(
            model=get_model(), config=load_type_config(entry["config"]), model_name=DEFAULT_MODEL
        )
        for entry in get_types()
    }


@lru_cache
def get_type_configs() -> dict[str, dict]:
    """Each type's full config, by name, for its fields and its checks."""
    return {entry["name"]: load_type_config(entry["config"]) for entry in get_types()}


@lru_cache
def get_engine() -> Any:
    return make_engine(DATABASE_URL)


def get_session() -> Iterator[Session]:
    """One session per request, as in orkid. Writes commit on it; the rest is closed at the end."""
    with Session(get_engine()) as session:
        yield session


def get_repository(session: Annotated[Session, Depends(get_session)]) -> DocumentRepository:
    return DocumentRepository(session)


def _jsonable(value: Any) -> Any:
    """Decimals are sent as whole numbers when they are whole, otherwise as floats."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _fields_of(document_type: str) -> dict[str, dict]:
    config = next(entry["config"] for entry in get_types() if entry["name"] == document_type)
    return {field["name"]: field for field in load_type_config(config)["fields"]}


def _is_number(value: Any) -> bool:
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool))


def _check_correction(fields: dict[str, Any], document_type: str) -> None:
    """Refuse a correction that names a field the document's type does not have, or a wrong kind of value."""
    known = _fields_of(document_type)
    errors: list[str] = []
    for name, value in fields.items():
        spec = known.get(name)
        if spec is None:
            errors.append(f"unknown field: {name}")
        elif spec.get("kind") == "list":
            if value is not None and not isinstance(value, list):
                errors.append(f"{name} must be a list")
                continue
            item_spec = {item["name"]: item for item in spec.get("item", [])}
            for index, item in enumerate(value or []):
                if not isinstance(item, dict):
                    errors.append(f"{name}[{index}] must be an object")
                    continue
                for key, item_value in item.items():
                    if key not in item_spec:
                        errors.append(f"unknown field in {name}[{index}]: {key}")
                    elif item_spec[key]["type"] == "amount" and not _is_number(item_value):
                        errors.append(f"{name}[{index}].{key} must be a number")
        elif spec["type"] == "amount" and not _is_number(value):
            errors.append(f"{name} must be a number")
    if errors:
        raise HTTPException(status_code=422, detail=errors)


@app.post("/documents/extract")
def extract(
    file: Annotated[UploadFile, File()],
    classifier: Annotated[Any, Depends(get_classifier)],
    predictors: Annotated[dict[str, Any], Depends(get_predictors)],
    type_configs: Annotated[dict[str, dict], Depends(get_type_configs)],
    repository: Annotated[DocumentRepository, Depends(get_repository)],
) -> dict:
    if file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(status_code=415, detail="send a PNG or JPEG image")
    data = file.file.read()
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="the file is not a readable image")

    started = time.perf_counter()
    classification = classifier.classify(image)
    document_type = classification.document_type
    if document_type is None:
        result: dict[str, Any] = {"document_type": None, "confidence": classification.confidence}
        model = "none"
    else:
        predictor = predictors[document_type]
        result = {
            "document_type": document_type,
            "confidence": classification.confidence,
            **_jsonable(predictor.predict(image)),
        }
        model = predictor.model_name
    checks = type_configs[document_type].get("checks", {}) if document_type else {}
    needs_review, problems = review_document(result, checks)
    result["needs_review"] = needs_review
    result["problems"] = problems
    seconds = round(time.perf_counter() - started, 3)
    document_id = repository.save(
        image=data,
        content_type=file.content_type,
        model=model,
        seconds=seconds,
        result=result,
    )
    return {"id": document_id, "result": result}


@app.get("/documents/{document_id}")
def read_document(document_id: str, repository: Annotated[DocumentRepository, Depends(get_repository)]) -> dict:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no document with that id")
    return {
        "id": document.id,
        "created_at": document.created_at,
        "model": document.model,
        "seconds": document.seconds,
        "result": document.result,
        "corrections": document.corrections,
        "final": apply_corrections(document.result, document.corrections),
    }


@app.post("/documents/{document_id}/corrections")
def add_correction(
    document_id: str,
    fields: dict[str, Any],
    repository: Annotated[DocumentRepository, Depends(get_repository)],
) -> dict:
    document = repository.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="no document with that id")
    document_type = document.result.get("document_type")
    if document_type is None:
        raise HTTPException(status_code=422, detail="this document has no known type to correct against")
    _check_correction(fields, document_type)
    repository.add_correction(document_id, fields)
    return {"id": document_id, "correction": fields}
