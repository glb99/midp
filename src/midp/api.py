"""HTTP API: send a document image, get its type and fields back, and read them later.

    uv run uvicorn midp.api:app --reload

Each image is first classified against the registry of document types, then
extracted with that type's config. The result, the image and any corrections
are stored in SQLite under data/. An image of an unknown type is stored
without fields, so a reviewer can still see it.
"""

from __future__ import annotations

import time
from decimal import Decimal
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from midp.classifier import TypeClassifier
from midp.config import load_document_types, load_type_config
from midp.repository import ReceiptRepository, apply_corrections

REPO_ROOT = Path(__file__).resolve().parents[2]
TYPES_FILE = REPO_ROOT / "configs" / "document_types.yaml"
DATABASE_FILE = REPO_ROOT / "data" / "midp.sqlite3"
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
def get_repository() -> ReceiptRepository:
    return ReceiptRepository(DATABASE_FILE)


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


@app.post("/receipts/extract")
def extract(
    file: Annotated[UploadFile, File()],
    classifier: Annotated[Any, Depends(get_classifier)],
    predictors: Annotated[dict[str, Any], Depends(get_predictors)],
    repository: Annotated[ReceiptRepository, Depends(get_repository)],
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
    seconds = round(time.perf_counter() - started, 3)
    receipt_id = repository.save(
        image=data,
        content_type=file.content_type,
        model=model,
        seconds=seconds,
        result=result,
    )
    return {"id": receipt_id, "result": result}


@app.get("/receipts/{receipt_id}")
def read_receipt(receipt_id: str, repository: Annotated[ReceiptRepository, Depends(get_repository)]) -> dict:
    receipt = repository.get(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="no receipt with that id")
    return {
        "id": receipt.id,
        "created_at": receipt.created_at,
        "model": receipt.model,
        "seconds": receipt.seconds,
        "result": receipt.result,
        "corrections": receipt.corrections,
        "final": apply_corrections(receipt.result, receipt.corrections),
    }


@app.post("/receipts/{receipt_id}/corrections")
def add_correction(
    receipt_id: str,
    fields: dict[str, Any],
    repository: Annotated[ReceiptRepository, Depends(get_repository)],
) -> dict:
    receipt = repository.get(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="no receipt with that id")
    document_type = receipt.result.get("document_type")
    if document_type is None:
        raise HTTPException(status_code=422, detail="this document has no known type to correct against")
    _check_correction(fields, document_type)
    repository.add_correction(receipt_id, fields)
    return {"id": receipt_id, "correction": fields}
