"""HTTP API: send a receipt image, get its fields back, and read them again later.

    uv run uvicorn midp.api:app --reload

Each extraction is stored in SQLite under data/, with the image as uploaded
and the model's answer. Reviewers can add corrections, which are kept apart
from the original answer.
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

from midp.config import load_type_config
from midp.repository import ReceiptRepository, apply_corrections

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "configs" / "receipts.yaml"
DATABASE_FILE = REPO_ROOT / "data" / "midp.sqlite3"
ACCEPTED_TYPES = {"image/png", "image/jpeg"}

app = FastAPI(title="midp")


@lru_cache
def get_predictor() -> Any:
    """The predictor used by the API. Built once, from the receipt config."""
    from midp.gemini import DEFAULT_MODEL, gemini_model
    from midp.predictors import MultimodalPredictor

    return MultimodalPredictor(model=gemini_model(), config=load_type_config(CONFIG_FILE), model_name=DEFAULT_MODEL)


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


@lru_cache
def _config_fields() -> dict[str, dict]:
    return {field["name"]: field for field in load_type_config(CONFIG_FILE)["fields"]}


def _is_number(value: Any) -> bool:
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool))


def _check_correction(fields: dict[str, Any]) -> None:
    """Refuse a correction that names an unknown field or gives a value of the wrong kind."""
    known = _config_fields()
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
    predictor: Annotated[Any, Depends(get_predictor)],
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
    result = _jsonable(predictor.predict(image))
    seconds = round(time.perf_counter() - started, 3)
    receipt_id = repository.save(
        image=data,
        content_type=file.content_type,
        model=predictor.model_name,
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
    if not repository.exists(receipt_id):
        raise HTTPException(status_code=404, detail="no receipt with that id")
    _check_correction(fields)
    repository.add_correction(receipt_id, fields)
    return {"id": receipt_id, "correction": fields}
