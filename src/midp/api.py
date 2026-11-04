"""HTTP API: send a receipt image, get its fields back, and read them again later.

    uv run uvicorn midp.api:app --reload

Each extraction is stored in SQLite under data/, with the image as uploaded
and the model's answer. Nothing is corrected yet.
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

from midp.config import field_names, load_config
from midp.repository import ReceiptRepository

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

    fields = field_names(load_config(CONFIG_FILE))
    return MultimodalPredictor(model=gemini_model(), fields=fields, model_name=DEFAULT_MODEL)


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
    }
