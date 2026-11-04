"""HTTP API: send a receipt image, get its fields back.

    uv run uvicorn midp.api:app --reload

There is no storage yet: each request is read, predicted and answered.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from midp.config import field_names, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "configs" / "receipts.yaml"
ACCEPTED_TYPES = {"image/png", "image/jpeg"}

app = FastAPI(title="midp")


@lru_cache
def get_predictor() -> Any:
    """The predictor used by the API. Built once, from the receipt config."""
    from midp.gemini import gemini_model
    from midp.predictors import MultimodalPredictor

    fields = field_names(load_config(CONFIG_FILE))
    return MultimodalPredictor(model=gemini_model(), fields=fields)


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
) -> dict:
    if file.content_type not in ACCEPTED_TYPES:
        raise HTTPException(status_code=415, detail="send a PNG or JPEG image")
    try:
        image = Image.open(BytesIO(file.file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="the file is not a readable image")
    return _jsonable(predictor.predict(image))
