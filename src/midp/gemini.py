"""Gemini as the model behind MultimodalPredictor.

The key is read from GEMINI_API_KEY, in the environment or in a .env file
at the project root. It is never written into code or logs.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter

from dotenv import load_dotenv
from google import genai
from google.genai import types

# TODO: move to setting argument? And evaluate ither models (3.5 flash lite?)
DEFAULT_MODEL = "gemini-3.5-flash-lite"


# TODO: is it neccessary to be a class?
def gemini_model(model: str = DEFAULT_MODEL):
    load_dotenv()
    # TODO: move to setting argument?
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # TODO: custom model error like in bacteria?
        raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")
    client = genai.Client(api_key=api_key)
    usage: Counter[str] = Counter()
    lock = threading.Lock()

    def call(image: bytes, prompt: str, mime_type: str = "application/json") -> str:
        """The model's answer to the prompt about the image: JSON by default, or plain text with mime_type="text/plain"."""
        started = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type=mime_type,
                temperature=0,
            ),
        )
        meta = response.usage_metadata
        with lock:  # calls run in parallel; the totals must not race
            usage["calls"] += 1
            usage["seconds"] += round(time.perf_counter() - started, 3)
            if meta is not None:
                usage["prompt_tokens"] += meta.prompt_token_count or 0
                usage["output_tokens"] += meta.candidates_token_count or 0
                usage["thinking_tokens"] += meta.thoughts_token_count or 0
        return response.text or ""

    call.usage = usage  # running totals, read by evals.predict
    call.model = model  # name recorded with each run
    return call
