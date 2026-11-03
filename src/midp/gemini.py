"""Gemini as the model behind MultimodalPredictor.

The key is read from GEMINI_API_KEY, in the environment or in a .env file
at the project root. It is never written into code or logs.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from midp.predictors import ModelCall

# TODO: move to setting argument? And evaluate ither models (3.5 flash lite?)
DEFAULT_MODEL = "gemini-3.8-flash"


# TODO: is it neccessary to be a class?
def gemini_model(model: str = DEFAULT_MODEL) -> ModelCall:
    load_dotenv()
    # TODO: move to setting argument?
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # TODO: custom model error like in bacteria?
        raise RuntimeError("GEMINI_API_KEY is not set (environment or .env)")
    client = genai.Client(api_key=api_key)

    def call(image: bytes, prompt: str) -> str:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
        return response.text or ""

    return call
