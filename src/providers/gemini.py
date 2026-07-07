"""Gemini provider (M1). Requires GEMINI_API_KEY in .env / environment.
Import of google-genai is deferred so the rest of the project runs without it.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from src.providers.base import VLMProvider


class GeminiProvider(VLMProvider):
    name = "gemini"

    def __init__(self, model: str = None):
        from google import genai  # deferred import

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (put it in .env, never commit it)")
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def analyze(self, frame_bgr: np.ndarray, prompt: str) -> str:
        from google.genai import types

        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        cfg = None
        try:  # disable thinking for latency (supported on gemini-2.5 models)
            cfg = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        except Exception:
            pass
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"),
                prompt,
            ],
            config=cfg,
        )
        return resp.text or ""
