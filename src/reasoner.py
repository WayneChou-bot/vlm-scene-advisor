"""Prompt assembly + parse + validation (+ one retry). Contract tests use a mock VLM."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydantic import ValidationError

from src.providers.base import VLMProvider
from src.schema import SceneAnalysis

PROMPT = """You are a scene-reasoning assistant for an autonomous warehouse forklift,
watching a fixed CCTV camera view of a loading dock.

THE OPERATING ZONE: the patch of floor DIRECTLY in front of the forklift's own doorway,
between that doorway and the yellow-black floor marking line. The neighboring doorway
with stacked boxes and the pallet storage area to the right are NOT part of the zone.
Judge a person's zone by where their FEET are: "in_zone" if their feet are on that patch,
"near" if within about 2 m of its edge, "far" otherwise. Report the zone as you actually
see it - do NOT inflate "near" into "in_zone" to be safe; the downstream policy layer
already applies safety margins. Your job is accurate reporting.

SCENE LABEL DEFINITIONS (pick exactly one):
- "clear": no person in or near the zone; stored goods (pallets, boxes) at the side are NORMAL and do not matter.
- "occupied": a person is in or near the zone.
- "loading": active loading/unloading work is visibly happening (goods being moved right now).
- "blocked": the zone itself is physically obstructed by objects OTHER THAN the operating forklift (fallen goods, a parked vehicle). Stored pallets beside the aisle are NOT "blocked".

Respond with PURE JSON only - no markdown fences, no commentary - exactly this shape:
{
  "persons": [{"position": "left|right|center|behind", "zone": "in_zone|near|far", "confidence": 0.0}],
  "person_in_zone": false,
  "obstructions": ["..."],
  "scene": "clear|occupied|loading|blocked",
  "reasoning": "one or two short sentences: what you see and why it matters",
  "suggested_mode": "efficient|cautious|stop",
  "answer": "one short imperative recommendation for the forklift"
}

Rules: any person in_zone -> suggested_mode "stop". Person near -> "cautious".
Only "efficient" when no person is visible and the scene is clear. When unsure, be conservative."""

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class ReasonerResult:
    raw: str
    parsed: Optional[SceneAnalysis]
    valid: bool
    retries: int
    latency_ms: float
    error: Optional[str] = None


def strip_fences(text: str) -> str:
    text = FENCE_RE.sub("", text.strip()).strip()
    # tolerate leading/trailing prose around a JSON object
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    return text


def parse_analysis(raw: str) -> SceneAnalysis:
    """Raises on failure; caller decides fail-safe."""
    return SceneAnalysis.model_validate(json.loads(strip_fences(raw)))


def analyze_frame(provider: VLMProvider, frame_bgr: np.ndarray, max_retries: int = 1) -> ReasonerResult:
    retries = 0
    raw = ""
    error = None
    t0 = time.perf_counter()
    while True:
        try:
            raw = provider.analyze(frame_bgr, PROMPT)
            parsed = parse_analysis(raw)
            return ReasonerResult(raw, parsed, True, retries, (time.perf_counter() - t0) * 1000)
        except (json.JSONDecodeError, ValidationError, Exception) as e:  # provider errors included
            error = f"{type(e).__name__}: {e}"
            if retries >= max_retries:
                # invalid → parsed=None; policy layer will fail-safe to cautious
                return ReasonerResult(raw, None, False, retries, (time.perf_counter() - t0) * 1000, error)
            retries += 1
