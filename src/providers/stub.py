"""Stub provider (M0): zero-API development.

NOT a model. It fakes a plausible VLM answer so the pipeline runs without API cost:
motion mask (background subtraction vs. the first frame, assumed empty) gated by a
loose safety-vest color range. Works on both the synthetic 2D clip and the Blender
3D render. Documented as a stub everywhere; never used for accuracy claims.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

import math

from src.config import NEAR_FACTOR, ZONE_ANGLE, ZONE_AXES, ZONE_CENTER
from src.providers.base import VLMProvider

VEST_LO = np.array([2, 60, 90])     # loose warm gate (HSV) - motion-gated
VEST_HI = np.array([32, 255, 255])
CORE_LO = np.array([2, 60, 90])     # strict orange (vest core) - separates worker
CORE_HI = np.array([16, 255, 255])  # from the yellow forklift (hue ~22)
DIFF_THRESH = 28
MIN_PIXELS = 40


class StubProvider(VLMProvider):
    name = "stub"
    model = "bgsub+hsv-heuristic (not a model)"

    def __init__(self):
        self._bg = None  # first frame = empty-scene reference

    def analyze(self, frame_bgr: np.ndarray, prompt: str) -> str:
        h, w = frame_bgr.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if self._bg is None:
            self._bg = gray.astype(np.int16)
        motion = (np.abs(gray.astype(np.int16) - self._bg) > DIFF_THRESH).astype(np.uint8) * 255
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, VEST_LO, VEST_HI)
        mask = cv2.bitwise_and(color, color, mask=motion)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # connected components: the worker is the blob with the most strict-orange
        # (vest) pixels - the moving yellow forklift scores ~0 there
        core = cv2.inRange(hsv, CORE_LO, CORE_HI)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        best, score = 0, 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_PIXELS:
                continue
            sc = int((core[labels == i] > 0).sum())
            if sc > score:
                best, score = i, sc
        count = 0 if best == 0 or score < 40 else int(stats[best, cv2.CC_STAT_AREA])
        mask = (labels == best).astype(np.uint8) if count else mask * 0

        if count < MIN_PIXELS:
            return json.dumps(
                {
                    "persons": [],
                    "person_in_zone": False,
                    "obstructions": [],
                    "scene": "clear",
                    "reasoning": "No person is near the autonomous forklift. The aisle and doorway are clear.",
                    "suggested_mode": "efficient",
                    "answer": 'Signal forklift to enable "Efficient Mode."',
                }
            )

        ys, xs = np.nonzero(mask)
        # ground point: lowest pixel of the worker blob (~feet)
        cx, cy = xs.mean() / w, min(1.0, ys.max() / h)
        zx, zy = ZONE_CENTER
        rx, ry = ZONE_AXES
        th = math.radians(ZONE_ANGLE)
        dx, dy = cx - zx, cy - zy
        px_ = dx * math.cos(th) + dy * math.sin(th)   # rotate into ellipse frame
        py_ = -dx * math.sin(th) + dy * math.cos(th)
        d = (px_ / rx) ** 2 + (py_ / ry) ** 2

        position = "left" if cx < zx - 0.05 else ("right" if cx > zx + 0.05 else "center")

        if d <= 1.0:
            payload = {
                "persons": [{"position": position, "zone": "in_zone", "confidence": 0.95}],
                "person_in_zone": True,
                "obstructions": [],
                "scene": "occupied",
                "reasoning": f"A person has entered the forklift operating zone ({position}). Immediate hazard.",
                "suggested_mode": "stop",
                "answer": "Alert: halt the forklift and sound the zone alarm until clear.",
            }
        elif d <= NEAR_FACTOR ** 2:
            payload = {
                "persons": [{"position": position, "zone": "near", "confidence": 0.9}],
                "person_in_zone": False,
                "obstructions": [],
                "scene": "occupied",
                "reasoning": f"A person is approaching the operating zone from the {position}.",
                "suggested_mode": "cautious",
                "answer": "Alert: slow the forklift; person near the operating zone.",
            }
        else:
            payload = {
                "persons": [{"position": position, "zone": "far", "confidence": 0.85}],
                "person_in_zone": False,
                "obstructions": [],
                "scene": "clear",
                "reasoning": f"A person is visible far on the {position}; not near the operating zone.",
                "suggested_mode": "cautious",
                "answer": 'Keep "Cautious Mode" while a person is in view.',
            }
        return json.dumps(payload)
