"""Reasoner parse/validation tests with a mock VLM (we test OUR contract, not a model)."""
import json

import numpy as np
import pytest

from src.providers.base import VLMProvider
from src.reasoner import analyze_frame, parse_analysis, strip_fences

FRAME = np.zeros((36, 64, 3), np.uint8)

GOOD = json.dumps(
    {
        "persons": [],
        "person_in_zone": False,
        "obstructions": [],
        "scene": "clear",
        "reasoning": "Empty aisle.",
        "suggested_mode": "efficient",
        "answer": "Enable Efficient Mode.",
    }
)


class MockProvider(VLMProvider):
    name = "mock"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def analyze(self, frame, prompt):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_strip_json_fences():
    assert strip_fences(f"```json\n{GOOD}\n```") == GOOD


def test_strip_prose_around_json():
    assert json.loads(strip_fences(f"Sure! Here you go:\n{GOOD}\nHope that helps."))


def test_parse_valid():
    a = parse_analysis(GOOD)
    assert a.scene == "clear"


def test_analyze_frame_happy_path():
    res = analyze_frame(MockProvider([GOOD]), FRAME)
    assert res.valid and res.retries == 0 and res.parsed.suggested_mode == "efficient"
    assert res.latency_ms >= 0


def test_analyze_frame_retries_once_then_succeeds():
    p = MockProvider(["{not json", GOOD])
    res = analyze_frame(p, FRAME)
    assert res.valid and res.retries == 1 and p.calls == 2


def test_analyze_frame_invalid_twice_returns_invalid():
    p = MockProvider(["{not json", "also | not : json"])
    res = analyze_frame(p, FRAME)
    assert not res.valid and res.parsed is None and res.error
    assert res.raw  # raw preserved for telemetry


def test_analyze_frame_provider_exception_is_fail_safe_not_crash():
    p = MockProvider([RuntimeError("api down"), RuntimeError("api down")])
    res = analyze_frame(p, FRAME)
    assert not res.valid and res.parsed is None


def test_schema_violation_is_invalid():
    bad = GOOD.replace('"efficient"', '"turbo"')
    res = analyze_frame(MockProvider([bad, bad]), FRAME)
    assert not res.valid
