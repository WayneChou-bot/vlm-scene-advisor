"""DecisionPolicy tests — includes the non-negotiable fail-safe regression tests."""
import pytest

from src.policy import Mode, decide
from src.schema import SceneAnalysis


def make(persons=None, person_in_zone=False, scene="clear", suggested="efficient"):
    return SceneAnalysis.model_validate(
        {
            "persons": persons or [],
            "person_in_zone": person_in_zone,
            "obstructions": [],
            "scene": scene,
            "reasoning": "r",
            "suggested_mode": suggested,
            "answer": "a",
        }
    )


def person(zone="far", conf=0.9, position="right"):
    return {"position": position, "zone": zone, "confidence": conf}


# ---- fail-safe: NEVER efficient on invalid / low-confidence (safety-critical, no regression) ----

def test_invalid_output_never_efficient():
    d = decide(None)
    assert d.mode == Mode.CAUTIOUS and d.fail_safe
    assert d.mode != Mode.EFFICIENT


def test_low_confidence_never_efficient_even_if_scene_clear():
    d = decide(make(persons=[person(zone="far", conf=0.3)], scene="clear"))
    assert d.mode == Mode.CAUTIOUS and d.fail_safe


def test_low_confidence_even_when_vlm_suggests_efficient():
    d = decide(make(persons=[person(conf=0.1)], suggested="efficient"))
    assert d.mode != Mode.EFFICIENT


@pytest.mark.parametrize("conf", [0.0, 0.3, 0.59])
def test_below_threshold_is_fail_safe(conf):
    d = decide(make(persons=[person(conf=conf)]))
    assert d.fail_safe and d.mode == Mode.CAUTIOUS


# ---- STOP branch ----

def test_person_in_zone_stops():
    d = decide(make(persons=[person(zone="in_zone", conf=0.95)], person_in_zone=True, scene="occupied"))
    assert d.mode == Mode.STOP


def test_blocked_scene_stops():
    d = decide(make(scene="blocked"))
    assert d.mode == Mode.STOP


def test_person_in_zone_flag_alone_stops():
    d = decide(make(person_in_zone=True))
    assert d.mode == Mode.STOP


# ---- CAUTIOUS branch ----

def test_person_near_is_cautious():
    d = decide(make(persons=[person(zone="near", conf=0.9)], scene="occupied"))
    assert d.mode == Mode.CAUTIOUS and not d.fail_safe


def test_occupied_scene_is_cautious():
    d = decide(make(scene="occupied"))
    assert d.mode == Mode.CAUTIOUS


# ---- EFFICIENT: only clear + no persons ----

def test_clear_and_empty_is_efficient():
    d = decide(make(scene="clear"))
    assert d.mode == Mode.EFFICIENT and not d.fail_safe


def test_far_person_high_conf_clear_scene_is_not_efficient():
    # SPEC rule 4 requires "no persons"; a far person falls through to default cautious.
    d = decide(make(persons=[person(zone="far", conf=0.9)], scene="clear"))
    assert d.mode == Mode.CAUTIOUS and d.rule == "default_cautious"


# ---- policy has final say over VLM suggestion ----

def test_policy_overrides_vlm_suggestion():
    d = decide(make(person_in_zone=True, suggested="efficient"))
    assert d.mode == Mode.STOP


def test_loading_scene_defaults_cautious():
    d = decide(make(scene="loading"))
    assert d.mode == Mode.CAUTIOUS
