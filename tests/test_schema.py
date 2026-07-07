import pytest
from pydantic import ValidationError

from src.schema import PersonObservation, SceneAnalysis

VALID = {
    "persons": [{"position": "right", "zone": "near", "confidence": 0.87}],
    "person_in_zone": False,
    "obstructions": [],
    "scene": "occupied",
    "reasoning": "One person approaching from the right.",
    "suggested_mode": "cautious",
    "answer": "Switch to Cautious Mode.",
}


def test_valid_full_payload():
    a = SceneAnalysis.model_validate(VALID)
    assert a.persons[0].zone == "near"
    assert a.suggested_mode == "cautious"


def test_defaults_for_optional_lists():
    a = SceneAnalysis.model_validate(
        {
            "person_in_zone": False,
            "scene": "clear",
            "reasoning": "r",
            "suggested_mode": "efficient",
            "answer": "a",
        }
    )
    assert a.persons == [] and a.obstructions == []


@pytest.mark.parametrize("conf", [-0.1, 1.1, 2.0])
def test_confidence_out_of_bounds_rejected(conf):
    with pytest.raises(ValidationError):
        PersonObservation.model_validate({"position": "left", "zone": "far", "confidence": conf})


@pytest.mark.parametrize(
    "field,value",
    [
        ("scene", "party"),
        ("suggested_mode", "turbo"),
    ],
)
def test_illegal_enum_rejected(field, value):
    bad = dict(VALID)
    bad[field] = value
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(bad)


def test_missing_required_field_rejected():
    bad = dict(VALID)
    del bad["reasoning"]
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(bad)


def test_extra_fields_ignored_not_fatal():
    ok = dict(VALID)
    ok["hallucinated_extra"] = 123
    a = SceneAnalysis.model_validate(ok)
    assert not hasattr(a, "hallucinated_extra")
