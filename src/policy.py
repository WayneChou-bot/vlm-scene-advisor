"""Deterministic DecisionPolicy with fail-safe override. SPEC §5. TDD target.

Rule order (SPEC-literal):
  1. invalid output OR (persons present AND max confidence < CONF_THRESHOLD) → CAUTIOUS
  2. person_in_zone OR scene == "blocked"                                    → STOP
  3. any person.zone == "near" OR scene == "occupied"                        → CAUTIOUS
  4. scene == "clear" AND no persons                                         → EFFICIENT
  5. otherwise                                                               → CAUTIOUS

Invariant (regression-tested): low-confidence / invalid output NEVER yields EFFICIENT.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.schema import SceneAnalysis

CONF_THRESHOLD = 0.6


class Mode(str, Enum):
    EFFICIENT = "efficient"
    CAUTIOUS = "cautious"
    STOP = "stop"


@dataclass(frozen=True)
class Decision:
    mode: Mode
    rule: str          # which rule fired (for telemetry / explainability)
    fail_safe: bool    # True when decision came from the fail-safe branch


def decide(analysis: Optional[SceneAnalysis], conf_threshold: float = CONF_THRESHOLD) -> Decision:
    # Rule 1 — fail-safe: invalid output
    if analysis is None:
        return Decision(Mode.CAUTIOUS, "fail_safe_invalid_output", True)

    # Rule 1 — fail-safe: low confidence (only meaningful when persons reported)
    if analysis.persons:
        if max(p.confidence for p in analysis.persons) < conf_threshold:
            return Decision(Mode.CAUTIOUS, "fail_safe_low_confidence", True)

    # Rule 2 — hard stop
    if analysis.person_in_zone or analysis.scene == "blocked":
        return Decision(Mode.STOP, "stop_person_in_zone_or_blocked", False)

    # Rule 3 — caution
    if any(p.zone == "near" for p in analysis.persons) or analysis.scene == "occupied":
        return Decision(Mode.CAUTIOUS, "cautious_near_or_occupied", False)

    # Rule 4 — efficient only when provably boring
    if analysis.scene == "clear" and not analysis.persons:
        return Decision(Mode.EFFICIENT, "efficient_clear_no_person", False)

    # Rule 5 — default conservative
    return Decision(Mode.CAUTIOUS, "default_cautious", False)
