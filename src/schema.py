"""Pydantic contract for VLM structured output. SPEC §5."""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, ConfigDict

Position = Literal["left", "right", "center", "behind"]
Zone = Literal["in_zone", "near", "far"]
Scene = Literal["clear", "occupied", "loading", "blocked"]
ModeStr = Literal["efficient", "cautious", "stop"]


class PersonObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    position: Position
    zone: Zone
    confidence: float = Field(ge=0.0, le=1.0)


class SceneAnalysis(BaseModel):
    """The only shape we accept from a VLM. Anything else is invalid → fail-safe."""

    model_config = ConfigDict(extra="ignore")

    persons: List[PersonObservation] = Field(default_factory=list)
    person_in_zone: bool
    obstructions: List[str] = Field(default_factory=list)
    scene: Scene
    reasoning: str
    suggested_mode: ModeStr
    answer: str
