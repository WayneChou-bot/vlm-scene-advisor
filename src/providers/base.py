"""VLMProvider interface. A provider takes a BGR frame + prompt and returns raw text."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VLMProvider(ABC):
    name: str = "base"
    model: str = "n/a"

    @abstractmethod
    def analyze(self, frame_bgr: np.ndarray, prompt: str) -> str:
        """Return the raw model output (expected: JSON text, but never trusted)."""
        ...
