"""FrameSource abstraction (B layer input). Video file or webcam."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import cv2
import numpy as np

Frame = Tuple[int, float, np.ndarray]  # (frame_index, timestamp_s, bgr_frame)


class FrameSource(ABC):
    fps: float = 30.0

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        ...

    def release(self) -> None:
        pass


class VideoFileSource(FrameSource):
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[Frame]:
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield i, i / self.fps, frame
            i += 1

    def release(self) -> None:
        self.cap.release()


class WebcamSource(FrameSource):
    def __init__(self, index: int = 0, fps: float = 30.0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam index {index}")
        self.fps = fps

    def frames(self) -> Iterator[Frame]:
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield i, i / self.fps, frame
            i += 1

    def release(self) -> None:
        self.cap.release()
