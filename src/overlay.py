"""HUD overlay: Thinking/Answer panel + detection-zone visualization + disclaimer.

Visual target: the LinkedIn reference frame - dark translucent card, "Thinking:" and
"Answer:" labels, green glowing zone under the forklift. Colors follow decision mode.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import DISCLAIMER, ZONE_ANGLE, ZONE_AXES, ZONE_CENTER
from src.policy import Mode

FONT = cv2.FONT_HERSHEY_SIMPLEX

MODE_COLOR = {  # BGR
    Mode.EFFICIENT: (90, 210, 60),    # green
    Mode.CAUTIOUS: (30, 190, 250),    # amber
    Mode.STOP: (60, 60, 235),         # red
}
MODE_LABEL = {
    Mode.EFFICIENT: "EFFICIENT MODE",
    Mode.CAUTIOUS: "CAUTIOUS MODE",
    Mode.STOP: "STOP",
}


def _wrap(text: str, scale: float, thickness: int, max_w: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _rounded_rect(img, p1, p2, color, radius=14, alpha=0.72):
    """Draw a filled rounded rect blended onto img."""
    overlay = img.copy()
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_zone(frame: np.ndarray, mode: Mode) -> np.ndarray:
    h, w = frame.shape[:2]
    center = (int(ZONE_CENTER[0] * w), int(ZONE_CENTER[1] * h))
    axes = (int(ZONE_AXES[0] * w), int(ZONE_AXES[1] * h))
    color = MODE_COLOR[mode]

    glow = frame.copy()
    # soft outer glow
    cv2.ellipse(glow, center, (int(axes[0] * 1.15), int(axes[1] * 1.15)), ZONE_ANGLE, 0, 360, color, -1)
    cv2.addWeighted(glow, 0.18, frame, 0.82, 0, frame)
    fill = frame.copy()
    cv2.ellipse(fill, center, axes, ZONE_ANGLE, 0, 360, color, -1)
    cv2.addWeighted(fill, 0.38, frame, 0.62, 0, frame)
    cv2.ellipse(frame, center, axes, ZONE_ANGLE, 0, 360, color, 2, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    thinking: str,
    answer: str,
    mode: Mode,
    t: float = 0.0,
    latency_ms: Optional[float] = None,
    provider: str = "",
    fail_safe: bool = False,
) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = draw_zone(frame, mode)

    # ---- Thinking / Answer card (top-right) ----
    card_w = int(w * 0.40)
    pad, ls = 14, 20
    body_scale, body_th = 0.46, 1
    label_scale, label_th = 0.52, 2
    max_text_w = card_w - 2 * pad

    think_lines = _wrap(thinking, body_scale, body_th, max_text_w)
    ans_lines = _wrap(answer, body_scale, body_th, max_text_w)
    card_h = pad + 24 + len(think_lines) * ls + 16 + 24 + len(ans_lines) * ls + pad
    x1, y1 = w - card_w - 18, 18
    _rounded_rect(frame, (x1, y1), (x1 + card_w, y1 + card_h), (28, 24, 20))

    y = y1 + pad + 16
    cv2.putText(frame, "Thinking:", (x1 + pad, y), FONT, label_scale, (255, 255, 255), label_th, cv2.LINE_AA)
    y += 22
    for line in think_lines:
        cv2.putText(frame, line, (x1 + pad, y), FONT, body_scale, (225, 225, 225), body_th, cv2.LINE_AA)
        y += ls
    y += 14
    cv2.putText(frame, "Answer:", (x1 + pad, y), FONT, label_scale, (255, 255, 255), label_th, cv2.LINE_AA)
    y += 22
    for line in ans_lines:
        cv2.putText(frame, line, (x1 + pad, y), FONT, body_scale, (225, 225, 225), body_th, cv2.LINE_AA)
        y += ls

    # ---- mode badge (top-left) ----
    label = MODE_LABEL[mode] + (" (fail-safe)" if fail_safe else "")
    (tw, th_), _ = cv2.getTextSize(label, FONT, 0.55, 2)
    _rounded_rect(frame, (18, 18), (18 + tw + 46, 18 + th_ + 22), (28, 24, 20), radius=10)
    cv2.circle(frame, (36, 18 + (th_ + 22) // 2), 7, MODE_COLOR[mode], -1, cv2.LINE_AA)
    cv2.putText(frame, label, (50, 18 + th_ + 8), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- meta line (bottom-left, above disclaimer) ----
    meta = f"t={t:5.1f}s  provider={provider}"
    if latency_ms is not None:
        meta += f"  latency={latency_ms:.0f}ms"
    (mw, mh), _ = cv2.getTextSize(meta, FONT, 0.42, 1)
    _rounded_rect(frame, (12, h - 38 - mh - 6), (24 + mw, h - 32), (28, 24, 20), radius=8, alpha=0.6)
    cv2.putText(frame, meta, (18, h - 38), FONT, 0.42, (225, 225, 225), 1, cv2.LINE_AA)

    # ---- disclaimer watermark (mandatory, bottom strip) ----
    strip = frame.copy()
    cv2.rectangle(strip, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(strip, 0.55, frame, 0.45, 0, frame)
    (dw, _), _ = cv2.getTextSize(DISCLAIMER, FONT, 0.45, 1)
    cv2.putText(frame, DISCLAIMER, ((w - dw) // 2, h - 8), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return frame
