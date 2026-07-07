"""Generate a synthetic warehouse sample clip (zero licensing risk, zero API).

Layout loosely mirrors the reference frame: doorway with a generic autonomous
forklift, boxes on the right, hazard stripes, gray floor. A worker (orange vest)
walks in from the right, enters the operating zone, then leaves.

Timeline (16 s @ 24 fps):
  0-5 s   empty            -> expect EFFICIENT
  5-9.5 s person approaches -> expect CAUTIOUS (near)
  9.5-12 s person in zone   -> expect STOP
  12-16 s person leaves     -> CAUTIOUS then EFFICIENT
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ZONE_CENTER  # noqa: E402

W, H, FPS, DUR = 960, 540, 24, 16.0


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


def draw_scene(t: float) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)

    # walls (light industrial panels) and floor
    img[:] = (172, 168, 160)
    cv2.rectangle(img, (0, 0), (W, int(H * 0.52)), (196, 192, 186), -1)
    for x in range(0, W, 60):  # wall panel seams
        cv2.line(img, (x, 0), (x, int(H * 0.52)), (178, 174, 168), 1)
    cv2.rectangle(img, (0, int(H * 0.30)), (W, int(H * 0.34)), (150, 90, 40), -1)  # blue stripe
    cv2.rectangle(img, (0, int(H * 0.52)), (W, H), (120, 118, 114), -1)  # floor
    cv2.rectangle(img, (0, int(H * 0.52)), (W, int(H * 0.55)), (100, 98, 94), -1)

    # doorway (dark opening) behind forklift
    dx1, dx2 = int(W * 0.30), int(W * 0.52)
    cv2.rectangle(img, (dx1, int(H * 0.10)), (dx2, int(H * 0.53)), (70, 66, 62), -1)
    cv2.rectangle(img, (dx1 - 8, int(H * 0.08)), (dx2 + 8, int(H * 0.10)), (140, 136, 130), -1)

    # second doorway with boxes (right)
    bx1, bx2 = int(W * 0.66), int(W * 0.86)
    cv2.rectangle(img, (bx1, int(H * 0.14)), (bx2, int(H * 0.53)), (88, 84, 80), -1)
    for i, (bw, bh) in enumerate([(70, 46), (64, 40), (58, 36)]):
        y2 = int(H * 0.53) - i * (bh + 2)
        cv2.rectangle(img, (bx1 + 20, y2 - bh), (bx1 + 20 + bw, y2), (115, 150, 175), -1)  # low-sat cardboard
        cv2.rectangle(img, (bx1 + 20, y2 - bh), (bx1 + 20 + bw, y2), (85, 115, 140), 1)

    # hazard stripes on floor edge
    for x in range(0, W, 40):
        pts = np.array([[x, int(H * 0.60)], [x + 20, int(H * 0.60)],
                        [x + 12, int(H * 0.63)], [x - 8, int(H * 0.63)]])
        cv2.fillPoly(img, [pts], (0, 225, 225))  # pure yellow (hue 30, outside stub's vest range)
        pts2 = pts + [20, 0]
        cv2.fillPoly(img, [pts2], (30, 30, 30))

    # ---- generic autonomous forklift at doorway (no brand) ----
    fx, fy = int(ZONE_CENTER[0] * W), int(H * 0.545)
    body_w, body_h = 88, 110
    # mast
    cv2.rectangle(img, (fx - 14, fy - body_h - 78), (fx + 14, fy - body_h + 10), (60, 58, 56), -1)
    cv2.rectangle(img, (fx - 6, fy - body_h - 78), (fx + 6, fy - body_h + 10), (90, 88, 86), -1)
    # body (dark red, generic)
    cv2.rectangle(img, (fx - body_w // 2, fy - body_h), (fx + body_w // 2, fy), (38, 34, 150), -1)
    cv2.rectangle(img, (fx - body_w // 2, fy - body_h), (fx + body_w // 2, fy), (30, 26, 110), 2)
    # sensor head + screen
    cv2.rectangle(img, (fx - 22, fy - body_h - 14), (fx + 22, fy - body_h), (40, 40, 40), -1)
    cv2.circle(img, (fx - 12, fy - body_h - 7), 4, (200, 200, 60), -1)
    cv2.circle(img, (fx + 12, fy - body_h - 7), 4, (200, 200, 60), -1)
    cv2.rectangle(img, (fx - 26, fy - 66), (fx + 26, fy - 30), (180, 170, 160), -1)  # panel
    cv2.rectangle(img, (fx - 20, fy - 60), (fx + 20, fy - 36), (90, 70, 40), -1)     # screen
    # wheels
    cv2.circle(img, (fx - 30, fy), 12, (25, 25, 25), -1)
    cv2.circle(img, (fx + 30, fy), 12, (25, 25, 25), -1)

    # ---- worker (orange vest) path ----
    # off-screen right -> walks left toward zone center -> stands in zone -> leaves right
    if 5.0 <= t < 9.5:
        px = lerp(W + 30, fx + 150, (t - 5.0) / 4.5)
    elif 9.5 <= t < 12.0:
        px = lerp(fx + 150, fx + 30, (t - 9.5) / 1.2)
    elif 12.0 <= t < 16.0:
        px = lerp(fx + 30, W + 40, (t - 12.0) / 3.5)
    else:
        px = None

    if px is not None and px < W + 25:
        py = int(ZONE_CENTER[1] * H) - 28  # vest bottom (= blob lowest point) lands on zone center
        px = int(px)
        bob = int(3 * np.sin(t * 9))
        # legs
        cv2.line(img, (px - 6, py + 26 + bob), (px - 10, py + 52), (60, 60, 70), 6)
        cv2.line(img, (px + 6, py + 26 + bob), (px + 10, py + 52), (60, 60, 70), 6)
        # torso = safety vest (orange, the stub's cue)
        cv2.rectangle(img, (px - 13, py - 8 + bob), (px + 13, py + 28 + bob), (30, 120, 255), -1)
        cv2.line(img, (px - 6, py - 6 + bob), (px - 6, py + 26 + bob), (160, 230, 240), 2)
        cv2.line(img, (px + 6, py - 6 + bob), (px + 6, py + 26 + bob), (160, 230, 240), 2)
        # arms
        cv2.line(img, (px - 13, py + bob), (px - 20, py + 20 + bob), (30, 120, 255), 5)
        cv2.line(img, (px + 13, py + bob), (px + 20, py + 20 + bob), (30, 120, 255), 5)
        # head + helmet
        cv2.circle(img, (px, py - 18 + bob), 9, (150, 170, 190), -1)
        cv2.ellipse(img, (px, py - 22 + bob), (10, 7), 0, 180, 360, (30, 110, 250), -1)  # orange helmet

    return img


def main(out="data/sample_clip.mp4"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(DUR * FPS)
    for i in range(n):
        vw.write(draw_scene(i / FPS))
    vw.release()
    print(f"wrote {out} ({n} frames @ {FPS}fps)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_clip.mp4")
