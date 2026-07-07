"""Programmatic overlay checks (not aesthetics): zone color tracks mode, disclaimer strip present."""
import cv2
import numpy as np

from src.config import ZONE_CENTER
from src.overlay import MODE_COLOR, draw_hud
from src.policy import Mode

H, W = 540, 960


def render(mode):
    frame = np.full((H, W, 3), 120, np.uint8)
    return draw_hud(frame.copy(), "thinking text", "answer text", mode, t=1.0, provider="test")


def zone_pixel(img):
    return img[int(ZONE_CENTER[1] * H), int(ZONE_CENTER[0] * W)].astype(int)


def dominant_channel(px):
    return int(np.argmax(px))


def test_frame_shape_unchanged():
    out = render(Mode.EFFICIENT)
    assert out.shape == (H, W, 3)


def test_zone_color_tracks_mode():
    for mode in Mode:
        px = zone_pixel(render(mode))
        assert dominant_channel(px) == dominant_channel(np.array(MODE_COLOR[mode])), (
            f"zone tint should follow {mode}"
        )


def test_zone_differs_between_modes():
    assert not np.array_equal(zone_pixel(render(Mode.EFFICIENT)), zone_pixel(render(Mode.STOP)))


def test_disclaimer_strip_present():
    base = np.full((H, W, 3), 120, np.uint8)
    out = draw_hud(base.copy(), "t", "a", Mode.EFFICIENT)
    strip = out[H - 26 :, :, :]
    # strip darkened vs base and contains bright text pixels
    assert strip.mean() < 120
    assert (strip > 180).any()
