"""M2 programmatic verification: JSONL <-> frame correspondence + zone color + disclaimer.

Checks, per sampled telemetry record:
  1. the frame-annotation sidecar carries that record's thinking/answer/mode for all
     frames until the next sample (text<->frame mapping is correct);
  2. in the rendered MP4, the zone pixel tint matches the recorded mode's color;
  3. the disclaimer strip is present (darkened strip + bright text pixels) on every
     sampled frame.

Usage: python scripts/verify_overlay.py out/annotated.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ZONE_CENTER  # noqa: E402
from src.overlay import MODE_COLOR  # noqa: E402
from src.policy import Mode  # noqa: E402
from src.telemetry import read_jsonl  # noqa: E402


def main(video_path: str) -> int:
    stem = str(Path(video_path).with_suffix(""))
    telemetry = read_jsonl(stem + "_telemetry.jsonl")
    annotations = {r["frame_index"]: r for r in read_jsonl(stem + "_frame_annotations.jsonl")}
    n_frames = len(annotations)

    failures = []

    # --- 1. text<->frame mapping ---
    for i, rec in enumerate(telemetry):
        start = rec["frame_index"]
        end = telemetry[i + 1]["frame_index"] if i + 1 < len(telemetry) else n_frames
        expected_mode = rec["decision"]["mode"]
        expected_thinking = (
            rec["parsed"]["reasoning"] if rec["valid"]
            else "VLM output invalid or low confidence - fail-safe engaged."
        )
        for f in range(start, end):
            ann = annotations[f]
            if ann["mode"] != expected_mode or ann["thinking"] != expected_thinking:
                failures.append(f"frame {f}: annotation mismatch vs telemetry record @frame {start}")
                break

    # --- 2 & 3. pixel checks on rendered video at each sampled frame ---
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    zx, zy = int(ZONE_CENTER[0] * w), int(ZONE_CENTER[1] * h)

    for rec in telemetry:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame_index"])
        ok, frame = cap.read()
        if not ok:
            failures.append(f"frame {rec['frame_index']}: cannot read from video")
            continue
        px = frame[zy, zx].astype(int)
        expected = np.array(MODE_COLOR[Mode(rec["decision"]["mode"])])
        if int(np.argmax(px)) != int(np.argmax(expected)):
            failures.append(
                f"frame {rec['frame_index']}: zone tint {px.tolist()} does not match mode "
                f"{rec['decision']['mode']}"
            )
        strip = frame[h - 26 :, :, :]
        if not (strip.mean() < 130 and (strip > 180).any()):
            failures.append(f"frame {rec['frame_index']}: disclaimer strip missing")
    cap.release()

    print(f"telemetry records: {len(telemetry)} | frames: {n_frames}")
    if failures:
        print("FAIL")
        for f in failures[:20]:
            print(" -", f)
        return 1
    print("PASS: text<->frame mapping, zone color per mode, disclaimer present on all sampled frames")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "out/annotated.mp4"))
