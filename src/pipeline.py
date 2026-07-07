"""Pipeline: source -> cadence sampling -> VLM -> policy -> overlay -> annotated MP4 + JSONL.

Usage:
  python -m src.pipeline --input data/sample_clip.mp4 --provider stub --out out/annotated.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.frame_source import VideoFileSource
from src.overlay import draw_hud
from src.policy import Mode, decide
from src.reasoner import analyze_frame
from src.telemetry import JsonlLogger


def get_provider(name: str):
    if name == "stub":
        from src.providers.stub import StubProvider
        return StubProvider()
    if name == "gemini":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        from src.providers.gemini import GeminiProvider
        return GeminiProvider()
    raise ValueError(f"Unknown provider: {name}")


def run(
    input_path: str,
    provider_name: str = "stub",
    out_path: str = "out/annotated.mp4",
    jsonl_path: str = None,
    interval_s: float = 1.0,
    max_samples: int = 500,
    release_delay_s: float = 2.0,
) -> dict:
    provider = get_provider(provider_name)
    src = VideoFileSource(input_path)
    out_path = str(out_path)
    jsonl_path = jsonl_path or str(Path(out_path).with_suffix("")) + "_telemetry.jsonl"
    ann_path = str(Path(out_path).with_suffix("")) + "_frame_annotations.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (src.width, src.height)
    )

    # state held between samples (cadence sampling — SPEC §6)
    thinking = "Initializing scene reasoning..."
    answer = "Hold Cautious Mode."
    mode, fail_safe, latency_ms = Mode.CAUTIOUS, False, None
    next_sample_t = 0.0
    n_samples = 0
    SEVERITY = {Mode.EFFICIENT: 0, Mode.CAUTIOUS: 1, Mode.STOP: 2}
    last_restrictive_t = -1e9  # release delay: only relax after clear for release_delay_s

    with JsonlLogger(jsonl_path) as tlog, JsonlLogger(ann_path) as alog:
        for idx, t, frame in src.frames():
            if t >= next_sample_t and n_samples < max_samples:
                res = analyze_frame(provider, frame)
                decision = decide(res.parsed)
                new_mode = decision.mode
                if SEVERITY[new_mode] >= SEVERITY[mode]:
                    last_restrictive_t = t          # holding or escalating: reset timer
                    mode = new_mode
                elif t - last_restrictive_t >= release_delay_s:
                    mode = new_mode                 # relax only after a stable clear period
                # else: hold the more restrictive mode (deterministic, conservative-only)
                fail_safe, latency_ms = decision.fail_safe, res.latency_ms
                if res.valid:
                    thinking, answer = res.parsed.reasoning, res.parsed.answer
                else:
                    thinking = "VLM output invalid or low confidence - fail-safe engaged."
                    answer = "Hold Cautious Mode (fail-safe)."
                tlog.write(
                    {
                        "frame_index": idx,
                        "t": round(t, 3),
                        "provider": provider.name,
                        "model": provider.model,
                        "latency_ms": round(res.latency_ms, 1),
                        "valid": res.valid,
                        "retries": res.retries,
                        "error": res.error,
                        "raw": res.raw,
                        "parsed": res.parsed.model_dump() if res.parsed else None,
                        "decision": {
                            "mode": mode.value,
                            "raw_mode": decision.mode.value,
                            "rule": decision.rule,
                            "fail_safe": decision.fail_safe,
                        },
                    }
                )
                next_sample_t = t + interval_s
                n_samples += 1

            alog.write(
                {"frame_index": idx, "t": round(t, 3), "mode": mode.value,
                 "thinking": thinking, "answer": answer}
            )
            writer.write(
                draw_hud(frame, thinking, answer, mode, t=t,
                         latency_ms=latency_ms, provider=provider.name, fail_safe=fail_safe)
            )

    writer.release()
    src.release()
    return {"out": out_path, "jsonl": jsonl_path, "annotations": ann_path, "samples": n_samples}


def main():
    ap = argparse.ArgumentParser(description="VLM Scene Advisor pipeline")
    ap.add_argument("--input", required=True)
    ap.add_argument("--provider", default="stub", choices=["stub", "gemini"])
    ap.add_argument("--out", default="out/annotated.mp4")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--interval-s", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=500, help="cost guardrail")
    ap.add_argument("--release-delay-s", type=float, default=2.0,
                    help="mode may only relax after the zone has been clear this long")
    args = ap.parse_args()
    info = run(args.input, args.provider, args.out, args.jsonl, args.interval_s,
               args.max_samples, args.release_delay_s)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
