# VLM Scene Advisor

**Explainable scene reasoning for an autonomous warehouse forklift — a concept demo.**

A fixed CCTV view watches an AGV forklift's operating zone. Once per second, a
vision-language model reasons about the frame, and its reasoning is rendered onto the
video as a *Thinking / Answer* HUD. When a worker walks into the zone, you can read
exactly why the system escalates: **Efficient → Cautious → Stop** (halt the forklift,
sound the zone alarm), then recovers once the zone is clear.

> ⚠️ **Concept demo — not a functional safety system.** A single VLM is not
> safety-rated; false negatives are unacceptable in real industrial safety. This
> project demonstrates *explainable operational advisory* on top of (assumed)
> certified safety systems. No detection-accuracy claims are made.

## Demo

🎬 **[Watch the annotated demo video](https://github.com/WayneChou-bot/vlm-scene-advisor/releases/latest)** —
`annotated_gemini.mp4`: the HUD text is real Gemini 2.5 Flash reasoning, with live
per-frame latency shown on screen (not scripted placeholders).

## How it works

```
CCTV frame ──► cadence sampler (~1 fps) ──► VLM provider (Gemini / stub)
                                                    │  raw text
                                                    ▼
                                    parse → pydantic schema validation
                                                    │  structured scene analysis
                                                    ▼
                       ┌─────────────────────────────────────────────┐
                       │  DecisionPolicy (deterministic, unit-tested)│
                       │  fail-safe: invalid / low-confidence output │
                       │  can NEVER yield "efficient"                │
                       │  + 2s release delay (conservative-only)     │
                       └──────────────────┬──────────────────────────┘
                                          ▼
                 HUD overlay (Thinking/Answer, zone color, disclaimer)
                 + JSONL telemetry (raw, parsed, decision, latency)
                 + annotated MP4
```

Two design principles:

1. **The VLM only suggests — the policy decides.** The policy layer is pure Python
   with regression tests, including the non-negotiable rule that low-confidence or
   invalid model output can never unlock efficient mode. Fail closed.
2. **Perception should perceive; policy should decide.** Safety margins belong in
   the deterministic layer, not in the model's judgment (see the prompt-iteration
   notes below).

## Prompt iterations — what actually happened

| Run | Change | Result |
|-----|--------|--------|
| v1 | baseline, enums undefined, thinking on | person detection solid, but stored pallets were read as a "blocked" zone → false stops on empty frames; ~9.6 s median latency |
| v2 | scene-label definitions + thinking off | empty-frame noise gone; latency down to ~3.2 s; but a "be conservative" instruction leaked into zone perception and inflated *near* into *in_zone* |
| v3 | zone anchored to visual landmarks, feet-based judgment, "report accurately — the policy layer owns the safety margin" | full green→amber→red→amber arc; red fires ~2 s earlier than scripted truth (conservative side) |

Constant across all runs: 16/16 schema-valid JSON, ≤1 retry, and **zero
unsafe-direction errors** — the model never reported "clear" while a person was
visible. Single synthetic scene, N=16 per run: illustrative, **not** statistics.

## Quickstart

Zero-API development loop (stub provider fakes the VLM):

```bash
pip install -r requirements.txt
python scripts/make_sample_clip.py data/sample_clip.mp4    # synthetic 2D test clip
python -m src.pipeline --input data/sample_clip.mp4 --provider stub --out out/annotated.mp4
python scripts/verify_overlay.py out/annotated.mp4         # programmatic HUD verification
pytest                                                     # 36 tests incl. fail-safe regressions
```

Live VLM run (bring your own key):

```bash
cp .env.example .env       # add GEMINI_API_KEY — never commit .env
python -m src.pipeline --input <your_clip>.mp4 --provider gemini --max-samples 16
```

Windows + conda one-shot: `scripts\run_gemini_windows.bat`.

## The 3D scene

The demo footage is rendered from a procedurally-built Blender scene
(`scripts/build_scene_3d_v2.py`, packed copy in `assets/warehouse_scene_v2.blend`):
corrugated-steel warehouse, dock doorways, a rigged worker with walk/idle animations,
and a forklift that halts when the alert fires. All external assets are CC0
(poly.pizza models, Poly Haven PBR textures + HDRI) — see `assets/LICENSES.md`.
Render it yourself: open the .blend, press Ctrl+F12 (settings are pre-configured),
then feed the MP4 to the pipeline.

## Repository layout

```
src/        pipeline, providers (gemini/stub), reasoner, policy, overlay, telemetry
scripts/    Blender scene builder, clip generator, HUD verification, Windows runner
tests/      pytest — schema contract, policy fail-safe regressions, reasoner (mock VLM), overlay
assets/     packed Blender scene + asset licenses
```

## Honesty notes

Built spec-first with an explicit verification protocol: nothing is marked done
without stating how it was verified, all misjudgments observed in testing landed on
the conservative side, and detection accuracy remains **unverified** (no labeled
eval) — which is exactly why this README contains no accuracy numbers.
