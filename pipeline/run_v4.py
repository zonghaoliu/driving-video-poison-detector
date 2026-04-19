"""V4 runner - two-stage auto-CoT, one call per video.

Stage 1 (once): send Prompt 1 (text only) to GPT-4o. The returned text
is the global "evaluation steps" reused across all videos. Cached to
outputs/exp_v4/eval_steps_v4.txt.

Stage 2 (per video): multi-turn conversation
    [user]      Prompt 1
    [assistant] <stage-1 text>
    [user]      Prompt 2 + 8 full-resolution video frames
GPT-4o returns the final video-level JSON. Parsed and written directly
to outputs/exp_v4/vlm_raw/{video_id}.json and aggregated into
outputs/exp_v4/vlm_clean.json.

Each video is analyzed as 8 full (2688x784) frames extracted straight
from the mp4, preserving the three-layer GT / 3D-box / Gen vertical
layout the V4 prompt was written for. One API call per video, 11 total.
"""
from __future__ import annotations
import base64
import json
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path

import cv2
from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    MODEL,
    VIDEO_DIR,
    OUTPUT_DIR,
    NUM_FRAMES,
    FIRST_RUN_IDS,
)
from prompt_v4 import PROMPT_1, PROMPT_2


EXP_DIR = OUTPUT_DIR / "exp_v4"
RAW_DIR = EXP_DIR / "vlm_raw"
STEPS_FILE = EXP_DIR / "eval_steps_v4.txt"
CLEAN_JSON = EXP_DIR / "vlm_clean.json"


def _extract_8_frames(mp4_path: Path) -> list[bytes]:
    """Return 8 JPEG-encoded full-resolution frames from the mp4."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    frames = []
    for _ in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok2:
            raise RuntimeError("jpeg encode failed")
        frames.append(buf.tobytes())
    cap.release()
    if len(frames) != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} frames, got {len(frames)}")
    return frames


def _img_msg(jpg_bytes: bytes) -> dict:
    b64 = base64.b64encode(jpg_bytes).decode()
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": "high",  # V4 wants the full video, keep quality up
        },
    }


def stage_1_get_eval_steps(client: OpenAI) -> str:
    if STEPS_FILE.exists():
        return STEPS_FILE.read_text()
    resp = client.with_options(timeout=90.0).chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": PROMPT_1}],
    )
    text = resp.choices[0].message.content.strip()
    STEPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STEPS_FILE.write_text(text)
    print(f"[stage1] cached evaluation steps -> {STEPS_FILE}")
    return text


def stage_2_eval_video(client: OpenAI, eval_steps: str, video_id: str) -> dict:
    mp4_path = VIDEO_DIR / f"{video_id}.mp4"
    frames = _extract_8_frames(mp4_path)

    user2_content = [{"type": "text", "text": PROMPT_2}]
    for jpg in frames:
        user2_content.append(_img_msg(jpg))

    resp = client.with_options(timeout=120.0).chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": PROMPT_1},
            {"role": "assistant", "content": eval_steps},
            {"role": "user", "content": user2_content},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _normalize(raw: dict, video_id: str) -> dict:
    """V4 prompt shows 'video_id': '01.mp4' and 'attack_level': 'Decision'.
    Normalize to the same schema the human annotations use."""
    scores = raw.get("scores", {}) or {}

    def _f(k):
        try:
            return float(scores.get(k, 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    sem, log, dec = _f("semantic"), _f("logical"), _f("decision")
    attack = str(raw.get("attack_level", "")).strip().lower()
    if attack not in ("semantic", "logical", "decision"):
        attack = max(
            [("semantic", sem), ("logical", log), ("decision", dec)],
            key=lambda x: x[1],
        )[0]

    is_poisoned = bool(raw.get("is_poisoned"))
    if not isinstance(raw.get("is_poisoned"), bool):
        is_poisoned = max(sem, log, dec) > 0.5

    ordered = sorted([sem, log, dec], reverse=True)
    fs_raw = raw.get("final_score")
    try:
        final_score = float(fs_raw) if fs_raw is not None else round(
            0.5 * ordered[0] + 0.3 * ordered[1] + 0.2 * ordered[2], 3
        )
    except (TypeError, ValueError):
        final_score = round(0.5 * ordered[0] + 0.3 * ordered[1] + 0.2 * ordered[2], 3)

    return {
        "video_id": video_id,
        "is_poisoned": is_poisoned,
        "attack_level": attack,
        "scores": {"semantic": sem, "logical": log, "decision": dec},
        "final_score": final_score,
        "reasoning": raw.get("reasoning", ""),
    }


def main(video_ids: list[str] | None = None) -> None:
    ids = video_ids or FIRST_RUN_IDS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY)

    eval_steps = stage_1_get_eval_steps(client)
    print(f"[stage1] eval_steps length = {len(eval_steps)} chars")

    err_path = OUTPUT_DIR / "errors.log"
    aggregated = []
    with err_path.open("a") as err_log:
        err_log.write(f"\n=== V4 run @ {time.strftime('%F %T')} ===\n")
        for vid in ids:
            t0 = time.time()
            try:
                raw = stage_2_eval_video(client, eval_steps, vid)
                (RAW_DIR / f"{vid}.json").write_text(
                    json.dumps(raw, indent=2, ensure_ascii=False)
                )
                norm = _normalize(raw, vid)
                aggregated.append(norm)
                print(f"[{vid}] ok in {time.time() - t0:.1f}s -> "
                      f"sem={norm['scores']['semantic']} "
                      f"log={norm['scores']['logical']} "
                      f"dec={norm['scores']['decision']} "
                      f"attack={norm['attack_level']}")
            except Exception as e:
                err_log.write(f"V4 {vid}: {e}\n{traceback.format_exc()}\n")
                print(f"[{vid}] FAILED: {e}")

    CLEAN_JSON.write_text(json.dumps(aggregated, indent=2, ensure_ascii=False))
    print(f"wrote {len(aggregated)} records -> {CLEAN_JSON}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None)
