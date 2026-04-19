"""Step 3 - Call GPT-4o with Structured Outputs for both sub-tasks.

For each video we make:
  - 8 calls of sub-task A (per-frame semantic + cross-view), 12 images each
  - 3 calls of sub-task B (per-view temporal), 10 images each, only for
    front-facing views (FRONT_LEFT, FRONT, FRONT_RIGHT).

Total per video: 11 calls, 156 images.

Raw outputs are dumped to outputs/vlm_raw/{video_id}.json. Failures are
logged to outputs/errors.log; the failed call is skipped (not retried in
this first pass).
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path

from config import (
    FRAMES_DIR,
    VLM_RAW_DIR,
    OUTPUT_DIR,
    NUM_FRAMES,
    VIEWS,
    FRONT_VIEWS,
    FIRST_RUN_IDS,
)
from prompts import (
    build_semantic_prompt,
    SEMANTIC_SCHEMA,
    build_temporal_prompt,
    TEMPORAL_SCHEMA,
)
from vlm_backend import call_vlm


def _semantic_images(video_dir: Path, frame_idx: int) -> list[Path]:
    paths = []
    for view in VIEWS:
        paths.append(video_dir / f"f{frame_idx:02d}_{view}_gt.jpg")
        paths.append(video_dir / f"f{frame_idx:02d}_{view}_gen.jpg")
    return paths


def _temporal_images(video_dir: Path, view: str) -> list[Path]:
    paths = [video_dir / f"f00_{view}_gt.jpg"]
    for f in range(NUM_FRAMES):
        paths.append(video_dir / f"f{f:02d}_{view}_gen.jpg")
    paths.append(video_dir / f"f{NUM_FRAMES - 1:02d}_{view}_gt.jpg")
    return paths


def run_video(video_id: str, err_log) -> dict:
    video_dir = FRAMES_DIR / video_id
    result = {"video_id": video_id, "semantic": [], "temporal": []}

    # Sub-task A
    for f in range(NUM_FRAMES):
        try:
            out = call_vlm(
                build_semantic_prompt(f),
                _semantic_images(video_dir, f),
                SEMANTIC_SCHEMA,
            )
            out["frame_idx"] = f
            result["semantic"].append(out)
            print(f"  [{video_id}] semantic frame {f} ok")
        except Exception as e:
            err_log.write(f"{video_id} semantic f{f}: {e}\n{traceback.format_exc()}\n")
            print(f"  [{video_id}] semantic frame {f} FAILED: {e}")

    # Sub-task B
    for view in FRONT_VIEWS:
        try:
            out = call_vlm(
                build_temporal_prompt(view),
                _temporal_images(video_dir, view),
                TEMPORAL_SCHEMA,
            )
            out["view"] = view
            result["temporal"].append(out)
            print(f"  [{video_id}] temporal {view} ok")
        except Exception as e:
            err_log.write(f"{video_id} temporal {view}: {e}\n{traceback.format_exc()}\n")
            print(f"  [{video_id}] temporal {view} FAILED: {e}")

    return result


def main(video_ids: list[str] | None = None) -> None:
    ids = video_ids or FIRST_RUN_IDS
    VLM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    err_path = OUTPUT_DIR / "errors.log"
    with err_path.open("a") as err_log:
        err_log.write(f"\n=== run @ {time.strftime('%F %T')} ===\n")
        for vid in ids:
            t0 = time.time()
            print(f"[{vid}] starting")
            result = run_video(vid, err_log)
            (VLM_RAW_DIR / f"{vid}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False)
            )
            print(f"[{vid}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None)
