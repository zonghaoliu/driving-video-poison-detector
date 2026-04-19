"""Split each driving video into 144 sub-images (8 frames * 6 views * 3 layers).

Layout (per frame, full image is 2688x784):
  - 6 columns (views), left to right
  - 3 rows (layers), top to bottom: GT, 3D box, generated
Output naming: outputs/frames/{video_id}/f{frame:02d}_{view}_{layer}.jpg
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2

from config import VIDEO_DIR, FRAMES_DIR, NUM_FRAMES, VIEWS, LAYERS, FIRST_RUN_IDS


def split_video(video_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    frame_idx = 0
    while frame_idx < NUM_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        col_w = w // len(VIEWS)
        row_h = h // len(LAYERS)
        for ci, view in enumerate(VIEWS):
            x0 = ci * col_w
            x1 = (ci + 1) * col_w if ci < len(VIEWS) - 1 else w
            for ri, layer in enumerate(LAYERS):
                y0 = ri * row_h
                y1 = (ri + 1) * row_h if ri < len(LAYERS) - 1 else h
                tile = frame[y0:y1, x0:x1]
                fname = f"f{frame_idx:02d}_{view}_{layer}.jpg"
                cv2.imwrite(str(out_dir / fname), tile, [cv2.IMWRITE_JPEG_QUALITY, 90])
                written += 1
        frame_idx += 1
    cap.release()
    return written


def main(video_ids: list[str] | None = None) -> None:
    ids = video_ids or FIRST_RUN_IDS
    for vid in ids:
        path = VIDEO_DIR / f"{vid}.mp4"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        out = FRAMES_DIR / vid
        n = split_video(path, out)
        print(f"[ok] {vid}: wrote {n} tiles -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None)
