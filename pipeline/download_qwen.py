"""Download Qwen2.5-VL weights into QWEN_LOCAL_DIR.

Run once before switching VLM_BACKEND to "qwen":

    python3 pipeline/download_qwen.py

Requires ~16 GB free disk for the 7B model. Uses huggingface_hub, so you
can set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads, or export
HF_ENDPOINT=https://hf-mirror.com behind the GFW.
"""
from __future__ import annotations
import sys

from config import QWEN_MODEL_ID, QWEN_LOCAL_DIR


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub not installed. Run: pip install 'huggingface_hub[cli]'"
        )

    QWEN_LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {QWEN_MODEL_ID} -> {QWEN_LOCAL_DIR}")
    path = snapshot_download(
        repo_id=QWEN_MODEL_ID,
        local_dir=str(QWEN_LOCAL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
