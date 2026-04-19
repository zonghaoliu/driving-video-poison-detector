"""Central config for the auto-annotation pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "mixed_group_12"
OUTPUT_DIR = ROOT / "outputs"
FRAMES_DIR = OUTPUT_DIR / "frames"
# EXPERIMENT switch: "baseline" | "exp_A_anchors" | ...
EXPERIMENT = "exp_criteria_2"
VLM_RAW_DIR = OUTPUT_DIR / EXPERIMENT / "vlm_raw"
CLEAN_JSON = OUTPUT_DIR / EXPERIMENT / "vlm_clean.json"
REPORT_MD = OUTPUT_DIR / EXPERIMENT / "eval_report.md"
DEBUG_DIR = OUTPUT_DIR / "debug"
HUMAN_JSON = ROOT / "merge_tools" / "merged_output.json"
EVAL_STEPS_FILE = ROOT / "pipeline" / "eval_steps.json"

# Layout: 2688x784, 8 frames, 6 views (cols), 3 layers (rows top->bot)
NUM_FRAMES = 8
NUM_VIEWS = 6
NUM_LAYERS = 3
# nuScenes-style view order, left->right of the 6 columns
VIEWS = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"]
# Top -> bottom rows (per user description)
LAYERS = ["gt", "box3d", "gen"]

# Temporal sub-task uses only the front-facing views
FRONT_VIEWS = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"]

# VLM backend selector: "qwen" (local) | "openai" (cloud)
VLM_BACKEND = "qwen"

# OpenAI (kept for future comparison; fill OPENAI_API_KEY when switching back)
OPENAI_API_KEY = ""
MODEL = "gpt-4o"

# Qwen (local, 4090-friendly). 7B is bf16 ~15GB; drop to 3B if VRAM is tight.
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_LOCAL_DIR = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
QWEN_MAX_NEW_TOKENS = 1024
# Per-image pixel budget (Qwen2.5-VL uses 28x28 patches). These caps keep
# peak VRAM manageable when a single call feeds 10-12 sub-images at once.
QWEN_MIN_PIXELS = 128 * 28 * 28   # ~100 k pixels
QWEN_MAX_PIXELS = 512 * 28 * 28   # ~400 k pixels

# Discrete score grid (compatible with future logprob-weighted scoring)
SCORE_ENUM = [f"{i/10:.1f}" for i in range(11)]  # "0.0" .. "1.0"

# First-run video range (inclusive on both ends)
FIRST_RUN_IDS = [f"{i:02d}" for i in range(10)]
