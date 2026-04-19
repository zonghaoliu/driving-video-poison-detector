"""Step 0 - Auto-generate evaluation steps via GPT-4o (G-Eval style).

We ask the model, given the criteria, to write the concrete checklist a
human evaluator would follow for each of our two sub-tasks. The output is
saved to pipeline/eval_steps.json and reused across all videos so the
evaluation procedure stays deterministic.

Run once. Inspect the file. If unhappy, re-run.
"""
from __future__ import annotations
import json

from config import EVAL_STEPS_FILE
from criteria_2 import CRITERIA
from vlm_backend import call_vlm_text


SUBTASKS = {
    "per_frame_semantic_crossview": (
        "You will be shown all 6 surrounding camera views at one fixed "
        "time-step. For each camera you receive both the GT image and the "
        "generated image. You must judge:\n"
        "  (a) SEMANTIC errors: traffic-element identifiability problems "
        "      between GT and generated for the same camera.\n"
        "  (b) CROSS-VIEW consistency: whether the same physical object "
        "      looks consistent across adjacent cameras at this time-step. "
        "      Cross-view inconsistencies are reported under the LOGICAL "
        "      umbrella later, but you focus on observing them here."
    ),
    "per_view_temporal": (
        "You will be shown 8 consecutive generated frames from a single "
        "front-facing camera, plus the GT for the first and last frame "
        "as anchors. You must judge:\n"
        "  (a) LOGICAL errors: physical/temporal coherence across the 8 "
        "      generated frames (disappearances, jitter, impossible motion).\n"
        "  (b) DECISION errors: unsafe or illegal ego-vehicle behavior "
        "      visible from this camera (e.g. driving onto a sidewalk, "
        "      speeding toward a pedestrian)."
    ),
}


def build_prompt(subtask_name: str, subtask_desc: str) -> str:
    return f"""{CRITERIA}

You are designing the evaluation procedure for a sub-task called
"{subtask_name}".

Sub-task description:
{subtask_desc}

Write a numbered list of 4-7 concrete EVALUATION STEPS that an evaluator
would follow to score this sub-task. The steps must be specific and
actionable, must reference the error definitions above, and must end with
a step that assigns the numeric scores. Do NOT include any introduction,
explanation, or trailing notes - output ONLY the numbered list."""


def main() -> None:
    out: dict[str, str] = {}
    for name, desc in SUBTASKS.items():
        prompt = build_prompt(name, desc)
        steps = call_vlm_text(prompt).strip()
        out[name] = steps
        print(f"=== {name} ===\n{steps}\n")
    EVAL_STEPS_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved -> {EVAL_STEPS_FILE}")


if __name__ == "__main__":
    main()
