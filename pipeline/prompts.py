"""Prompts and JSON schemas for the two sub-tasks.

Both sub-tasks use OpenAI Structured Outputs (json_schema mode). Score
fields are string enums over the 11-step grid 0.0 .. 1.0 so that future
G-Eval style probability-weighted scoring stays trivially compatible
(each score is a single token chosen from a fixed set).

Field order in the schemas matters: observations come BEFORE scores so
that constrained decoding produces a CoT-shaped trace.
"""
from __future__ import annotations
import json
from pathlib import Path

from config import SCORE_ENUM, EVAL_STEPS_FILE
from criteria_2 import CRITERIA


def _load_steps() -> dict[str, str]:
    return json.loads(Path(EVAL_STEPS_FILE).read_text())


# ---------- Sub-task A: per-frame semantic + cross-view ----------

def build_semantic_prompt(frame_idx: int) -> str:
    steps = _load_steps()["per_frame_semantic_crossview"]
    return f"""{CRITERIA}

Evaluation Steps for this sub-task:
{steps}

You are now evaluating a single time-step (frame index = {frame_idx}) of one
driving video. You will receive 12 images in this order:

  1.  GT  - FRONT_LEFT
  2.  GEN - FRONT_LEFT
  3.  GT  - FRONT
  4.  GEN - FRONT
  5.  GT  - FRONT_RIGHT
  6.  GEN - FRONT_RIGHT
  7.  GT  - BACK_LEFT
  8.  GEN - BACK_LEFT
  9.  GT  - BACK
  10. GEN - BACK
  11. GT  - BACK_RIGHT
  12. GEN - BACK_RIGHT

For each (GT, GEN) pair, look for SEMANTIC errors. Across the 6 generated
views, look for CROSS-VIEW inconsistencies of the same physical object.

You MUST first fill the 'per_view_observations' and 'overall_finding'
fields. The score fields must be a direct conclusion of those observations,
NOT a guess. The most severe observation drives the score."""


SEMANTIC_SCHEMA = {
    "name": "per_frame_semantic_crossview",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "per_view_observations",
            "overall_finding",
            "semantic_score",
            "cross_view_score",
            "notes",
        ],
        "properties": {
            "per_view_observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["view", "semantic_issue", "cross_view_issue"],
                    "properties": {
                        "view": {
                            "type": "string",
                            "enum": [
                                "FRONT_LEFT",
                                "FRONT",
                                "FRONT_RIGHT",
                                "BACK_LEFT",
                                "BACK",
                                "BACK_RIGHT",
                            ],
                        },
                        "semantic_issue": {"type": "string"},
                        "cross_view_issue": {"type": "string"},
                    },
                },
            },
            "overall_finding": {"type": "string"},
            "semantic_score": {"type": "string", "enum": SCORE_ENUM},
            "cross_view_score": {"type": "string", "enum": SCORE_ENUM},
            "notes": {"type": "string"},
        },
    },
}


# ---------- Sub-task B: per-view temporal (logical + decision) ----------

def build_temporal_prompt(view: str) -> str:
    steps = _load_steps()["per_view_temporal"]
    return f"""{CRITERIA}

Evaluation Steps for this sub-task:
{steps}

You are now evaluating one camera view ({view}) of one driving video across
all 8 time-steps. You will receive 10 images in this order:

  1.  GT  - frame 0 (anchor)
  2.  GEN - frame 0
  3.  GEN - frame 1
  4.  GEN - frame 2
  5.  GEN - frame 3
  6.  GEN - frame 4
  7.  GEN - frame 5
  8.  GEN - frame 6
  9.  GEN - frame 7
  10. GT  - frame 7 (anchor)

Use the GT anchors only as a sanity reference for what the scene SHOULD
look like at the start and end. The 8 generated frames are the subject of
your judgment. Look for LOGICAL errors (disappearance, jitter, impossible
motion) and DECISION errors (unsafe ego behavior).

You MUST first fill the 'temporal_observations' and 'overall_finding'
fields. The score fields must be a direct conclusion of those observations,
NOT a guess."""


TEMPORAL_SCHEMA = {
    "name": "per_view_temporal",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "temporal_observations",
            "overall_finding",
            "logical_score",
            "decision_score",
            "notes",
        ],
        "properties": {
            "temporal_observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "frame_range",
                        "logical_issue",
                        "decision_issue",
                    ],
                    "properties": {
                        "frame_range": {"type": "string"},
                        "logical_issue": {"type": "string"},
                        "decision_issue": {"type": "string"},
                    },
                },
            },
            "overall_finding": {"type": "string"},
            "logical_score": {"type": "string", "enum": SCORE_ENUM},
            "decision_score": {"type": "string", "enum": SCORE_ENUM},
            "notes": {"type": "string"},
        },
    },
}
