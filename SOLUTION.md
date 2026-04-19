# Prompt-Based Auto-Annotation Pipeline

Automated safety evaluation for generative driving videos. Given
multi-view driving clips, the system detects semantic, logical, and
decision-level errors in the generated frames, scores their severity
(0.0–1.0), and produces JSON annotations comparable to human labelling.

**Current best configuration**: Subtask pipeline (Exp A) with `top3`
aggregation and `0.7` poisoned threshold. See the
[Experiment Log](EXPERIMENT_LOG.md) for full comparison of all runs.

---

## 1. Problem

Each video in `mixed_group_12/` has 8 frames, each containing 6 camera
views (FL / F / FR / BL / B / BR) and 3 vertical layers (ground truth /
3D bounding box / generated result). About 30% of the 100 videos contain
adversarial errors. The task is to:

1. Detect whether a video is "poisoned" (contains severe errors)
2. Classify the dominant error type (semantic / logical / decision)
3. Score each error dimension on a 0.0–1.0 severity scale
4. Compare automated annotations against human ground truth

---

## 2. Approach — Subtask Pipeline

### Core idea

GPT-4o's attention degrades when processing many images at once. Instead
of sending an entire video in one call, we decompose the evaluation into
**11 small, focused API calls per video**, then aggregate results
deterministically.

### Step-by-step flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING  (preprocess.py, once)                               │
│                                                                     │
│  mp4  →  decode 8 frames  →  split each into 6 cols × 3 rows       │
│  output: 144 sub-images per video                                   │
│  naming: outputs/frames/{vid}/f{frame}_{VIEW}_{layer}.jpg           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AUTO-COT GENERATION  (gen_eval_steps.py, once, cached)             │
│                                                                     │
│  G-Eval style: ask GPT-4o to write evaluation steps from our        │
│  criteria (criteria.py). Steps are cached in eval_steps.json and    │
│  injected into every subsequent prompt — same steps for all videos. │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VLM EVALUATION  (run_vlm.py, 11 calls per video)                   │
│                                                                     │
│  Subtask A — per-frame semantic + cross-view  (×8, one per frame)   │
│    Input: 12 images (6 GT + 6 GEN for one time-step)                │
│    Job:   compare GT↔GEN for semantic errors per view;              │
│           compare 6 GEN views for cross-view inconsistency          │
│    Output: semantic_score, cross_view_score, notes                  │
│                                                                     │
│  Subtask B — per-view temporal  (×3, FL / F / FR only)              │
│    Input: 10 images (8 GEN frames + first & last GT as anchors)     │
│    Job:   check temporal coherence across 8 frames (logical);       │
│           check ego-vehicle behaviour (decision)                    │
│    Output: logical_score, decision_score, notes                     │
│                                                                     │
│  All calls use OpenAI Structured Outputs (json_schema mode).        │
│  Schema forces: observations → reasoning → scores (CoT ordering).   │
│  Scores are string enums ("0.0" .. "1.0") for future compatibility  │
│  with G-Eval probability-weighted scoring.                          │
│                                                                     │
│  Raw output → outputs/{exp}/vlm_raw/{vid}.json                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AGGREGATION  (postprocess.py, no API calls)                        │
│                                                                     │
│  Collapse 11 sub-call scores into one video-level annotation:       │
│                                                                     │
│    semantic = reduce(8 semantic_scores)                              │
│    logical  = max(reduce(8 cross_view_scores),                      │
│                   reduce(3 logical_scores))                         │
│    decision = reduce(3 decision_scores)                             │
│                                                                     │
│  reduce strategy is configurable (max / top2 / top3 / p75).         │
│  Best so far: top3 (mean of top 3 values — dampens single           │
│  hallucinated outliers).                                            │
│                                                                     │
│    attack_level = argmax(semantic, logical, decision)                │
│    is_poisoned  = any(score > threshold)    [best: 0.7]             │
│    final_score  = 0.5 × max + 0.3 × mid + 0.2 × min               │
│                                                                     │
│  Output → outputs/{exp}/vlm_clean.json                              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EVALUATION  (evaluate.py)                                          │
│                                                                     │
│  Compare vlm_clean.json vs merge_tools/merged_output.json:          │
│    - is_poisoned:  Accuracy, Precision, Recall, F1, confusion matrix│
│    - attack_level: Accuracy, 3×3 confusion matrix                   │
│    - Per-dimension: MAE, Pearson r                                  │
│    - Top-5 largest disagreements with side-by-side reasoning        │
│                                                                     │
│  Output → outputs/{exp}/eval_report.md                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Prompt design

| Component | Source | Role |
|---|---|---|
| Error definitions | Distilled from Prompt V1 | What counts as semantic / logical / decision error |
| Calibration anchors | Our addition | 6 score-to-scenario examples that pin the 0.0–1.0 scale |
| Hard rules | Our addition | "no specific error = 0.0", "unsure ≤ 0.2", etc. |
| Evaluation steps | Auto-generated by GPT-4o (G-Eval style) | Procedural checklist injected into every prompt |
| Output schema | OpenAI Structured Outputs | Guarantees field order, enforces CoT before scores |

---

## 3. Baseline Comparison — V4 Reproduction

We also implemented the original Prompt Version 4 verbatim as a
controlled baseline (`pipeline/prompt_v4.py` + `pipeline/run_v4.py`).

V4 uses a two-stage flow: (1) ask GPT-4o to generate evaluation steps
from the criteria, (2) feed the steps + 8 full-resolution frames
(2688×784, containing all 3 layers) into a follow-up call. One API call
per video, output is the final JSON directly.

---

## 4. Results (10-video pilot)

| Metric | Subtask + top3 + 0.7 | V4 verbatim |
|---|---|---|
| **is_poisoned Accuracy** | **0.60** | 0.20 |
| is_poisoned F1 | 0.00 | **0.33** |
| False Positives | **2** | 8 |
| Recall | 0.00 | **1.00** |
| **attack_level Accuracy** | **0.80** | 0.30 |
| semantic MAE | 0.45 | **0.26** |
| logical MAE | **0.25** | 0.48 |
| decision MAE | **0.02** | 0.43 |
| semantic Pearson r | -0.08 | **+0.11** |
| logical Pearson r | -0.37 | **+0.44** |

**Key findings**:
- The subtask pipeline wins on classification (attack_level 80%,
  only 2 false positives)
- V4 wins on ranking quality (only version with positive Pearson
  correlations), likely because full frames preserve temporal context
- Aggregation strategy matters as much as the prompt — switching from
  `max` to `top3` on the same raw data improved attack_level accuracy
  from 40% → 80% with zero additional API cost
- Pearson correlations at n=10 are statistically unreliable; the full
  100-video run is needed before drawing conclusions

---

## 5. How to Run

```bash
# Preprocess (once)
python3 pipeline/preprocess.py

# Subtask pipeline
python3 pipeline/gen_eval_steps.py                   # once, cached
python3 pipeline/run_vlm.py                          # 110 calls for 10 videos
python3 pipeline/postprocess.py \
    --raw outputs/exp_A_anchors/vlm_raw \
    --out outputs/exp_A_top3_t07/vlm_clean.json \
    --reduce top3 --threshold 0.7
python3 pipeline/evaluate.py outputs/exp_A_top3_t07/vlm_clean.json

# V4 baseline
python3 pipeline/run_v4.py
python3 pipeline/evaluate.py outputs/exp_v4/vlm_clean.json

# Custom subset
python3 pipeline/preprocess.py 00 01 02
python3 pipeline/run_vlm.py 00 01 02
```

The `EXPERIMENT` variable in `pipeline/config.py` controls output
routing. Change it to start a fresh experiment without overwriting
previous results.

---

## 6. Output Schema

Both pipelines produce the same JSON shape, matching
`merge_tools/merged_output.json`:

```json
{
  "video_id": "00",
  "is_poisoned": false,
  "attack_level": "semantic",
  "scores": { "semantic": 0.3, "logical": 0.2, "decision": 0.0 },
  "final_score": 0.21,
  "reasoning": "one-line description of the dominant finding"
}
```

---

## 7. Repository Layout

```
Project/
├── mixed_group_12/           100 driving videos (00.mp4 – 99.mp4)
├── merge_tools/
│   ├── *.csv                 3 annotators' raw labels
│   ├── convert_merge.py      Majority-vote merger
│   └── merged_output.json    Human ground truth
├── pipeline/
│   ├── config.py             Global config + experiment switch
│   ├── criteria.py           Error definitions + anchors + hard rules
│   ├── gen_eval_steps.py     Auto-CoT step generator (G-Eval)
│   ├── eval_steps.json       Cached evaluation steps
│   ├── preprocess.py         Video → 144 sub-images
│   ├── prompts.py            Subtask prompts + JSON schemas
│   ├── run_vlm.py            Subtask runner (11 calls / video)
│   ├── postprocess.py        Aggregation (configurable reduce + threshold)
│   ├── evaluate.py           Metrics + report generation
│   ├── prompt_v4.py          V4 verbatim text (locked)
│   └── run_v4.py             V4 runner (1 call / video)
├── outputs/
│   ├── frames/               Preprocessed sub-images
│   ├── exp_A_anchors/        ★ Current best raw data
│   ├── exp_A_top3_t07/       ★ Current best aggregated + report
│   ├── exp_v4/               V4 baseline
│   ├── baseline/             First run (archived)
│   └── exp_v1_criteria/      V1-verbatim experiment (archived)
├── SOLUTION.md               This document
├── EXPERIMENT_LOG.md         Detailed experiment comparison
└── CodeReader/               Project structure + dev log
```

---

## 8. Next Steps

1. Scale to 100 videos (~1100 API calls, ~15 min) with current best
2. Verify whether Pearson turns positive at larger sample size
3. Explore a hybrid: subtask split for semantic/decision + V4's
   full-frame approach for logical/temporal
4. Write up findings for the project report
