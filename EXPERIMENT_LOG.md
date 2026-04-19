# Experiment Log

## Current Best: Experiment A + top3 aggregation + threshold 0.7

- **Output directory**: `outputs/exp_A_anchors/`
  - `vlm_raw/` — 10 raw VLM outputs (11 sub-calls each)
  - Aggregated: `outputs/exp_A_top3_t07/vlm_clean.json`
  - Report: `outputs/exp_A_top3_t07/eval_report.md`
- **Criteria file**: `pipeline/criteria.py` — distilled from V1, with
  calibration anchors and hard rules added
- **Aggregation**: `top3` reduce + `0.7` poisoned threshold
- **API calls**: 110 total (11 per video × 10 videos)

### Why this is the best

It gives the **best balance of classification accuracy and low false
positives** across all experiments. See the comparison table below.

---

## All Experiments Compared

### Approach summary

| # | Name | Prompt source | Image input | Calls/video | Aggregation |
|---|---|---|---|---|---|
| ① | **Exp A + top3+0.7** | Distilled V1 + our anchors/hard rules | 144 sub-images (per-frame & per-view) | 11 | top-3 mean, thr 0.7 |
| ② | V4 reproduction | Prompt V4 verbatim (two-stage auto-CoT) | 8 full-res frames (2688×784) | 1 | direct output |
| ③ | V1 criteria + top3+0.7 | V1 verbatim (all 8 scoring examples) | 144 sub-images | 11 | top-3 mean, thr 0.7 |

### Head-to-head results (10 videos)

| Metric | ① Exp A + top3+0.7 | ② V4 | ③ V1 criteria + top3+0.7 |
|---|---|---|---|
| **is_poisoned Accuracy** | **0.60** | 0.20 | 0.20 |
| is_poisoned F1 | 0.00 | **0.33** | 0.20 |
| False Positives | **2** | 8 | 7 |
| Recall | 0.00 | **1.00** | 0.50 |
| **attack_level Accuracy** | **0.80** | 0.30 | **0.80** |
| semantic MAE | 0.45 | **0.26** | 0.65 |
| logical MAE | **0.25** | 0.48 | 0.42 |
| decision MAE | **0.02** | 0.43 | 0.08 |
| semantic Pearson r | -0.08 | **+0.11** | -0.66 |
| logical Pearson r | -0.37 | **+0.44** | -0.12 |

### Observations

1. **No single approach wins on every metric.** ① is strongest on
   classification (attack_level 80%, only 2 FP). ② is strongest on
   ranking (logical Pearson +0.44, only version with positive
   correlation).

2. **The Pearson numbers are unreliable at n=10.** With only 10
   data points (and human decision scores all being 0.0), correlations
   swing wildly. We need the full 100-video run before drawing
   conclusions about ranking quality.

3. **Aggregation matters as much as the prompt.** On the same raw
   data (Exp A), switching from `max` to `top3` improved attack_level
   accuracy from 0.40 → 0.80 and cut FP from 8 → 2, with zero
   additional API cost.

4. **V4 gives the model entire frames** (GT + 3D-box + Gen in one
   image), which helps temporal/logical reasoning but hurts
   classification because the model cannot zoom into fine details. This
   also leads to "slot scoring" — V4 tends to output the same handful of
   score combinations (0.3/0.7/0.4 permutations) for every video.

5. **V1 verbatim criteria with all 8 scoring examples** (③) caused the
   model to over-match: it saw minor issues and treated them as severe
   because they resembled an example scenario. semantic Pearson dropped
   to -0.66.

---

## File Locations

```
outputs/
├── baseline/                  First run, original distilled criteria, max agg
│   ├── vlm_raw/               Raw sub-call outputs
│   ├── vlm_clean.json         Aggregated (max, thr 0.5)
│   └── eval_report.md
│
├── exp_A_anchors/             ① Exp A: distilled criteria + anchors + hard rules
│   └── vlm_raw/               Raw sub-call outputs (reused by all agg sweeps)
│
├── exp_A_max_t05/             ① aggregated with max, thr 0.5
├── exp_A_max_t07/             ① aggregated with max, thr 0.7
├── exp_A_top2_t05/            ① aggregated with top2, thr 0.5
├── exp_A_top2_t07/            ① aggregated with top2, thr 0.7
├── exp_A_top3_t07/            ① aggregated with top3, thr 0.7  ★ BEST
├── exp_A_p75_t05/             ① aggregated with p75, thr 0.5
│
├── exp_v4/                    ② V4 reproduction
│   ├── eval_steps_v4.txt      Stage-1 generated evaluation steps
│   ├── vlm_raw/               Per-video raw outputs
│   ├── vlm_clean.json         Normalized annotations
│   └── eval_report.md
│
├── exp_v1_criteria/           ③ V1 verbatim criteria
│   └── vlm_raw/               Raw sub-call outputs
├── v1c_max_t05/               ③ aggregated with max, thr 0.5
├── v1c_top2_t05/              ③ aggregated with top2, thr 0.5
├── v1c_top3_t07/              ③ aggregated with top3, thr 0.7
└── v1c_p75_t05/               ③ aggregated with p75, thr 0.5
```

---

## Pipeline Architecture (current best = ①)

```
mp4 video
  │
  ▼
preprocess.py ─── 144 sub-images (8 frames × 6 views × 3 layers)
  │
  ▼
gen_eval_steps.py ─── GPT-4o generates evaluation steps (once, cached)
  │
  ▼
run_vlm.py ─── 11 GPT-4o calls per video
  │              8× per-frame: 12 images (6 GT + 6 GEN) → semantic + cross-view
  │              3× per-view:  10 images (8 GEN + 2 GT)  → logical + decision
  │              Uses Structured Outputs (json_schema), scores as enum strings
  │
  ▼
postprocess.py ─── Rule-based aggregation (configurable reduce + threshold)
  │                 semantic = reduce(8 semantic_scores)
  │                 logical  = max(reduce(8 cross_view), reduce(3 logical))
  │                 decision = reduce(3 decision_scores)
  │
  ▼
evaluate.py ─── Compare vs merge_tools/merged_output.json
                 Acc / Prec / Recall / F1 / confusion matrix / MAE / Pearson
```

---

## Next Steps

1. **Scale to 100 videos** with ① (Exp A + top3+0.7) — need ~1100 API
   calls, estimated ~15 min runtime
2. Evaluate whether Pearson turns positive at n=100
3. Consider a hybrid approach: use ①'s sub-task split for
   semantic/decision, and ②'s full-frame approach for logical/temporal
4. Write up findings for the project report
