"""Step 4 - Aggregate sub-task outputs into one annotation per video.

The aggregator is configurable so we can sweep different rules over the
SAME raw VLM outputs without burning more API calls.

CLI:
    python postprocess.py                              # default rule + default raw dir
    python postprocess.py --raw outputs/exp_A_anchors/vlm_raw \\
                          --out outputs/exp_A_top2_t07/vlm_clean.json \\
                          --reduce top2 --threshold 0.7

reduce options:
    max     - max over all sub-call scores (original behaviour)
    top2    - mean of the top-2 scores (more robust to a single hallucination)
    top3    - mean of the top-3 scores
    p75     - 75th percentile (interpolated)

Cross-view scores are folded into the LOGICAL bucket the same way under
every reduce strategy: cv contribution is reduced separately, then
combined with the temporal logical contribution via max.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from config import VLM_RAW_DIR, CLEAN_JSON, FIRST_RUN_IDS


def _f(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _reduce(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    vs = sorted(values, reverse=True)
    if how == "max":
        return vs[0]
    if how == "top2":
        return sum(vs[:2]) / min(2, len(vs))
    if how == "top3":
        return sum(vs[:3]) / min(3, len(vs))
    if how == "p75":
        # interpolated 75th percentile of the original (ascending) values
        asc = sorted(values)
        if len(asc) == 1:
            return asc[0]
        idx = 0.75 * (len(asc) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(asc) - 1)
        frac = idx - lo
        return asc[lo] * (1 - frac) + asc[hi] * frac
    raise ValueError(f"unknown reduce: {how}")


def aggregate_one(raw: dict, reduce: str, threshold: float) -> dict:
    sem_calls = raw.get("semantic", [])
    tmp_calls = raw.get("temporal", [])

    sem_vals = [_f(c.get("semantic_score")) for c in sem_calls]
    cv_vals = [_f(c.get("cross_view_score")) for c in sem_calls]
    log_vals = [_f(c.get("logical_score")) for c in tmp_calls]
    dec_vals = [_f(c.get("decision_score")) for c in tmp_calls]

    sem = _reduce(sem_vals, reduce)
    cv = _reduce(cv_vals, reduce)
    log_t = _reduce(log_vals, reduce)
    log = max(cv, log_t)  # cross-view goes into logical
    dec = _reduce(dec_vals, reduce)

    scores = {"semantic": sem, "logical": log, "decision": dec}
    attack_level = max(scores, key=scores.get)
    is_poisoned = any(v > threshold for v in scores.values())

    ordered = sorted(scores.values(), reverse=True)
    final_score = round(0.5 * ordered[0] + 0.3 * ordered[1] + 0.2 * ordered[2], 3)

    # Surface the highest-scoring sub-call's note as reasoning
    def _argmax_call(calls, key):
        if not calls:
            return None
        return max(calls, key=lambda c: _f(c.get(key)))

    if attack_level == "semantic":
        src = _argmax_call(sem_calls, "semantic_score")
    elif attack_level == "decision":
        src = _argmax_call(tmp_calls, "decision_score")
    else:
        src_t = _argmax_call(tmp_calls, "logical_score")
        src_c = _argmax_call(sem_calls, "cross_view_score")
        src = src_t if (_f((src_t or {}).get("logical_score")) >= _f((src_c or {}).get("cross_view_score"))) else src_c
    reasoning = (src or {}).get("overall_finding") or (src or {}).get("notes") or ""

    return {
        "video_id": raw["video_id"],
        "is_poisoned": is_poisoned,
        "attack_level": attack_level,
        "scores": {
            "semantic": round(sem, 3),
            "logical": round(log, 3),
            "decision": round(dec, 3),
        },
        "final_score": final_score,
        "reasoning": reasoning,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=VLM_RAW_DIR)
    p.add_argument("--out", type=Path, default=CLEAN_JSON)
    p.add_argument("--reduce", choices=["max", "top2", "top3", "p75"], default="max")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    out = []
    for vid in FIRST_RUN_IDS:
        path = args.raw / f"{vid}.json"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        raw = json.loads(path.read_text())
        out.append(aggregate_one(raw, args.reduce, args.threshold))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} records [reduce={args.reduce}, threshold={args.threshold}] -> {args.out}")


if __name__ == "__main__":
    main()
