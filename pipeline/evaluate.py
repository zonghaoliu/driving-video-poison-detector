"""Step 5 - Compare VLM annotations against human ground truth.

Metrics produced (only over the IDs present in vlm_clean.json):
  - is_poisoned binary: accuracy, precision, recall, F1, confusion matrix
  - attack_level multiclass (semantic/logical/decision): accuracy, 3x3 CM
  - per-dimension scores: MAE and Pearson correlation
  - top-K disagreement cases with both reasonings side by side

Result is written to outputs/eval_report.md (Markdown).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from statistics import mean

from config import CLEAN_JSON, REPORT_MD, HUMAN_JSON


CLASSES = ["semantic", "logical", "decision"]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    n = len(y_true)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _confusion(y_true: list[str], y_pred: list[str], classes: list[str]) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def main() -> None:
    # Optional CLI override: `python evaluate.py outputs/exp_v4/vlm_clean.json`
    if len(sys.argv) > 1:
        clean_path = Path(sys.argv[1])
        report_path = clean_path.parent / "eval_report.md"
    else:
        clean_path = CLEAN_JSON
        report_path = REPORT_MD
    vlm = {str(r["video_id"]).replace(".mp4", ""): r for r in json.loads(clean_path.read_text())}
    human_all = json.loads(Path(HUMAN_JSON).read_text())
    human = {r["video_id"]: r for r in human_all if r["video_id"] in vlm}

    ids = sorted(vlm.keys())
    if not ids:
        print("no overlap between vlm_clean.json and human annotations")
        return

    y_true_p = [bool(human[i]["is_poisoned"]) for i in ids]
    y_pred_p = [bool(vlm[i]["is_poisoned"]) for i in ids]
    bin_m = _binary_metrics(y_true_p, y_pred_p)

    y_true_a = [str(human[i]["attack_level"]).lower() for i in ids]
    y_pred_a = [str(vlm[i]["attack_level"]).lower() for i in ids]
    cm = _confusion(y_true_a, y_pred_a, CLASSES)
    a_acc = sum(1 for t, p in zip(y_true_a, y_pred_a) if t == p) / len(ids)

    dim_stats = {}
    for dim in CLASSES:
        h = [float(human[i]["scores"][dim]) for i in ids]
        v = [float(vlm[i]["scores"][dim]) for i in ids]
        mae = mean(abs(a - b) for a, b in zip(h, v))
        dim_stats[dim] = {"mae": mae, "pearson": _pearson(h, v)}

    diffs = []
    for i in ids:
        d = sum(abs(float(human[i]["scores"][k]) - float(vlm[i]["scores"][k])) for k in CLASSES)
        diffs.append((d, i))
    diffs.sort(reverse=True)
    top_k = diffs[:5]

    report = []
    report.append(f"# VLM vs Human Evaluation Report\n")
    report.append(f"Videos evaluated: **{len(ids)}** ({', '.join(ids)})\n")

    report.append("## 1. is_poisoned (binary)\n")
    report.append(_md_table(
        ["Accuracy", "Precision", "Recall", "F1", "TP", "TN", "FP", "FN"],
        [[f"{bin_m['acc']:.3f}", f"{bin_m['prec']:.3f}", f"{bin_m['rec']:.3f}",
          f"{bin_m['f1']:.3f}", bin_m['tp'], bin_m['tn'], bin_m['fp'], bin_m['fn']]],
    ))

    report.append("\n## 2. attack_level (3-class)\n")
    report.append(f"Accuracy: **{a_acc:.3f}**\n\nConfusion matrix (rows = human, cols = VLM):\n")
    report.append(_md_table([""] + CLASSES, [[CLASSES[i]] + cm[i] for i in range(3)]))

    report.append("\n## 3. Per-dimension score regression\n")
    report.append(_md_table(
        ["Dimension", "MAE", "Pearson r"],
        [[d, f"{dim_stats[d]['mae']:.3f}", f"{dim_stats[d]['pearson']:.3f}"] for d in CLASSES],
    ))

    report.append("\n## 4. Top-5 largest disagreements\n")
    for d, vid in top_k:
        h, v = human[vid], vlm[vid]
        report.append(f"### video {vid} (sum |Δ| = {d:.2f})\n")
        report.append(_md_table(
            ["source", "attack_level", "semantic", "logical", "decision", "is_poisoned"],
            [
                ["human", h["attack_level"], h["scores"]["semantic"], h["scores"]["logical"],
                 h["scores"]["decision"], h["is_poisoned"]],
                ["VLM", v["attack_level"], v["scores"]["semantic"], v["scores"]["logical"],
                 v["scores"]["decision"], v["is_poisoned"]],
            ],
        ))
        report.append(f"- **human reasoning:** {h.get('reasoning','')}\n")
        report.append(f"- **VLM reasoning:** {v.get('reasoning','')}\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
