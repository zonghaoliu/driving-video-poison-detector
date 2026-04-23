import csv
import json
from collections import defaultdict
from pathlib import Path

# Agreement threshold for majority voting (can be adjusted based on needs)
AGREEMENT_THRESHOLD = 0.5

# Low score threshold for debugging
LOW_SCORE_THRESHOLD = 0.2


def preprocess_scores(data_rows):
    """
    Preprocess scores from a single CSV file according to the rules:
    1. Find entry threshold: min of max scores of poisoned videos
    2. For scores above threshold from poisoned videos, quantile normalize to 0-1, then map to 0.5-1.0
    3. Non-poisoned videos and scores below threshold remain unchanged

    Args:
        data_rows: List of dicts representing rows from one CSV file

    Returns:
        List of preprocessed rows
    """
    if not data_rows:
        return []

    # Separate poisoned and non-poisoned rows
    poisoned_rows = []
    non_poisoned_rows = []

    for row in data_rows:
        is_pois = row['is_poisoned'].lower() == 'true' if row['is_poisoned'].lower() in ['true', 'false'] else row['is_poisoned']
        if is_pois == True or str(is_pois).lower() == 'true':
            # Deep copy to avoid modifying original during preprocessing
            poisoned_rows.append({k: v for k, v in row.items()})
        else:
            non_poisoned_rows.append(row)

    # If no poisoned videos, return original data unchanged
    if not poisoned_rows:
        return data_rows

    # Rule 1: Find entry threshold
    # For each poisoned video, find its highest score among semantic, logical, decision
    max_scores = []
    for row in poisoned_rows:
        scores = [float(row['semantic']), float(row['logical']), float(row['decision'])]
        max_scores.append(max(scores))

    # Entry threshold is the lowest "high score" among all poisoned videos
    entry_threshold = min(max_scores)

    # Collect all scores from poisoned videos that are above the threshold
    above_threshold_scores = []  # (value, row_idx, dimension)

    for i, row in enumerate(poisoned_rows):
        for dim in ['semantic', 'logical', 'decision']:
            val = float(row[dim])
            if val >= entry_threshold:
                above_threshold_scores.append((val, i, dim))

    # If we have scores above threshold, apply quantile normalization
    if len(above_threshold_scores) > 1:
        # Extract just the values
        values = [x[0] for x in above_threshold_scores]

        # Rule 2: Quantile normalization to 0.0-1.0
        sorted_values = sorted(values)
        n = len(sorted_values)

        # Create mapping from original value to quantile (0-1), handle ties
        value_to_quantile = {}
        for i, val in enumerate(sorted_values):
            if val not in value_to_quantile:
                indices = [j for j, v in enumerate(sorted_values) if v == val]
                avg_rank = sum(indices) / len(indices)
                value_to_quantile[val] = (avg_rank / (n - 1)) if n > 1 else 0.5

        # Rule 3: Map 0.0-1.0 to 0.5-1.0
        quantile_to_scaled = {val: 0.5 + 0.5 * q for val, q in value_to_quantile.items()}

        # Update the scores in poisoned_rows
        for val, row_idx, dim in above_threshold_scores:
            poisoned_rows[row_idx][dim] = quantile_to_scaled[val]

    elif len(above_threshold_scores) == 1:
        # Single value above threshold - map to 1.0
        val, row_idx, dim = above_threshold_scores[0]
        poisoned_rows[row_idx][dim] = 1.0

    # Combine back and return
    result_rows = poisoned_rows + non_poisoned_rows
    return result_rows


def merge_csv_files(csv_files, output_json="merged_output.json"):
    """
    Merge multiple CSV files into a single JSON with majority voting logic.
    Preprocesses each file individually before merging.
    """
    # First, read and preprocess each file individually
    all_preprocessed_rows = []

    for csv_file in csv_files:
        rows = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))

        # print(f"ALL rows for {csv_file}:\n{rows}")

        # Preprocess this file's scores
        preprocessed = preprocess_scores(rows)
        all_preprocessed_rows.extend(preprocessed)

        # print(f"ALL rows for {csv_file}:\n{preprocessed}")

        # Debug: show threshold for this file
        poisoned = [r for r in rows if r['is_poisoned'].lower() == 'true']
        if poisoned:
            maxs = [max(float(r['semantic']), float(r['logical']), float(r['decision'])) for r in poisoned]
            threshold = min(maxs)
            print(f"Preprocessed: {csv_file} (entry threshold: {threshold:.3f})")
        else:
            print(f"Preprocessed: {csv_file} (no poisoned videos)")

    # Collect all data grouped by video_id
    video_data = defaultdict(lambda: {
        'is_poisoned': [],
        'semantic': [],
        'logical': [],
        'decision': [],
        'reasoning': []
    })

    # print(f"ALL rows:\n{all_preprocessed_rows}")

    # Process preprocessed rows
    for row in all_preprocessed_rows:
        vid = row['video_id']
        is_pois = row['is_poisoned'].lower() == 'true' if row['is_poisoned'].lower() in ['true', 'false'] else row['is_poisoned']

        video_data[vid]['is_poisoned'].append(is_pois)
        video_data[vid]['semantic'].append(float(row['semantic']))
        video_data[vid]['logical'].append(float(row['logical']))
        video_data[vid]['decision'].append(float(row['decision']))
        video_data[vid]['reasoning'].append(row['reasoning'])

    # print(f"ALL video data:\n{video_data.items()}")

    result = []
    result_debug = []

    for vid, data in video_data.items():
        total_files = len(data['semantic'])

        # print(f"Processing video ... {vid}")

        def calc_majority_score(scores):
            agree = [s for s in scores if s >= AGREEMENT_THRESHOLD]
            disagree = [s for s in scores if s < AGREEMENT_THRESHOLD]

            if len(agree) > total_files / 2:
                return sum(agree) / len(agree), 'agree'
            elif len(disagree) > total_files / 2:
                return sum(disagree) / len(disagree), 'disagree'
            else:
                return sum(scores) / len(scores), 'tie'

        sem_score, sem_status = calc_majority_score(data['semantic'])
        log_score, log_status = calc_majority_score(data['logical'])
        dec_score, dec_status = calc_majority_score(data['decision'])

        # print(f" - Semantic score={sem_score}, status: {sem_status} from scores {data['semantic']}")
        # print(f" - Logical  score={log_score}, status: {log_status} from scores {data['logical']}")
        # print(f" - Decision score={dec_score}, status: {dec_status} from scores {data['decision']}")

        true_count = sum(1 for x in data['is_poisoned'] if x == True or str(x).lower() == 'true')
        false_count = total_files - true_count

        if true_count > false_count:
            is_poisoned = True
        elif false_count > true_count:
            is_poisoned = False
        else:
            final_avg = (sem_score + log_score + dec_score) / 3
            is_poisoned = final_avg > LOW_SCORE_THRESHOLD

        reasoning = max(data['reasoning'], key=len)

        scores = {
            "semantic": round(sem_score, 1),
            "logical": round(log_score, 1),
            "decision": round(dec_score, 1)
        }
        final_score = round((scores["semantic"] + scores["logical"] + scores["decision"]) / 3, 2)
        attack_level = max(scores, key=scores.get)

        entry = {
            "video_id": vid,
            "is_poisoned": is_poisoned,
            "attack_level": attack_level,
            "scores": scores,
            "final_score": final_score,
            "reasoning": reasoning,
        }
        result.append(entry)

        if is_poisoned is True and final_score < LOW_SCORE_THRESHOLD:
            result_debug.append((vid, "false poisoned", final_score))
        elif is_poisoned is False and final_score >= LOW_SCORE_THRESHOLD:
            result_debug.append((vid, "false not poisoned", final_score))

    # print(f"Results:\n{result}")

    result.sort(key=lambda x: x['video_id'])

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nMerged {len(csv_files)} CSV files")
    print(f"Output: {len(result)} unique videos -> '{output_json}'")

    print(f"\nDebug Info:")
    print(f"   - Videos marked to debug: {len(result_debug)}")
    for vid, issue, score in result_debug:
        if issue == "false poisoned":
            print(f"   - {vid}: Marked as poisoned but low score (< {LOW_SCORE_THRESHOLD}) ({score})")
        elif issue == "false not poisoned":
            print(f"   - {vid}: Marked as not poisoned but high score (>= {LOW_SCORE_THRESHOLD}) ({score})")

    print("\nStats Summary:")
    print(f"   - Total unique videos: {len(result)}")
    if len(result) > 0:
        poisoned_count = sum(1 for x in result if x['is_poisoned'] == True)
        print(f"   - Poisoned count: {poisoned_count}")
        print(f"   - Poisoned rate: {poisoned_count / len(result):.2%}")

    return result


def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <csv1> <csv2> [csv3...] [output.json]")
        print("Example: python script.py a.csv b.csv c.csv merged.json")
        return

    if sys.argv[-1].endswith('.json'):
        output = sys.argv[-1]
        inputs = sys.argv[1:-1]
    else:
        output = "merged_output_raw.json"
        inputs = sys.argv[1:]

    for f in inputs:
        if not Path(f).exists():
            print(f"Error: File '{f}' not found")
            return

    merge_csv_files(inputs, output)


if __name__ == "__main__":
    main()