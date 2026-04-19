import csv
import json
from collections import defaultdict
from pathlib import Path

# Agreement threshold for majority voting (can be adjusted based on needs)
AGREEMENT_THRESHOLD = 0.5

# Low score threshold for debugging
LOW_SCORE_THRESHOLD = 0.3

def merge_csv_files(csv_files, output_json="merged_output.json"):
    """
    Merge multiple CSV files into a single JSON with majority voting logic.

    Args:
        csv_files: List of paths to CSV files
        output_json: Output JSON file path
    """
    # Collect all data grouped by video_id
    video_data = defaultdict(lambda: {
        'is_poisoned': [],
        'semantic': [],
        'logical': [],
        'decision': [],
        'reasoning': []
    })

    # Read all CSV files
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['video_id']
                # Convert is_poisoned to boolean
                is_pois = row['is_poisoned'].lower() == 'true' if row['is_poisoned'].lower() in ['true', 'false'] else row['is_poisoned']

                video_data[vid]['is_poisoned'].append(is_pois)
                video_data[vid]['semantic'].append(float(row['semantic']))
                video_data[vid]['logical'].append(float(row['logical']))
                video_data[vid]['decision'].append(float(row['decision']))
                video_data[vid]['reasoning'].append(row['reasoning'])

    result = []
    result_debug = []

    for vid, data in video_data.items():
        total_files = len(data['semantic'])

        # Helper function for majority score calculation
        def calc_majority_score(scores):
            agree = [s for s in scores if s > AGREEMENT_THRESHOLD]
            disagree = [s for s in scores if s <= AGREEMENT_THRESHOLD]

            # Majority agree (>AGREEMENT_THRESHOLD)
            if len(agree) > total_files / 2:
                return sum(agree) / len(agree), 'agree'
            # Majority disagree (<=AGREEMENT_THRESHOLD)
            elif len(disagree) > total_files / 2:
                return sum(disagree) / len(disagree), 'disagree'
            # Tie: average all
            else:
                return sum(scores) / len(scores), 'tie'

        # Calculate merged scores for each dimension
        sem_score, sem_status = calc_majority_score(data['semantic'])
        log_score, log_status = calc_majority_score(data['logical'])
        dec_score, dec_status = calc_majority_score(data['decision'])

        # Majority vote for is_poisoned
        true_count = sum(1 for x in data['is_poisoned'] if x == True or str(x).lower() == 'true')
        false_count = total_files - true_count

        if true_count > false_count:
            is_poisoned = True
        elif false_count > true_count:
            is_poisoned = False
        else:  # Tie-breaker: use final score (should be impossible on odd number of files)
            final_avg = (sem_score + log_score + dec_score) / 3
            is_poisoned = final_avg > LOW_SCORE_THRESHOLD

        # Longest reasoning
        reasoning = max(data['reasoning'], key=len)

        # Calculate final metrics
        scores = {
            "semantic": round(sem_score, 1),
            "logical": round(log_score, 1),
            "decision": round(dec_score, 1)
        }
        final_score = round((sem_score + log_score + dec_score) / 3, 2)
        attack_level = max(scores, key=scores.get)

        entry = {
            "video_id": vid,
            "is_poisoned": is_poisoned,
            "attack_level": attack_level,
            "scores": scores,
            "final_score": final_score,
            "reasoning": reasoning,
            # Debug info (optional, remove if not needed)
            # "_debug": {
            #     "sources_count": total_files,
            #     "agreement_status": {
            #         "semantic": sem_status,
            #         "logical": log_status,
            #         "decision": dec_status
            #     },
            #     "is_poisoned_vote": {"true": true_count, "false": false_count}
            # }
        }
        result.append(entry)

        # Exotic values debug
        if is_poisoned is True and final_score < LOW_SCORE_THRESHOLD:
            result_debug.append((vid, "false poisoned", final_score))
        elif is_poisoned is False and final_score >= LOW_SCORE_THRESHOLD:
            result_debug.append((vid, "false not poisoned", final_score))

        # Collect stats
        stats = {
            "total_sources": total_files,
            "poisoned_vote": {"true": true_count, "false": false_count},
            "poisoned_rate": true_count / total_files if total_files > 0 else 0
        }

    # Sort by video_id for consistency (just to make sure)
    result.sort(key=lambda x: x['video_id'])

    # Write output JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged {len(csv_files)} CSV files")
    print(f"✅ Output: {len(result)} unique videos → '{output_json}'")

    # Print debug results if any
    print(f"\n🔍 Debug Info:")
    print(f"   - Videos marked to debug: {len(result_debug)}")
    for vid, issue, score in result_debug:
        if issue == "false poisoned":
            print(f"   - {vid}: Marked as poisoned but low score (< {LOW_SCORE_THRESHOLD}) ({score})")
        elif issue == "false not poisoned":
            print(f"   - {vid}: Marked as not poisoned but high score (>= {LOW_SCORE_THRESHOLD}) ({score})")

    # Print stats summary (total num of videos, poisoned count, poisoned rate)
    print("\n📊 Stats Summary:")
    print(f"   - Total unique videos: {len(result)}")
    if len(result) > 0:
        poisoned_count = sum(1 for x in result if x['is_poisoned'] == True)
        print(f"   - Poisoned count: {poisoned_count}")
        print(f"   - Poisoned rate: {poisoned_count / len(result):.2%}")

    return result


def main():
    import sys

    # Usage: python script.py file1.csv file2.csv file3.csv [output.json]
    if len(sys.argv) < 3:
        print("Usage: python script.py <csv1> <csv2> [csv3...] [output.json]")
        print("Example: python script.py a.csv b.csv c.csv merged.json")
        return

    # Last argument is output if it ends with .json, otherwise use default
    if sys.argv[-1].endswith('.json'):
        output = sys.argv[-1]
        inputs = sys.argv[1:-1]
    else:
        output = "merged_output.json"
        inputs = sys.argv[1:]

    # Validate files exist
    for f in inputs:
        if not Path(f).exists():
            print(f"❌ Error: File '{f}' not found")
            return

    merge_csv_files(inputs, output)


if __name__ == "__main__":
    main()