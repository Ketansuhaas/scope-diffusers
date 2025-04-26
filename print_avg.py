import json
import argparse
import re
from collections import defaultdict


def compute_avg_scores(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    metric_totals = defaultdict(float)
    metric_counts = defaultdict(int)

    for item in data:
        for key, value in item.items():
            match = re.match(r"(normal|best_scope)_(.+)_score", key)
            if match:
                prefix, metric = match.groups()
                metric_key = f"{prefix}_{metric}"
                metric_totals[metric_key] += value
                metric_counts[metric_key] += 1

    print("Averaged Scores:")
    for metric_key in sorted(metric_totals.keys()):
        avg = metric_totals[metric_key] / metric_counts[metric_key]
        print(f"{metric_key}: {avg:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()
    compute_avg_scores(args.json_path)
