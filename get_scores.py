import argparse
import ast
import os
import json
import torch
import pandas as pd

from interpolator.interpolator import get_interpolator
from helpers import get_all_hparam_combinations
from scorers.scores import *


def sanitize_for_path(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")

def parse_args():
    parser = argparse.ArgumentParser(description="Compute modular scores for baseline vs scope outputs.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpolation_method", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, default="exp_dump/eval_output")
    parser.add_argument("--csv_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.csv_path)

    # df = df.head(20)

    interpolator_cls = get_interpolator(args.interpolation_method.lower())
    hparam_combos = get_all_hparam_combinations(interpolator_cls)

    model_dir = sanitize_for_path(args.model_name)
    top_level_dir = os.path.join(
        args.exp_dir,
        model_dir,
        args.interpolation_method,
        f"steps_{args.num_inference_steps}",
        f"seed_{args.seed}"
    )

    print(f"Looking under: {top_level_dir}")

    scorers = [
        CLIPScorer(device=device),
        VQAScorer(device=device),
        VQACompositeScorer(device=device)
        # HPSv2Scorer(device=device),
        # Add more scorers here if needed
    ]

    for scorer in scorers:
        results = []
        print(f"\n===> Scoring with: {scorer.name()}")

        for idx, row in df.iterrows():
            prompt_schedule_list = ast.literal_eval(row["schedule"])
            if not prompt_schedule_list:
                continue
            final_prompt = prompt_schedule_list[-1]

            scope_scores = {}
            best_score = float("-inf")
            best_path = None
            best_suffix = None
            baseline_score = None

            row_folder = os.path.join(top_level_dir, f"row_{idx}")
            baseline_path = os.path.join(row_folder, "baseline.png")

            if not os.path.exists(baseline_path):
                continue
            
            if scorer.name == "vqa_composite":
                subdescriptions = ast.literal_eval(row["subdescriptions"])
                baseline_score = scorer.compute(baseline_path, subdescriptions)
            else:
                baseline_score = scorer.compute(baseline_path, final_prompt)

            for combo in hparam_combos:
                combo["seed"] = args.seed
                suffix_parts = [f"{k}_{v}" for k, v in combo.items() if k != "seed"]
                suffix_str = "_".join(suffix_parts)

                final_out_dir = os.path.join(row_folder, suffix_str)
                scope_path = os.path.join(final_out_dir, "scope.png")

                if not os.path.exists(scope_path):
                    continue

                if scorer.name == "vqa_composite":
                    subdescriptions = ast.literal_eval(row["subdescriptions"])
                    score = scorer.compute(scope_path, subdescriptions)
                else:
                    score = scorer.compute(scope_path, final_prompt)

                scope_scores[suffix_str] = score
                if score > best_score:
                    best_score = score
                    best_path = scope_path
                    best_suffix = suffix_str

            if baseline_score is not None and best_score > float("-inf"):
                results.append({
                    "image_id": idx,
                    f"normal_{scorer.name()}_score": baseline_score,
                    f"best_scope_{scorer.name()}_score": best_score,
                    "difference": best_score - baseline_score,
                    "best_path": best_path,
                    "best_suffix": best_suffix,
                    f"scope_{scorer.name()}_scores": scope_scores
                })

        output_json_path = os.path.join(top_level_dir, f"{scorer.name()}_scores.json")
        print(f"Saving {scorer.name()} scores to {output_json_path}")
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()