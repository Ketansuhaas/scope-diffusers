import argparse
import ast
import os
import json
import torch
import clip
import numpy as np
import pandas as pd
from PIL import Image

from interpolator.interpolator import get_interpolator
from helpers import get_all_hparam_combinations

###############################################################################
# Matches run_scope.py
###############################################################################
def sanitize_for_path(name: str) -> str:
    """
    Replaces slashes/spaces to ensure valid directory names.
    Must match run_scope.py's approach.
    """
    return name.replace("/", "_").replace(" ", "_")


def compute_clip_similarity(
    model,
    preprocess,
    image_path: str,
    text_prompt: str,
    device: str = "cuda"
) -> float:
    """
    Compute cosine similarity between the image at `image_path` and `text_prompt`
    using an openai/clip model.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text_prompt], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = (image_features * text_features).sum().item()
    return similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Generalized script to compute CLIP scores (SCoPE vs Baseline).")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Name/path of the diffusion model (same as run_scope).")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Same as run_scope.py")
    parser.add_argument("--seed", type=int, default=42,
                        help="Same as run_scope.py")
    parser.add_argument("--interpolation_method", type=str, default="nlerp_og",
                        help="Which interpolator class to use (e.g. nlerp_og).")
    parser.add_argument("--exp_dir", type=str, default="exp_dump/eval_output",
                        help="Root directory where run_scope.py saved images.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV path with prompt schedules (same as run_scope).")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CSV (same as run_scope.py)
    df = pd.read_csv(args.csv_path)
    df = df.head(4)  # Sync with run_scope.py limit for now

    # Load CLIP
    print("Loading CLIP model ...")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    # Get interpolator class to get hparam combos
    interpolator_cls = get_interpolator(args.interpolation_method.lower())
    hparam_combos = get_all_hparam_combinations(interpolator_cls)

    # Construct base dir like in main.py
    model_dir = sanitize_for_path(args.model_name)
    top_level_dir = os.path.join(
        args.exp_dir,
        model_dir,
        args.interpolation_method,
        f"steps_{args.num_inference_steps}",
        f"seed_{args.seed}"
    )
    print(f"Looking for images under: {top_level_dir}")

    results = []

    for idx, row in df.iterrows():
        prompt_schedule_list = ast.literal_eval(row["schedule"])
        if not prompt_schedule_list:
            continue
        final_prompt = prompt_schedule_list[-1]

        scope_clip_scores = {}
        best_scope_clip_score = float("-inf")
        best_suffix = None
        best_path = None
        normal_clip_score = None

        row_folder = os.path.join(top_level_dir, f"row_{idx}")

        for combo in hparam_combos:
            combo["seed"] = args.seed  # Match main.py
            combo_copy = combo.copy()
            if "interpolation_period" not in combo_copy:
                continue
            interpolation_period = combo_copy.pop("interpolation_period")

            suffix_parts = [f"{k}_{v}" for k, v in combo.items() if k != "seed"]
            suffix_str = "_".join(suffix_parts)

            final_out_dir = os.path.join(row_folder, suffix_str)
            baseline_path = os.path.join(final_out_dir, "baseline.png")
            scope_path = os.path.join(final_out_dir, "scope.png")
            schedule_path = os.path.join(final_out_dir, "prompt_schedule.txt")

            if not (os.path.exists(baseline_path) and os.path.exists(scope_path) and os.path.exists(schedule_path)):
                continue

            if normal_clip_score is None:
                normal_clip_score = compute_clip_similarity(
                    clip_model, preprocess, baseline_path, final_prompt, device
                )

            scope_score = compute_clip_similarity(
                clip_model, preprocess, scope_path, final_prompt, device
            )
            scope_clip_scores[suffix_str] = scope_score

            if scope_score > best_scope_clip_score:
                best_scope_clip_score = scope_score
                best_suffix = suffix_str
                best_path = scope_path

        if normal_clip_score is not None and best_scope_clip_score > float("-inf"):
            results.append({
                "image_id": idx,
                "normal_clip_score": normal_clip_score,
                "best_scope_clip_score": best_scope_clip_score,
                "difference": best_scope_clip_score - normal_clip_score,
                "best_path": best_path,
                "best_suffix": best_suffix,
                "scope_clip_scores": scope_clip_scores
            })

    # Save JSON directly inside top-level directory
    output_json_path = os.path.join(top_level_dir, "clip_scores.json")
    print(f"\nProcessed {len(results)} rows. Saving to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()