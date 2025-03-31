import argparse
import ast
import os
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline

from interpolator.interpolator import get_interpolator
from helpers import build_step_callback, get_all_hparam_combinations


def encode_prompt_schedule(pipe, prompts, device):
    """
    Encodes a list of text prompts into embeddings with classifier-free guidance on.
    Returns a stacked tensor of shape [num_prompts, 2, seq_len, embed_dim],
    where dimension 1 is (negative_embeds, positive_embeds).
    """
    prompt_embeddings = []
    for prompt in prompts:
        embeds = pipe._encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        prompt_embeddings.append(embeds)
    return torch.stack(prompt_embeddings, dim=0)


def sanitize_for_path(name: str) -> str:
    """
    Replaces slashes and spaces in strings to ensure valid, clean directory names.
    """
    return name.replace("/", "_").replace(" ", "_")


def run_pipeline(
    csv_path: str,
    model_name: str,
    num_inference_steps: int,
    seed: int,
    interpolation_method: str,
    exp_dir: str = "exp_dump/eval_output"
):
    """
    Main pipeline to load the model, run interpolation, and save outputs
    with a clean directory structure.
    """

    print(f"\n--- run_pipeline() ---")
    print(f"model_name           = {model_name}")
    print(f"num_inference_steps  = {num_inference_steps}")
    print(f"seed                 = {seed}")
    print(f"interpolation_method = {interpolation_method}")
    print(f"---------------------\n")

    # Load CSV and prepare device
    df = pd.read_csv("/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/genai_dataset_schedules_fixed.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)

    os.makedirs(exp_dir, exist_ok=True)

    interpolator_cls = get_interpolator(interpolation_method.lower())

    # Retrieve possible hyperparam combos (example usage)
    hparam_combos = get_all_hparam_combinations(interpolator_cls)

    # # For demonstration, limit to first 10
    # df = df.head(4)

    # Create top-level output directory for this configuration
    #   exp_dir/<model_name_sanitized>/<interpolation_method>/steps_<num_inference_steps>/seed_<seed>/
    model_dir = sanitize_for_path(model_name)
    top_level_dir = os.path.join(
        exp_dir,
        model_dir,
        interpolation_method,
        f"steps_{num_inference_steps}",
        f"seed_{seed}"
    )
    os.makedirs(top_level_dir, exist_ok=True)

    for idx, row in df.iterrows():
        prompt_schedule_list = ast.literal_eval(row["schedule"])

        # Each row can have multiple hparam combos
        for combo in hparam_combos:
            combo["seed"] = seed
            torch.manual_seed(combo["seed"])

            # Encode entire prompt schedule into embeddings
            prompt_embeddings = encode_prompt_schedule(pipe, prompt_schedule_list, device)

            # Build the interpolator
            combo_copy = combo.copy()
            interpolation_period = combo_copy.pop("interpolation_period")
            interpolator = interpolator_cls(
                embeddings=prompt_embeddings,
                interpolation_period=interpolation_period,
                device=device,
                **combo_copy
            )

            step_callback = build_step_callback(interpolator)

            # Interpolate first step's embeddings
            initial_embedding = interpolator(0)
            # initial_embedding shape = [2, seq_len, embed_dim]
            neg_embeds = initial_embedding[0].unsqueeze(0)  # [1, seq_len, embed_dim]
            pos_embeds = initial_embedding[1].unsqueeze(0)  # [1, seq_len, embed_dim]

            # Generate with interpolation
            output_interp = pipe(
                prompt_embeds=pos_embeds,
                neg_prompt_embeds=neg_embeds,
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["prompt_embeds"]
            )
            scope_image = output_interp.images[0]

            # Baseline: Just prompt with final prompt
            torch.manual_seed(combo["seed"])  # reset seed
            output_baseline = pipe(
                prompt_embeds=prompt_embeddings[-1][1].unsqueeze(0),  # Use the last prompt's positive embedding
                neg_prompt_embeds=prompt_embeddings[-1][0].unsqueeze(0),  # Use the last prompt's negative embedding
                num_images_per_prompt=1,
                num_inference_steps=num_inference_steps
            )
            normal_image = output_baseline.images[0]

            # Build subdir for this row + hparam
            # e.g. row_3 / something_{val}...
            # You could incorporate more hparam details if you like
            row_dir = os.path.join(top_level_dir, f"row_{idx}")
            # For instance, you can create a subdir with leftover combo keys
            # or just keep them in the row_dir. If you want more structure, do:
            suffix_parts = []
            for k, v in combo.items():
                if k != "seed":
                    suffix_parts.append(f"{k}_{v}")
            # suffix_parts.append(f"period_{interpolation_period}")
            suffix_str = "_".join(suffix_parts)

            final_out_dir = os.path.join(row_dir, suffix_str)
            os.makedirs(final_out_dir, exist_ok=True)

            # Save images
            scope_image.save(os.path.join(final_out_dir, "scope.png"))
            normal_image.save(os.path.join(final_out_dir, "baseline.png"))

            # Save schedule
            with open(os.path.join(final_out_dir, "prompt_schedule.txt"), "w") as f:
                f.write(str(prompt_schedule_list))

            # Side-by-side figure
            if random.choice([True, False]):
                images = [normal_image, scope_image]
            else:
                images = [scope_image, normal_image]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.subplots_adjust(wspace=0.05)
            for ax, img in zip(axes, images):
                ax.imshow(img)
                ax.axis("off")
            fig.patch.set_facecolor('white')
            plt.savefig(os.path.join(final_out_dir, "comparison.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run interpolation-based scope diffusion.")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Model name or local path.")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--interpolation_method", type=str, default="nlerp_og",
                        help="Interpolation method (e.g. nlerp_og).")
    parser.add_argument("--exp_dir", type=str, default="exp_dump/eval_output",
                        help="Base directory to store output.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="CSV path with prompt schedules (same as run_scope).")     

    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv_path,
        model_name=args.model_name,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        interpolation_method=args.interpolation_method,
        exp_dir=args.exp_dir
    )


if __name__ == "__main__":
    main()