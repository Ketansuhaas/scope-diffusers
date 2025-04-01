import argparse
import ast
import os
import random
import gc

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline

from interpolator.interpolator import get_interpolator
from helpers import build_step_callback, get_all_hparam_combinations

def encode_prompt_schedule(pipe, prompts, device):
    prompt_embeddings = []
    for prompt in prompts:
        with torch.no_grad():
            embeds = pipe._encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
        prompt_embeddings.append(embeds)
    return torch.stack(prompt_embeddings, dim=0)

def sanitize_for_path(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")

def run_pipeline(
    csv_path: str,
    model_name: str,
    num_inference_steps: int,
    seed: int,
    interpolation_method: str,
    exp_dir: str = "exp_dump/eval_output",
    hf_cache_dir: str = "./",
):
    print(f"\n--- run_pipeline() ---")
    print(f"model_name           = {model_name}")
    print(f"num_inference_steps  = {num_inference_steps}")
    print(f"seed                 = {seed}")
    print(f"interpolation_method = {interpolation_method}")
    print(f"---------------------\n")

    df = pd.read_csv(csv_path)
    # df = df.head(20)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir = hf_cache_dir
    ).to(device)

    os.makedirs(exp_dir, exist_ok=True)
    interpolator_cls = get_interpolator(interpolation_method.lower())
    hparam_combos = get_all_hparam_combinations(interpolator_cls)

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

        try:
            prompt_embeddings = encode_prompt_schedule(pipe, prompt_schedule_list, device)

            # Generate baseline only once per row
            with torch.no_grad():
                torch.manual_seed(seed)
                baseline_out = pipe(
                    prompt_embeds=prompt_embeddings[-1][1].unsqueeze(0),
                    neg_prompt_embeds=prompt_embeddings[-1][0].unsqueeze(0),
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps
                )
            baseline_image = baseline_out.images[0]

            row_dir = os.path.join(top_level_dir, f"row_{idx}")
            os.makedirs(row_dir, exist_ok=True)
            baseline_image.save(os.path.join(row_dir, "baseline.png"))

            with open(os.path.join(row_dir, "prompt_schedule.txt"), "w") as f:
                f.write(str(prompt_schedule_list))

            del baseline_out, baseline_image
            
            for combo in hparam_combos:
                combo["seed"] = seed
                torch.manual_seed(seed)
                
                combo_copy = combo.copy()
                interpolation_period = combo_copy.pop("interpolation_period")
                interpolator = interpolator_cls(
                    embeddings=prompt_embeddings,
                    interpolation_period=interpolation_period,
                    device=device,
                    **combo_copy
                )
                step_callback = build_step_callback(interpolator)

                with torch.no_grad():
                    initial_embedding = interpolator(0)
                    neg_embeds = initial_embedding[0].unsqueeze(0)
                    pos_embeds = initial_embedding[1].unsqueeze(0)

                    torch.manual_seed(seed)
                    output_interp = pipe(
                        prompt_embeds=pos_embeds,
                        neg_prompt_embeds=neg_embeds,
                        num_images_per_prompt=1,
                        num_inference_steps=num_inference_steps,
                        callback_on_step_end=step_callback,
                        callback_on_step_end_tensor_inputs=["prompt_embeds"]
                    )
                scope_image = output_interp.images[0]

                suffix_parts = [f"{k}_{v}" for k, v in combo.items() if k != "seed"]
                suffix_str = "_".join(suffix_parts)
                final_out_dir = os.path.join(row_dir, suffix_str)
                os.makedirs(final_out_dir, exist_ok=True)
                scope_image.save(os.path.join(final_out_dir, "scope.png"))

                del scope_image, output_interp, pos_embeds, neg_embeds, interpolator, step_callback
                torch.cuda.empty_cache()
                gc.collect()

            del prompt_embeddings
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing row {idx} with seed {seed} and params {combo}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

def main():
    parser = argparse.ArgumentParser(description="Run interpolation-based scope diffusion.")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpolation_method", type=str, default="nlerp_og")
    parser.add_argument("--exp_dir", type=str, default="exp_dump/debug")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--hf_cache_dir",type=str, default="./")
    args = parser.parse_args()
    
    run_pipeline(
        csv_path=args.csv_path,
        model_name=args.model_name,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        interpolation_method=args.interpolation_method,
        exp_dir=args.exp_dir,
        hf_cache_dir=args.hf_cache_dir
    )

if __name__ == "__main__":
    main()
