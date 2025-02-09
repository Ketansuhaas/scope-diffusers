from t2v_metrics.t2v_metrics import VQAScore
clip_flant5_score = VQAScore(model='clip-flant5-xxl') # our recommended scoring model
import os
import ast
import re
from PIL import Image
import torch
import clip
from collections import Counter

# Define base folder path
base_folder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump"
experiment_subfolder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump/iccv_5_stages"
full_path = os.path.join(base_folder, experiment_subfolder)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize results
results = []

# Regular expression pattern to extract Python-like lists
list_pattern = re.compile(r"\[\s*(?:\".*?\"(?:,|\s)*)+\s*\]", re.DOTALL)

SEEDS = [0]
STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
STD_DEV = [3, 5]
results = []

# Iterate over all subfolders (image IDs)
for image_id in range(1600):
    print(f"Processing image {image_id}...")
    # if len(results) >= 30:  # Limit to 30 prompts
    #     break

    best_scope_clip_score = 0
    scope_clip_scores = {}
    best_step = None
    best_path = None
    normal_clip_score = None  # Initialize normal clip score
    best_std_dev = None

    for seed in SEEDS:
        for step in STEPS:
            for std_dev in STD_DEV:
                
                image_folder = os.path.join(full_path, f"{image_id}/seed_{seed}/step_size_{step}_{std_dev}")

                # Validate folder contents
                prompt_schedule_path = os.path.join(image_folder, "prompt_schedule.txt")
                normal_image_path = os.path.join(image_folder, "normal_image.png")
                scope_image_path = os.path.join(image_folder, "scope_image.png")
                if not (os.path.exists(prompt_schedule_path) and os.path.exists(normal_image_path) and os.path.exists(scope_image_path)):
                    continue

                # Read and validate prompt schedule
                try:
                    with open(prompt_schedule_path, "r") as file:
                        content = file.read().strip()
                    prompt_schedule = ast.literal_eval(content)
                    if not isinstance(prompt_schedule, list) or len(prompt_schedule) == 0:
                        raise ValueError("Invalid prompt schedule.")
                except Exception as e:
                    print(f"Skipping {prompt_schedule_path} due to error: {e}")
                    continue

                last_prompt = prompt_schedule[-1]

                # Load and preprocess images
                try:
                    text = clip.tokenize([last_prompt], context_length=77, truncate=True).to(device)
                    normal_clip_score = float(clip_flant5_score(images=[normal_image_path], texts=[text]).cpu().numpy()[0][0])
                    scope_clip_score = float(clip_flant5_score(images=[scope_image_path], texts=[text]).cpu().numpy()[0][0])

                    #  Track scores
                    scope_clip_scores[f"seed_{seed}_step_{step}_std_dev_{std_dev}"] = scope_clip_score
                    if scope_clip_score > best_scope_clip_score:
                        best_scope_clip_score = scope_clip_score
                        best_step = step
                        best_path = scope_image_path
                        best_std_dev = std_dev


                except Exception as e:
                    print(f"Error processing images for {image_folder}: {e}")
                    continue

    # Append results for valid image IDs
    if best_step is not None:
        results.append({ 
            "image_id": image_id,
            "normal_vqa_score": normal_clip_score,
            "best_scope_vqa_score": best_scope_clip_score,
            "difference": best_scope_clip_score - normal_clip_score,
            "best_path": best_path,
            "scope_vqa_scores": scope_clip_scores,
            "best_step": best_step,
            "best_std_dev": best_std_dev
        })
    # save results to a json file
    import json
    with open('vqa_scores_iccv_5stages.json', 'w') as f:
        json.dump(results, f, indent=4)

# Step size comparison
step_size_scores = {}
for result in results:
    for key, value in result["scope_vqa_scores"].items():
        step = int(key.split("_")[-1])  # Extract step size
        if step not in step_size_scores:
            step_size_scores[step] = 0
        if value > result["normal_vqa_score"]:
            step_size_scores[step] += 1

# Best step occurrences
best_step_counter = Counter(result["best_step"] for result in results)

# Determine the step size with the maximum occurrences of best_step
most_common_best_step, max_occurrences = best_step_counter.most_common(1)[0]

# Print results
print("Step size scores:", step_size_scores)
print("Occurrences of best_step for all step sizes:")
for step, count in sorted(best_step_counter.items()):
    if step == most_common_best_step:
        print(f"Step {step}: {count} (Max Occurrences)")
    else:
        print(f"Step {step}: {count}")
print(f"Most common best step: {most_common_best_step} with {max_occurrences} occurrences")

# exit()

# Output results
scope_scores = []
normal_scores = []
for result in results:
    # print(f"Image ID: {result['image_id']}")
    # print(f"  CLIP Score for 'normal_image.png': {result['normal_clip_score']}")
    # print(f"  CLIP Score for 'scope_image.png': {result['best_scope_clip_score']}")
    normal_scores.append(result['normal_vqa_score'])
    scope_scores.append(result['best_scope_vqa_score'])

import numpy as np
print(f"average vqa Score (SCoPE, based on best step_size): {np.mean(np.array(scope_scores))}")
print(f"average vqa Score (Normal): {np.mean(np.array(normal_scores))}")



# Compute statistics
# Compare performance and calculate average improvement
if results:
    better_scope = []
    for result in results:
        if result['best_scope_vqa_score'] > result['normal_vqa_score']:
            better_scope.append(result['best_scope_vqa_score'] - result['normal_vqa_score'])

    total_images = len(results)
    num_better_scope = len(better_scope)
    num_better_normal = total_images - num_better_scope

    # Compute average improvement for scope images
    avg_improvement = np.mean(better_scope) if better_scope else 0.0

    # Output the comparison
    print(f"\n--- Performance Comparison ---")
    print(f"Scope images perform better for {num_better_scope} out of {total_images} images ({(num_better_scope / total_images) * 100:.2f}%).")
    print(f"Normal images perform better for {num_better_normal} out of {total_images} images ({(num_better_normal / total_images) * 100:.2f}%).")
    print(f"Average improvement of scope images over normal images: {avg_improvement:.4f}")
else:
    print("No results to compare performance.")

