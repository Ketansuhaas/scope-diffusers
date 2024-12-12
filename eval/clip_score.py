import os
import ast
import re
from PIL import Image
import torch
import clip

# Define base folder path
base_folder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump"
experiment_subfolder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump/nlerp_model_stabilityai-stable-diffusion-2-1-base/num_inference_50_TEMP_1.0_STEP_SIZE_5_SEED_42/prompt_exp_V5_filter_advanced_none_basic_Part_Relation_num_prompts_10_filter_Num_Nouns"
full_path = os.path.join(base_folder, experiment_subfolder)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize results
results = []

# Regular expression pattern to extract Python-like lists
list_pattern = re.compile(r"\[\s*(?:\".*?\"(?:,|\s)*)+\s*\]", re.DOTALL)

# Iterate over all subfolders (image IDs)
for image_id in os.listdir(full_path):
    image_folder = os.path.join(full_path, image_id)

    # Check if the folder contains the necessary files
    prompt_schedule_path = os.path.join(image_folder, "prompt_schedule.txt")
    normal_image_path = os.path.join(image_folder, "normal_image.png")
    scope_image_path = os.path.join(image_folder, "scope_image.png")
    
    if not (os.path.exists(prompt_schedule_path) and os.path.exists(normal_image_path) and os.path.exists(scope_image_path)):
        continue  # Skip folders without required files

    # Read and parse the prompt schedule
    with open(prompt_schedule_path, "r") as file:
        content = file.read().strip()
        if not content:
            print(f"File {prompt_schedule_path} is empty. Skipping.")
            continue

        # Extract the list using regex
        match = list_pattern.search(content)
        if not match:
            print(f"File {prompt_schedule_path} does not contain a valid list. Skipping.")
            continue

        # Safely parse the extracted list
        try:
            prompt_schedule = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            print(f"File {prompt_schedule_path} contains an invalid list structure. Skipping.")
            continue

    # Extract the last prompt
    if not isinstance(prompt_schedule, list) or len(prompt_schedule) == 0:
        print(f"File {prompt_schedule_path} does not contain a valid prompt list. Skipping.")
        continue

    last_prompt = prompt_schedule[-1]

    # Load images
    normal_image = preprocess(Image.open(normal_image_path)).unsqueeze(0).to(device)
    scope_image = preprocess(Image.open(scope_image_path)).unsqueeze(0).to(device)

    # Prepare the prompt
    text = clip.tokenize([last_prompt], context_length=77, truncate=True).to(device)

    # Compute CLIP scores
    with torch.no_grad():
        normal_image_features = model.encode_image(normal_image)
        scope_image_features = model.encode_image(scope_image)
        text_features = model.encode_text(text)

        # Normalize features
        normal_image_features /= normal_image_features.norm(dim=-1, keepdim=True)
        scope_image_features /= scope_image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        normal_clip_score = (normal_image_features @ text_features.T).item()
        scope_clip_score = (scope_image_features @ text_features.T).item()

    # Append results
    results.append({
        "image_id": image_id,
        "normal_clip_score": normal_clip_score,
        "scope_clip_score": scope_clip_score
    })


# Output results
scope_scores = []
normal_scores = []
for result in results:
    print(f"Image ID: {result['image_id']}")
    print(f"  CLIP Score for 'normal_image.png': {result['normal_clip_score']}")
    print(f"  CLIP Score for 'scope_image.png': {result['scope_clip_score']}")
    normal_scores.append(result['normal_clip_score'])
    scope_scores.append(result['scope_clip_score'])

import numpy as np
print(f"average CLIP Score (SCoPE): {np.mean(np.array(scope_scores))}")
print(f"average CLIP Score (Normal): {np.mean(np.array(normal_scores))}")

# Compute statistics
# Compare performance and calculate average improvement
if results:
    better_scope = []
    for result in results:
        if result['scope_clip_score'] > result['normal_clip_score']:
            better_scope.append(result['scope_clip_score'] - result['normal_clip_score'])

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


# Find top 5 images where scope_image improves the most
if better_scope:
    # Sort results by the improvement in descending order
    results_with_improvement = [
        {
            "image_id": result["image_id"],
            "improvement": result["scope_clip_score"] - result["normal_clip_score"],
            "normal_image_path": os.path.join(full_path, result["image_id"], "normal_image.png"),
            "scope_image_path": os.path.join(full_path, result["image_id"], "scope_image.png")
        }
        for result in results
        if result["scope_clip_score"] > result["normal_clip_score"]
    ]
    results_with_improvement = sorted(results_with_improvement, key=lambda x: -x["improvement"])

    # Get the top 5 results
    top_5_results = results_with_improvement[:5]

    # Print the paths and improvements for the top 5
    print("\n--- Top 5 Images with Best Scope Improvement ---")
    for i, result in enumerate(top_5_results, 1):
        print(f"Rank {i}:")
        print(f"  Image ID: {result['image_id']}")
        print(f"  Improvement: {result['improvement']:.4f}")
        print(f"  Normal Image Path: {result['normal_image_path']}")
        print(f"  Scope Image Path: {result['scope_image_path']}")
else:
    print("\nNo images where scope_image performed better.")
