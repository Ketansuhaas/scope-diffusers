import os
import ast
import re
from PIL import Image
import torch
import clip
from collections import Counter

# Define base folder path
base_folder = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump"
experiment_subfolder = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump/iccv"
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
results = []

# Iterate over all subfolders (image IDs)
for image_id in os.listdir(full_path):
    # if len(results) >= 30:  # Limit to 30 prompts
    #     break

    best_scope_clip_score = 0
    scope_clip_scores = {}
    best_step = None
    best_path = None
    normal_clip_score = None  # Initialize normal clip score

    for seed in SEEDS:
        for step in STEPS:
            image_folder = os.path.join(full_path, f"{image_id}/seed_{seed}/step_size_{step}")

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
                normal_image = preprocess(Image.open(normal_image_path)).unsqueeze(0).to(device)
                scope_image = preprocess(Image.open(scope_image_path)).unsqueeze(0).to(device)
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

                # Track scores
                scope_clip_scores[f"seed_{seed}_step_{step}"] = scope_clip_score
                if scope_clip_score > best_scope_clip_score:
                    best_scope_clip_score = scope_clip_score
                    best_step = step
                    best_path = scope_image_path

            except Exception as e:
                print(f"Error processing images for {image_folder}: {e}")
                continue

    # Append results for valid image IDs
    if best_step is not None:
        results.append({
            "image_id": image_id,
            "normal_clip_score": normal_clip_score,
            "best_scope_clip_score": best_scope_clip_score,
            "difference": best_scope_clip_score - normal_clip_score,
            "best_path": best_path,
            "scope_clip_scores": scope_clip_scores,
            "best_step": best_step
        })

# Step size comparison
step_size_scores = {}
for result in results:
    for key, value in result["scope_clip_scores"].items():
        step = int(key.split("_")[-1])  # Extract step size
        if step not in step_size_scores:
            step_size_scores[step] = 0
        if value > result["normal_clip_score"]:
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

# save results to a json file
import json
with open('clip_scores.json', 'w') as f:
    json.dump(results, f, indent=4)

# Output results
scope_scores = []
normal_scores = []
for result in results:
    # print(f"Image ID: {result['image_id']}")
    # print(f"  CLIP Score for 'normal_image.png': {result['normal_clip_score']}")
    # print(f"  CLIP Score for 'scope_image.png': {result['best_scope_clip_score']}")
    normal_scores.append(result['normal_clip_score'])
    scope_scores.append(result['best_scope_clip_score'])

import numpy as np
print(f"average CLIP Score (SCoPE, based on best step_size): {np.mean(np.array(scope_scores))}")
print(f"average CLIP Score (Normal): {np.mean(np.array(normal_scores))}")



# Compute statistics
# Compare performance and calculate average improvement
if results:
    better_scope = []
    for result in results:
        if result['best_scope_clip_score'] > result['normal_clip_score']:
            better_scope.append(result['best_scope_clip_score'] - result['normal_clip_score'])

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


# Step-wise accuracy performance comparisons
step_wise_accuracy = {step: 0 for step in STEPS}
total_images = len(results)

for result in results:
    for key, value in result["scope_clip_scores"].items():
        step = int(key.split("_")[-1])  # Extract step size
        if value > result["normal_clip_score"]:
            step_wise_accuracy[step] += 1

# Convert to percentages
step_wise_accuracy_percentage = {
    step: (count / total_images) * 100 for step, count in step_wise_accuracy.items()
}

# Print step-wise accuracies
print("\n--- Step-wise Accuracy Comparisons ---")
print("Step: Percentage of cases where SCoPE outperformed Normal:")
for step, accuracy in sorted(step_wise_accuracy_percentage.items()):
    print(f"Step {step}: {accuracy:.2f}%")



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

