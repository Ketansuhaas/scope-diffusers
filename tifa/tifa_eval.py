from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
import openai
import re
import os
from dotenv import load_dotenv
import ast
from collections import Counter
import numpy as np
import random

np.random.seed(0)

# Load environment variables from .env (ensure your API key is stored there)
load_dotenv()

# # Initialize the OpenAI client
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

openai.api_key = os.getenv('OPENAI_API_KEY')
unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
vqa_model = VQAModel("mplug-large")
    

# Define base folder path
base_folder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump"
experiment_subfolder = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump/iccv_5_stages"
full_path = os.path.join(base_folder, experiment_subfolder)

# Regular expression pattern to extract Python-like lists
list_pattern = re.compile(r"\[\s*(?:\".*?\"(?:,|\s)*)+\s*\]", re.DOTALL)

SEEDS = [0]
STEPS = [1, 2, 3, 4, 5, 6, 7, 8]
STD_DEV = [3, 5]
results = []
count = 0

# Iterate over all subfolders (image IDs)
# Iterate over all subfolders (image IDs)
for image_id in range(10):
    print(f"Processing image {image_id}...")
    count += 1

    best_scope_tifa_score = 0
    scope_tifa_scores = {}
    best_step = None
    best_path = None
    normal_tifa_score = None  # Initialize normal clip score

    # try:
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

                # print(last_prompt)


                if normal_tifa_score is None:
                    gpt3_questions = get_question_and_answers(last_prompt)
                    # Filter questions with UnifiedQA
                    filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
                    result_normal = tifa_score_single(vqa_model, filtered_questions, normal_image_path)
                    normal_tifa_score = result_normal['tifa_score']    
                result_scope = tifa_score_single(vqa_model, filtered_questions, scope_image_path)
                scope_tifa_score = result_scope['tifa_score']


                scope_tifa_scores[f"seed_{seed}_step_{step}_std_dev_{std_dev}"] = scope_tifa_score
                if normal_tifa_score is not None:
                    print(f"Normal TIFA score: {normal_tifa_score}, Scope TIFA score: {scope_tifa_score} at step {step}, std_dev {std_dev} for image {image_id}")
                if scope_tifa_score > best_scope_tifa_score:
                    best_scope_tifa_score = scope_tifa_score
                    best_step = step
                    best_path = image_folder
                    best_scope_tifa_dets = result_scope
    
    if best_step is not None:
        print(f"Best scope TIFA score: {best_scope_tifa_score} at seed 0, step {best_step} for image {image_id}")
        print(f"Path: {best_path}")
        results.append({
            "image_id": image_id,
            "normal_tifa_score": normal_tifa_score,
            "best_scope_tifa_score": best_scope_tifa_score,
            "difference": best_scope_tifa_score - normal_tifa_score,
            "best_path": best_path,
            "scope_tifa_scores": scope_tifa_scores,
            "best_step": best_step,
            "best_scope_tifa_dets": best_scope_tifa_dets,
            "normal_tifa_dets": result_normal
        })

    # except Exception as e:
    #     print(f"Error processing {image_folder}: {e}")

        


# Step size comparison
step_size_scores = {}
for result in results:
    for key, value in result["scope_tifa_scores"].items():
        step = int(key.split("_")[-1])  # Extract step size
        if step not in step_size_scores:
            step_size_scores[step] = 0
        if value > result["normal_tifa_score"]:
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
with open('tifa_scores_sdxl.json', 'w') as f:
    json.dump(results, f, indent=4)


# Output results
scope_scores = []
normal_scores = []
for result in results:
    normal_scores.append(result['normal_tifa_score'])
    scope_scores.append(result['best_scope_tifa_score'])

import numpy as np
print(f"average TIFA Score (SCoPE, based on best step_size): {np.mean(np.array(scope_scores))}")
print(f"average TIFA Score (Normal): {np.mean(np.array(normal_scores))}")


# Compute statistics
# Compare performance and calculate average improvement
if results:
    better_scope = []
    for result in results:
        if result['best_scope_tifa_score'] > result['normal_tifa_score']:
            better_scope.append(result['best_scope_tifa_score'] - result['normal_tifa_score'])

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


# # Generate questions with GPT-3.5-turbo
# gpt3_questions = get_question_and_answers(text)
    
# # Filter questions with UnifiedQA
# filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
    
# # See the questions
# print(filtered_questions)

# # calucluate TIFA score
# result_normal = tifa_score_single(vqa_model, filtered_questions, img_path_normal)
# result_scope = tifa_score_single(vqa_model, filtered_questions, img_path_scope)
# # print(f"TIFA score is {result['tifa_score']}")   # 0.33
# # print(result)

# print(result_normal)
# print()
# print(result_scope)
# import json
# # convert to json format and save as a json file
# # normal_json = json.dumps(result_normal, indent=4)
# # scope_json = json.dumps(result_scope, indent=4)
# with open('normal.json', 'w') as f:
#     json.dump(result_normal, f, indent=4)

# with open('scope.json', 'w') as f:
#     json.dump(result_scope, f, indent=4)