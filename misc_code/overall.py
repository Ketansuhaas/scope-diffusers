import json
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Step 1: Load your updated VQA comparison JSON
with open(
    "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump/sdc/stabilityai_stable-diffusion-2-1/spherical_de_casteljau/steps_50/seed_42/vqa_scores.json",
    "r",
) as f:
    vqa_data = json.load(f)

# Step 2: Map scores by image_id
normal_scores = {entry["image_id"]: entry["normal_vqa_score"] for entry in vqa_data}
scope_scores = {entry["image_id"]: entry["best_scope_vqa_score"] for entry in vqa_data}

# Step 3: print mean values of both
normal_mean = np.mean(list(normal_scores.values()))
scope_mean = np.mean(list(scope_scores.values()))
print(f"Normal Mean: {normal_mean}")
print(f"Scope Mean: {scope_mean}")

# Step 4: print win rate
normal_wins = sum(1 for k in normal_scores if normal_scores[k] > scope_scores[k])
scope_wins = sum(1 for k in scope_scores if scope_scores[k] > normal_scores[k])
total = len(normal_scores)
print(f"Normal Wins: {normal_wins} ({normal_wins / total:.2%})")
print(f"Scope Wins: {scope_wins} ({scope_wins / total:.2%})")
