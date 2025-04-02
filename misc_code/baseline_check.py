import json
from collections import defaultdict
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Step 1: Load your updated VQA comparison JSON
with open("/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump/debug/stabilityai_stable-diffusion-2-1/nlerp_og/steps_50/seed_42/vqa_scores.json", "r") as f:
    vqa_data = json.load(f)

# Step 2: Map scores by image_id
normal_scores = {entry["image_id"]: entry["normal_vqa_score"] for entry in vqa_data}
scope_scores = {entry["image_id"]: entry["best_scope_vqa_score"] for entry in vqa_data}

# Step 3: Load GenAI-Bench dataset
dataset = load_dataset("BaiqiL/GenAI-Bench")["train"]

# Step 4: Aggregate normal and scope scores per tag
tag_normal_scores = defaultdict(list)
tag_scope_scores = defaultdict(list)

for example in tqdm(dataset, desc="Processing dataset"):
    image_id = example["Index"]
    if image_id not in normal_scores or image_id not in scope_scores:
        continue

    tags = example["Tags"].get("basic", []) + example["Tags"].get("advanced", [])
    normal = normal_scores[image_id]
    scope = scope_scores[image_id]

    for tag in tags:
        tag_normal_scores[tag].append(normal)
        tag_scope_scores[tag].append(scope)

# Step 5: Compute averages
tag_avg_scores = {}
for tag in sorted(set(tag_normal_scores.keys()).union(tag_scope_scores.keys())):
    normal_avg = np.mean(tag_normal_scores[tag]) if tag in tag_normal_scores else float('nan')
    scope_avg = np.mean(tag_scope_scores[tag]) if tag in tag_scope_scores else float('nan')
    tag_avg_scores[tag] = (normal_avg, scope_avg)

# Step 6: Print comparison
print("\nTag-wise VQA Score Comparison (Normal vs. Best SCoPE):\n")
print(f"{'Tag':<20} {'Normal':>10} {'SCoPE':>10} {'Delta':>10}")
print("-" * 50)
for tag, (normal, scope) in tag_avg_scores.items():
    delta = scope - normal
    print(f"{tag:<20} {normal:>10.4f} {scope:>10.4f} {delta:>10.4f}")