import json
from collections import defaultdict
import numpy as np

# Step 1: Load your updated VQA comparison JSON
with open("/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump/debug/stabilityai_stable-diffusion-2-1/nlerp_og/steps_50/seed_42/vqa_scores.json", "r") as f:
    vqa_data = json.load(f)

# Step 2: Aggregate scores per setting
setting_scores = defaultdict(list)
setting_best_counts = defaultdict(int)
total_prompts = len(vqa_data)

for entry in vqa_data:
    best_suffix = entry["best_suffix"]
    all_scores = entry["scope_vqa_scores"]

    for setting, score in all_scores.items():
        setting_scores[setting].append(score)

    setting_best_counts[best_suffix] += 1

# Step 3: Print table header
print(f"{'Model':<10} {'Setting':<30} {'Mean VQA':>10} {'% Best':>8}")
print("-" * 65)

# Step 4: Print results for each setting
for setting in sorted(setting_scores.keys()):
    scores = setting_scores[setting]
    mean_vqa = np.mean(scores)
    percent_best = 100 * setting_best_counts.get(setting, 0) / total_prompts
    print(f"{'SD2.1':<10} {setting:<30} {mean_vqa:>10.4f} {percent_best:>7.2f}%")