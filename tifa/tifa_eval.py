from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
import openai
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env (ensure your API key is stored there)
load_dotenv()

# # Initialize the OpenAI client
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

openai.api_key = os.getenv('OPENAI_API_KEY')
unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
vqa_model = VQAModel("mplug-large")
    

# Define base folder path
base_folder = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump"
experiment_subfolder = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/exp_dump/nlerp_model_stabilityai-stable-diffusion-2-1-base/num_inference_50_TEMP_1.0_STEP_SIZE_5_SEED_42/prompt_exp_V1_filter_advanced_none_basic_Spatial_Relation_num_prompts_30_filter_Num_Nouns"
full_path = os.path.join(base_folder, experiment_subfolder)

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

    print(last_prompt)
    exit()
    


# Generate questions with GPT-3.5-turbo
gpt3_questions = get_question_and_answers(text)
    
# Filter questions with UnifiedQA
filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
    
# See the questions
print(filtered_questions)

# calucluate TIFA score
result_normal = tifa_score_single(vqa_model, filtered_questions, img_path_normal)
result_scope = tifa_score_single(vqa_model, filtered_questions, img_path_scope)
# print(f"TIFA score is {result['tifa_score']}")   # 0.33
# print(result)

print(result_normal)
print()
print(result_scope)
import json
# convert to json format and save as a json file
# normal_json = json.dumps(result_normal, indent=4)
# scope_json = json.dumps(result_scope, indent=4)
with open('normal.json', 'w') as f:
    json.dump(result_normal, f, indent=4)

with open('scope.json', 'w') as f:
    json.dump(result_scope, f, indent=4)