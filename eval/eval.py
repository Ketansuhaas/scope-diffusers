import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sd_pipe
from ezcolorlog import root_logger as logger
import os
from config import Config
import pandas as pd
from genai_prompts.dataset import GenAIDataset
from genai_prompts.filter_dataset import filter_dataset
from genai_prompts.scope_prompt_gen import get_progressive_prompts
import json
import torch
import re
from PIL import Image
import random
from prompts.system_prompts.sys_prompts import system_prompts
import matplotlib.pyplot as plt

def get_scope_image(prompt_schedule, scope_sd_pipe, config):
    torch.manual_seed(config.SEED)
    image = scope_sd_pipe(
            interpolation_technique=config.INTERPOLATION_TECHNIQUE,
            prompt_schedule=prompt_schedule,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            callback=None,
            callback_steps=1,
            temperature=config.TEMPERATURE,
    ).images[0]

    return image


def get_normal_sd_image(prompt, sd_pipe, config):
    torch.manual_seed(config.SEED)
    image = sd_pipe(
            prompt,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            callback=None,
            callback_steps=1,
    ).images[0]

    return image



config = Config()

# print configuration values
logger.info(f"System Prompt: {config.SYSTEM_PROMPT}")

# create experiment directories based on configuration
exp_dir = config.create_exp_name()
logger.info(f"Experiment Directory: {exp_dir}")
# rewrite the experiment directory
os.makedirs(exp_dir, exist_ok=True)

sys_prompt = system_prompts[config.SYSTEM_PROMPT]

if config.PROVIDE_PROMPTS:

    # Loading the dataset
    if os.path.exists(config.GENAI_CSV_PATH):  # Corrected line
        dataset = pd.read_csv(config.GENAI_CSV_PATH)
        logger.info(f"Dataset loaded with columns: {dataset.columns}")
    else:
        dataset = GenAIDataset()  # Create an instance of the dataset if not loaded from a CSV
        dataset = dataset.create_dataframe()  # Create a DataFrame from the dataset
        logger.info(f"Dataset created with columns: {dataset.columns}")
        
        # Save the dataset to a CSV file
        dataset.to_csv(config.GENAI_CSV_PATH)
        logger.info(f"Dataset saved to {config.GENAI_CSV_PATH}")

    logger.info(f"Filtering dataset with tags: {config.FILTER_TAGS}, before filtering: {len(dataset)} entries")
    # Filter the dataset based on the configuration
    dataset = filter_dataset(dataset, config.FILTER_TAGS, config.NUM_FILTER, config.FILTER_BY)
    logger.info(f"Filtered dataset size: {len(dataset)} entries")

    prompt_exp = exp_dir.split("/prompt_exp_")[-1]
    prompt_exp = f"prompt_exp_{prompt_exp}"
    scope_prompts_path = os.path.join(f"prompt_dump/{prompt_exp}", "scope_prompts.json")
    if os.path.exists(scope_prompts_path):
        logger.info(f"Loading existing prompts from {scope_prompts_path}")
    else:
        # create scheduled prompts
        logger.info(f"Generating Prompts for {len(dataset)} entries, using system prompt: {sys_prompt}")
        # prompts = dataset['Prompt'].tolist()
        logger.info(f"Generating Prompts for index: {dataset.index}")

        json_format_responses = []
        for index in dataset.index:
            prompt = dataset['Prompt'][index]
            progressive_prompts = get_progressive_prompts(
                sys_prompt, prompt)
            json_format_responses.append({
            index: {
                "initial_prompt": prompt,
                "progressive_prompts": progressive_prompts
            }
        })

        os.makedirs(os.path.dirname(scope_prompts_path), exist_ok=True)  # Create directory if it doesn't exist

        # Save the results into a JSON file for future use
        with open(scope_prompts_path, 'w') as f:
            json.dump(json_format_responses, f, indent=4)

        logger.info(f"Results saved in {scope_prompts_path}")


else:
    logger.info(f"Generating Prompts based on System Prompt: {config.SYSTEM_PROMPT}")
    dataset = None
    scope_prompts_path= None

with open(scope_prompts_path, 'r') as f:
    prompts = json.load(f)


# set pipes
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
scope_sd_pipe = sdp_scope.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache')
normal_sd_pipe = sd_pipe.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True,cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache')


device = "cuda" if torch.cuda.is_available() else "cpu"
scope_sd_pipe.to(device)
normal_sd_pipe.to(device)

for prompt in prompts:

    prompt_id = list(prompt.keys())[0]
    initial_prompt = list(prompt.values())[0]['initial_prompt']
    progressive_prompts = list(prompt.values())[0]['progressive_prompts']
    match = re.search(r'\[([\s\S]*?)\]', progressive_prompts)
    prompt_schedule_match = match.group(1)
    prompt_schedule_list = [prompt.strip().strip('"') for prompt in prompt_schedule_match.split('",')]
    prompt_schedule_list = [prompt for prompt in prompt_schedule_list if prompt]  # Remove empty prompts


    prompt_schedule = []
    step_size = config.STEP_SIZE
                 
    for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
        prompt_schedule.append((stage_id*step_size,p))

    final_prompt = prompt_schedule_list[-1]  # Get the final prompt

    logger.info(f"\nPrompt ID: {prompt_id}")
    logger.info(f"Initial Prompt: {initial_prompt}")
    logger.info(f"Progressive Prompts: {progressive_prompts}")
    logger.info(f"Final Prompt: {final_prompt}")


    # running scope
    scope_image = get_scope_image(prompt_schedule, scope_sd_pipe, config)

    # running normal sd
    normal_image = get_normal_sd_image(final_prompt, normal_sd_pipe, config)

    # save image inot each index folder for both scope and normal images
    index_dir = os.path.join(exp_dir, prompt_id)
    os.makedirs(index_dir, exist_ok=True)

    # save images seperately
    scope_image.save(os.path.join(index_dir, "scope_image.png"))
    normal_image.save(os.path.join(index_dir, "normal_image.png"))

    # save prompt schedule to a text file
    with open(os.path.join(index_dir, "prompt_schedule.txt"), 'w') as f:
        f.write(progressive_prompts)
        
    # Randomly switch left and right images
    if random.choice([True, False]):
        images = [normal_image, scope_image]  # Switch images
        switch_info = "normal-left_scope-right"
    else:
        images = [scope_image, normal_image]  # Original order
        switch_info = "scope-left_normal-right"

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns for side-by-side images

    # Add white space between subplots (adjust wspace to set the separation width)
    fig.subplots_adjust(wspace=0.05)  # Adjust as needed for desired separation

    # Display each image in its subplot
    for ax, img, title in zip(axes, images, ["Image 1", "Image 2"]):
        ax.imshow(img)
        ax.axis("off")  # Remove axes for a clean look

    # Add a blank white background for the figure to fill in any gaps
    fig.patch.set_facecolor('white')

    # Save the plot to a file
    grid_image_filename = "1v1.png" #f"grid_image_{switch_info}.png"
    plt.savefig(os.path.join(index_dir, grid_image_filename), bbox_inches='tight', pad_inches=0.1)

    # Close the plot to free up memory
    plt.close(fig)
