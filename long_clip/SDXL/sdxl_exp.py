import sys
sys.path.append('..')
from diffusers import DiffusionPipeline
import torch
from open_clip_long import factory as open_clip
import torch.nn as nn
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from SDXL_img2img import image2image
import random
#-------------------------------------------------------------------------------------------------------


from SDXL_pipeline_prompt_interpolation import get_image as get_image_scope
from SDXL_pipeline_backup import get_image


base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 50
high_noise_frac = 0.8

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


def get_scope_image(base, prompt_schedule):
    
    image = get_image_scope(
        pipe=base,
        prompt_schedule = prompt_schedule,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

        
    image = image2image(
        pipe=refiner,
        prompt=prompt_schedule[-1][-1],
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
        
    return image

def get_normal_image(base, prompt):
    
    image = get_image(
        pipe=base,
        prompt = prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

        
    image = image2image(
        pipe=refiner,
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
        
    return image



scope_prompts_path = "/projectnb/vkolagrp/ketanss/scope-diffusers/prompt_dump/genai_300_longclip_iccv/ICCV_LongCLIP_SDXL.json"
exp_dir = "/projectnb/vkolagrp/ketanss/scope-diffusers/exp_dump/iccv_longclip"
import json
with open(scope_prompts_path, 'r') as f:
    prompts = json.load(f)
    prompts = [{k: v} for d in prompts for k, v in d.items()]

import ast 
import os
import matplotlib.pyplot as plt

for idx, prompt in enumerate(prompts):
    for step_size in range(1,9):
        # if config.PROVIDE_PROMPTS:
        prompt_id = list(prompt.keys())[0]
        initial_prompt = list(prompt.values())[0]['initial_prompt']
        progressive_prompts = list(prompt.values())[0]['progressive_prompts']
        prompt_schedule_list = ast.literal_eval(progressive_prompts)
        prompt_schedule = []
                    
        for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
            prompt_schedule.append((stage_id*step_size,p))

        final_prompt = prompt_schedule_list[-1]  # Get the final prompt

        # running scope
        scope_image = get_scope_image(base, prompt_schedule)

        # running normal sd
        normal_image = get_normal_image(base, final_prompt)

        # save image inot each index folder for both scope and normal images
        
        # save image inot each index folder for both scope and normal images
        index_dir = os.path.join(exp_dir, f"{prompt_id}", "seed_0", f"step_size_{step_size}")
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