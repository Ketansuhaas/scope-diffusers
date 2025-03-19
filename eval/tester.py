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
# from genai_prompts.scope_prompt_gen import get_progressive_prompts, get_progressive_prompts_from_scratch
import json
import torch
import re
from PIL import Image
import random
from prompts.system_prompts.sys_prompts import system_prompts
import matplotlib.pyplot as plt
from scope_diffuser_xl import SCoPEDiffusionXLPipeline
from diffusers import DiffusionPipeline
# from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as SCoPEDiffusionXLPipeline
import torch

config = Config()


# set pipes
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
scope_sd_pipe = sdp_scope.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)
normal_sd_pipe = sd_pipe.from_pretrained(config.MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True)

def get_scope_image(prompt_schedule, config):
    torch.manual_seed(config.SEED)
    image = scope_sd_pipe(
            interpolation_technique=config.INTERPOLATION_TECHNIQUE,
            prompt_schedule=prompt_schedule,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            callback=None,
            callback_steps=1,
            temperature=config.TEMPERATURE,
            stdev = config.STDEV
    ) #.images[0]

    return image


def get_normal_sd_image(prompt, config):
    torch.manual_seed(config.SEED)
    image = sd_pipe(
            prompt,
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            callback=None,
            callback_steps=1,
    ).images[0]

    return image


device = "cuda" if torch.cuda.is_available() else "cpu"
scope_sd_pipe.to(device)
normal_sd_pipe.to(device)


import ast 
import pandas as pd
df = pd.read_csv('/projectnb/vkolagrp/ketanss/scope-diffusers-ketan/exp_dump/iccv_cvpr/genai_dataset_preprocessed.csv')
# prompts = prompts[:300]

distances = {
    1: [] , 3:[] , 5: [], 7: []
}

for idx, prompt in enumerate(df['Schedule']):
    if idx == 100:
        break

    for seed in range(1):
        for step_size in [1,3,5,7]:#[1,2,3,4,5,6,7,8]:
            for stdev in [3]:
                config.SEED = 42+seed
                config.STEP_SIZE = step_size
                config.STDEV = stdev
                if config.PROVIDE_PROMPTS:
                    prompt_id = idx #list(prompt.keys())[0]
                    # initial_prompt = list(prompt.values())[0]['initial_prompt']
                    progressive_prompts = prompt #list(prompt.values())[0]['progressive_prompts']
                    # print('progressive')
                    # prompt_schedule_list = ast.literal_eval(progressive_prompts)
                    prompt_schedule_list = ast.literal_eval(prompt)
                    initial_prompt = prompt_schedule_list[0]
                    # match = re.search(r'\[([\s\S]*?)\]', progressive_prompts)
                    # prompt_schedule_match = match.group(1)
                    # prompt_schedule_list = [prompt.strip().strip('"') for prompt in prompt_schedule_match.split('",')]
                    # prompt_schedule_list = [prompt for prompt in prompt_schedule_list if prompt]
                    print(prompt_schedule_list)
                    # exit()
                else:
                    print(prompt)
                    prompt_id = idx
                    # initial_prompt = prompt['initial_prompt']
                    progressive_prompts = prompt #prompt['progressive_prompts']
                    prompt_schedule_list = ast.literal_eval(prompt) #progressive_prompts

                prompt_schedule = []
                step_size = config.STEP_SIZE
                            
                for stage_id, p in enumerate(prompt_schedule_list):       # change step size in the prompt schedule
                    prompt_schedule.append((stage_id*step_size,p))

                final_prompt = prompt_schedule_list[-1]  # Get the final prompt
                initial_prompt = prompt_schedule_list[0]
                logger.info(f"\nPrompt ID: {prompt_id}")
                logger.info(f"Initial Prompt: {initial_prompt}")
                logger.info(f"Progressive Prompts: {progressive_prompts}")
                logger.info(f"Final Prompt: {final_prompt}")


                # running scope
                scope_image = get_scope_image(prompt_schedule, config)
                distances[step_size].append(scope_image)

print(distances)
import pickle

# Save the dictionary to a file
with open('distances.pkl', 'wb') as f:
    pickle.dump(distances, f)



