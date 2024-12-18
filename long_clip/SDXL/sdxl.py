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

use_interpolation = False
if use_interpolation:
    from SDXL_pipeline_prompt_interpolation import get_image
else:
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
step_size = 20

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

if use_interpolation:
    image = get_image(
        pipe=base,
        prompt_schedule = prompt_schedule,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
else:
    image = get_image(
        pipe=base,
        prompt=prompt_schedule[-1][-1],
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
    
image_name = "sdxl_org.png"
image.save(image_name)
