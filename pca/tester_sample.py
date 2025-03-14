import torch
from scope_diffuser_svd import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import pandas as pd
import random

examples = 10

random.seed(42)

# Load the SCoPE Diffusion model
pipe = sdp_scope.from_pretrained(
    'stabilityai/stable-diffusion-2-1', 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
)
pipe = pipe.to("cuda:0")

# Read the JSON file
i = "0"
prompt1 = "A skilled magician in a tailored black tuxedo dramatically pulls a fluffy white rabbit from a classic top hat, illuminated by soft spotlighting. The scene captures the enchanting moment with rich, vibrant colors, intricate details in the magician's expressions, and a whimsical atmosphere filled with sparkling magic dust."
prompt2 = "A magician pulls a rabbit from a top hat."
start_alpha = 0.3

torch.manual_seed(42)
image = pipe(
    prompt1 = prompt1,
    prompt2 = prompt2,
    start_alpha = start_alpha,
    dynamic = False,
    num_inference_steps=50,
    callback=None,
    callback_steps=1,
).images[0]
# Define the path
base_path = "generated_images_sample"
image_path = f"{base_path}/{i}"
os.makedirs(image_path, exist_ok=True)
image.save(f"{image_path}/{start_alpha}.png")


