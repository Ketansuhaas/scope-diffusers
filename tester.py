import torch
from scope_diffuser_svd import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the SCoPE Diffusion model
pipe = sdp_scope.from_pretrained(
    'stabilityai/stable-diffusion-2-1', 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    cache_dir = '/projectnb/vkolagrp/ketanss/scope-diffusers/sdpcache'
)
pipe = pipe.to("cuda:0")
image_scope_list = []

prompt_schedule = [(0,"A photo of dog")]

torch.manual_seed(42)

image = pipe(
    prompt = "A photo of dog",
    num_inference_steps=50,
    callback=None,
    callback_steps=1,
    # temperature=temp,
).images[0]

image.save("/projectnb/vkolagrp/ketanss/scope-diffusers/experiments/svd.png")