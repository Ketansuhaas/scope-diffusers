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

from SDXL_pipeline import get_image
# from SDXL_pipeline_backup import get_image
from SDXL_img2img import image2image

base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt_list =  [
                "This is a delightful image of a small, white, curly-furred dog standing on a lush, green lawn.",
                "This is a delightful image of a small, white, curly-furred dog standing on a lush, green lawn. The dog is actively engaged in play, holding a purple frisbee in its mouth. The frisbee, which bears the black and white logo \"KONG FLYER\", is held firmly by the dog who is looking directly at the camera, possibly awaiting the next throw. The contrast of the purple frisbee against the white fur of the dog and the green grass adds a vibrant pop of color to the scene. The dog's eager eyes and playful stance suggest it's ready for more fun and games. This image captures a beautiful moment of joy and playfulness between a pet and its owner."
                # "A waterfall made of glowing stars, pouring from the sky",

                # "A waterfall of stars pouring from the sky, lighting the landscape, with mist filled with sparkling particles, and constellations forming in the mist",
            ]

image = get_image(
    pipe=base,
    prompt_list=prompt_list,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

# image.save("sdxl_latent.png")
    
    
image = image2image(
    pipe=refiner,
    prompt=prompt_list[-1],
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
    
image_name = "sdxl_int.png"
image.save(image_name)
