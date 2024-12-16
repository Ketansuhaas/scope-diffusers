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

# prompt_schedule = [
#     (0,'A magnificent dragon sits on a mountain, under a dramatic sky, gazing fiercely.'), 
#     (step_size,'A dragon with iridescent scales sits on a smoky mountain under a dramatic sky, gazing fiercely into the distance.'), 
#     (step_size*2,"A magnificent dragon, adorned with iridescent scales, perches majestically on a craggy, smoke-wreathed mountain, under a dramatic chiaroscuro sky. The scene is imbued with a misty ambiance, highlighting intricate details in the dragon's wings, as it gazes fiercely into the distance.")
# ]

# prompt_schedule = [

#     (0, "A forest clearing with a small waterfall and a pool of clear water. Sunlight filters through the trees, and rocks surround the pool."),

#     (step_size, "A tranquil forest clearing with a small waterfall cascading into a clear, still pool. Rays of sunlight filter through tall trees, illuminating moss-covered rocks scattered around the water. A few wildflowers bloom along the edges of the pool."),

#     (step_size*2, "A serene forest clearing, where a gentle waterfall flows into a crystal-clear pool, sending ripples across its surface. Sunlight streams through a canopy of tall trees, creating golden beams that dance on moss-covered rocks. Around the pool’s edge, delicate wildflowers in shades of white and purple bloom, while ferns and shrubs frame the scene. The sound of the waterfall blends with the soft rustling of leaves, adding to the tranquil atmosphere.")
# ]
# prompt_schedule = [

#     (0, "A mountain lake surrounded by pine trees and rocky cliffs. A wooden cabin sits by the water’s edge, with smoke rising from its chimney. Snow-capped peaks reflect in the calm water under a pale blue sky."),

#     (step_size, "A serene mountain lake surrounded by dense pine forests and rugged cliffs. A cozy wooden cabin with smoke curling from its chimney sits at the water’s edge. Snow-capped peaks loom in the background, their reflections shimmering on the lake’s glassy surface. A small wooden dock juts into the lake, and a canoe is tied to it. The sky is pale blue, dotted with fluffy white clouds."),

#     (step_size*2, "A tranquil mountain lake framed by towering pine forests and craggy cliffs, with a rustic wooden cabin nestled at the water’s edge. Smoke rises lazily from the chimney as sunlight reflects off the calm, glassy surface of the lake. Snow-capped peaks rise majestically in the distance, their mirrored images rippling softly in the water. A small wooden dock stretches into the lake, where a red canoe rocks gently. Wildflowers bloom along the shore, and an eagle soars high above under a pale blue sky streaked with wisps of cloud. The air feels crisp and carries the scent of pine and fresh water.")
# ]

# prompt_schedule = [

#     (0, "A desert oasis with a small pool of water surrounded by palm trees. Sand dunes stretch into the distance under a blazing sun, and a lone camel drinks from the pool’s edge while a hawk circles overhead."),

#     (step_size, "A tranquil desert oasis with a crystal-clear pool reflecting the tall, swaying palm trees that surround it. Sand dunes rise in golden waves beyond the oasis, their ridges catching the harsh sunlight. The air shimmers with heat, and a lone camel drinks from the pool’s edge while a hawk circles overhead."),

#     (step_size*2, "A vibrant desert oasis where a shimmering pool of water mirrors the lush palm trees arching overhead. Surrounding the oasis are undulating sand dunes, their golden crests glowing under the intense midday sun. The air ripples with heat, and the distant horizon blurs into a hazy mirage. A lone camel kneels to drink from the pool, while small desert flowers bloom in the sparse shade nearby. Above, a hawk soars lazily in the deep blue sky, its cry echoing faintly. A few nomadic tents stand at the edge of the oasis, adding a touch of life to the barren landscape.")
# ]

prompt_schedule = [

    (0, "A mountain lake, a boat, pine trees, rocky cliffs, a wooden cabin with smoke from chimney, snow-capped peaks"),

    # (step_size, "A serene mountain lake surrounded by dense pine forests and rugged cliffs. A cozy wooden cabin with smoke curling from its chimney sits at the water's edge. Snow-capped peaks loom in the background, their reflections shimmering on the lake's glassy surface. A small wooden dock juts into the lake, and a canoe is tied to it. The sky is pale blue, dotted with fluffy white clouds."),

    (step_size, "A tranquil mountain lake with a boat, framed by towering pine forests and craggy cliffs, with a rustic wooden cabin nestled at the water's edge. Smoke rises lazily from the chimney as sunlight reflects off the calm, glassy surface of the lake. Snow-capped peaks rise majestically in the distance, their mirrored images rippling softly in the water. A small wooden dock stretches into the lake, where a red canoe rocks gently. Wildflowers bloom along the shore, and an eagle soars high above under a pale blue sky streaked with wisps of cloud. The air feels crisp and carries the scent of pine and fresh water.")
]
# prompt_schedule = [
#     (0,"A little girl on the street holding pocket watches for sale"),
#     (step_size,"A little girl on a quiet street holding worn pocket watches for sale, old buildings with faded signs in the background, cobblestone path"),
#     (step_size*2,"A little girl on a quiet street holding worn pocket watches for sale, old buildings with faded signs in the background, cobblestone path, scattered leaves on the ground, dim evening light")
# ]
# prompt_schedule = [
#     (0, 'A vintage teapot with steam sits on a wooden table with lace. Soft light highlights its floral patterns.'), 
#     (step_size, 'A vintage ceramic teapot with steam from its spout sits on a wooden kitchen table with lace. Soft light illuminates the floral patterns of the teapot.'), 
#     (step_size*2, 'A vintage ceramic teapot, steam curling gracefully from its spout, rests on a weathered wood kitchen table adorned with delicate lace. Soft, warm light filters through a nearby window, illuminating the intricate floral patterns of the teapot, creating a cozy and inviting atmosphere.')
# ]

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
