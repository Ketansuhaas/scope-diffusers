import torch
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
from ezcolorlog import root_logger as logger
import os
import argparse
import json
from get_preprocessed_prompts import get_preprocessed_prompt_lists


from experiments_pipeline import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run SCoPE Diffusion experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["seed", "model", "temperature","overall"],
        default="overall",
        help="Type of experiment to run: 'seed' or 'model'",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default=1,
        help="ID of the experiment",
    )
    parser.add_argument(
        "--exp_desc",
        type=str,
        help="Description of the experiment",
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Run the selected experiment
    if args.experiment == "overall":
        # Custom config for seed experiment
        config_overall = {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1",
            "DEVICE": "cuda",  # or "cpu"
            "seed": 42,
            "num_inference_steps": 50,
            "step_sizes": [1,2,3,5,7], #trying to remove step size
            "temperatures": [1]   # doesn't matter for cslerp, and it is tau for emslerp
        }

        prompts = get_preprocessed_prompt_lists(param='num_nouns', count=100, ascending=True)

        for exp_id in range(len(prompts)):
            # config_overall["prompt_schedule"] = prompts[exp_id]

            config_overall["prompt_schedule"] = [
                
'A magnificent dragon sits on a mountain, under a dramatic sky, gazing fiercely.', 
'A dragon with iridescent scales sits on a smoky mountain under a dramatic sky, gazing fiercely into the distance.', 
"A magnificent dragon, adorned with iridescent scales, perches majestically on a craggy, smoke-wreathed mountain, under a dramatic chiaroscuro sky. The scene is imbued with a misty ambiance, highlighting intricate details in the dragon's wings, as it gazes fiercely into the distance.",

# 'A baker pulls a golden loaf from an oven, surrounded by breads on shelves, in a cozy bakery with the smell of fresh baked goods.', 
# 'A skilled baker pulls a golden loaf of bread from a brick oven, surrounded by artisan breads on wooden shelves, in a cozy bakery filled with the aroma of fresh baked goods.', 
# 'A skilled baker joyfully pulling a golden, steaming loaf of crusty bread from a vintage brick oven, surrounded by rustic wooden shelves filled with assorted artisan breads, bathed in soft warm light, evoking a cozy, inviting bakery atmosphere, rich with the aroma of freshly baked goods.'
]
            exp_seed = SCoPE_Exp_overall(config_overall, args.exp_name, str(exp_id))
            exp_seed.run()
            exit()
        
    else:
        logger.error("Invalid experiment type selected.")



