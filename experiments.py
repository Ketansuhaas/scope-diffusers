import torch
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
from ezcolorlog import root_logger as logger
import os
import argparse

from experiments_pipeline import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run SCoPE Diffusion experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["seed", "model"],
        required=True,
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
        required=True,
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
    if args.experiment == "seed":
        # Custom config for seed experiment
        config_seed = {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "DEVICE": "cuda",  # or "cpu"
            "seed_list": [42, 123, 999],
            "num_inference_steps": 50,
            "step_sizes": [5, 10, 15, 20],
        }

        # Define prompt schedule based on step sizes
        config_seed["prompt_schedule"] = [
            (0, "A man in a workshop."),  
            (config_seed["step_sizes"][0], "A man shaping clay on a wheel in a workshop."),  
            (config_seed["step_sizes"][1], "A man shaping clay on a wheel in a cluttered workshop."),  
            (config_seed["step_sizes"][2], "A man shaping clay on a wheel in a cluttered workshop, with tools scattered around."),  
            (config_seed["step_sizes"][3], "A man shaping clay on a wheel in a cluttered workshop, tools scattered around, and sunlight streaming through a window."),  
        ]

        exp_seed = SCoPE_Exp_Seed(config_seed, args.exp_name, args.exp_id)
        exp_seed.run()

    elif args.experiment == "model":
        # Custom config for model experiment
        config_model = {
            "MODEL_ID": "",  # Placeholder, will be set in the loop
            "DEVICE": "cuda",  # or "cpu"
            "seed": 42,
            "model_ids": ["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
            "num_inference_steps": 50,
            "step_sizes": [5, 10, 15, 20],
        }

        # Define prompt schedule based on step sizes
        config_model["prompt_schedule"] = [
            (0, "A man in a workshop."),  
            (config_model["step_sizes"][0], "A man shaping clay on a wheel in a workshop."),  
            (config_model["step_sizes"][1], "A man shaping clay on a wheel in a cluttered workshop."),  
            (config_model["step_sizes"][2], "A man shaping clay on a wheel in a cluttered workshop, with tools scattered around."),  
            (config_model["step_sizes"][3], "A man shaping clay on a wheel in a cluttered workshop, tools scattered around, and sunlight streaming through a window."),  
        ]

        exp_model = SCoPE_Exp_Model(config_model, args.exp_name, args.exp_id)
        exp_model.run()

    else:
        logger.error("Invalid experiment type selected.")