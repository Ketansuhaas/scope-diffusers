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
        choices=["seed", "model", "temperature"],
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
        config_seed["prompt_schedule"] = prompt_schedule = [
            (0, "An astronaut riding a horse on a barren landscape"),  # Basic layout
            (config_seed["step_sizes"][0], "An astronaut riding a horse on a barren, dusty landscape with stars visible in the distance"),  # Add stars in the distance
            (config_seed["step_sizes"][1], "An astronaut riding a horse on a barren, dusty landscape, with stars and a faint view of a distant planet in the background"),  # Add distant planet
            (config_seed["step_sizes"][2], "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a faint view of a distant planet, the astronaut's visor reflecting starlight"),  # Add starlight reflection
            (config_seed["step_sizes"][3], "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a distant planet in the background, the astronaut's visor reflecting starlight, as comet trails streak across the sky")  # Add comet trails
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


        config_model["prompt_schedule"] = prompt_schedule = [
            (0, "A cityscape at night."),  # Basic layout
            (config_model["step_sizes"][0], "A cityscape at night, illuminated by neon lights."),  # Add neon lights
            (config_model["step_sizes"][1], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers and flying cars zooming past."),  # Middle prompt
            (config_model["step_sizes"][2], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers, flying cars zooming past, and pedestrians below."),  # Add pedestrians
            (config_model["step_sizes"][3], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers, flying cars zooming past, pedestrians below, and holographic advertisements flickering."),  # Add holographic ads
        ]

        exp_model = SCoPE_Exp_Model(config_model, args.exp_name, args.exp_id)
        exp_model.run()

    elif args.experiment == "num_inference_steps":
        # Custom config for num_inference_steps experiment
        config_num_inference_steps = {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "DEVICE": "cuda",  # or "cpu"
            "seed": 42,
            "num_inference_steps_list": [25, 50, 75, 100, 300, 500, 700, 900],
            "step_sizes": [5, 10, 15, 20],
        }

        for num_inference_steps in config_num_inference_steps["num_inference_steps_list"]:
            config_num_inference_steps["num_inference_steps"] = num_inference_steps
            config_num_inference_steps["prompt_schedule"] = prompt_schedule = [
                (0, "A cityscape at night."),  # Basic layout
                (config_num_inference_steps["step_sizes"][0], "A cityscape at night, illuminated by neon lights."),  # Add neon lights
                (config_num_inference_steps["step_sizes"][1], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers and flying cars zooming past."),  # Middle prompt
                (config_num_inference_steps["step_sizes"][2], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers, flying cars zooming past, and pedestrians below."),  # Add pedestrians
                (config_num_inference_steps["step_sizes"][3], "A cyberpunk cityscape at night, neon lights illuminating towering skyscrapers, flying cars zooming past, pedestrians below, and holographic advertisements flickering."),  # Add holographic ads
            ]
            exp_num_inference_steps = SCoPE_Exp_Num_Inference_Steps(config_num_inference_steps, args.exp_name, args.exp_id)
            exp_num_inference_steps.run()

    # Run the selected experiment
    if args.experiment == "temperature":
        # Custom config for temperature experiment
        config_temperature = {
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "DEVICE": "cuda",  # or "cpu"
            "temperature_list": [0, 0.1, 0.5, 1.0, 1.5, 2.0],  # List of temperatures to try
            "num_inference_steps": 50,
            "step_sizes": [5, 10, 15, 20],  # Example step sizes
        }

        # Define prompt schedule based on step sizes
        config_temperature["prompt_schedule"] = prompt_schedule = [
            (0, "An astronaut riding a horse on a barren landscape"),  # Basic layout
            (config_temperature["step_sizes"][0], "An astronaut riding a horse on a barren, dusty landscape with stars visible in the distance"),  # Add stars in the distance
            (config_temperature["step_sizes"][1], "An astronaut riding a horse on a barren, dusty landscape, with stars and a faint view of a distant planet in the background"),  # Add distant planet
            (config_temperature["step_sizes"][2], "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a faint view of a distant planet, the astronaut's visor reflecting starlight"),  # Add starlight reflection
            (config_temperature["step_sizes"][3], "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a distant planet in the background, the astronaut's visor reflecting starlight, as comet trails streak across the sky")  # Add comet trails
        ]

        exp_temperature = SCoPE_Exp_Temperature(config_temperature, args.exp_name, args.exp_id)
        exp_temperature.run()


    else:
        logger.error("Invalid experiment type selected.")