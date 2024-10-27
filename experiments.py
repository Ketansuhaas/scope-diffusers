import torch
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
from ezcolorlog import root_logger as logger
import os
import argparse
import json

from experiments_pipeline import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run SCoPE Diffusion experiments.")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["seed", "model", "temperature","overall"],
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
            "step_sizes": [1,2,3], 
            "temperatures": [1]   # doesn't matter for cslerp, and it is tau for emslerp
        }

        with open('/projectnb/vkolagrp/ketanss/scope-diffusers/genai_prompts/scope_prompts_responses_universal_new.json', 'r') as file:
            data = json.load(file)

        for exp_id in range(len(data)):
            text = data[exp_id]['progressive_prompts']
            import re
            # Use regex to find the content inside the square brackets
            match = re.search(r'\[([\s\S]*?)\]', text)

            if match:
                # Extract the content
                content = match.group(1)
                
                # Split the content into individual prompts
                prompts = [prompt.strip().strip('"') for prompt in content.split('",')]
                
                # Remove any empty strings
                prompts = [prompt for prompt in prompts if prompt]
                
                # Print the result
                print(len(prompts))
            else:
                print("No matching content found.")
                
            config_overall["prompt_schedule"] = prompts
            exp_seed = SCoPE_Exp_overall(config_overall, args.exp_name, str(exp_id))
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
    elif args.experiment == "temperature":
        # Custom config for temperature experiment
        config_temperature = {
            "MODEL_ID": "stabilityai/stable-diffusion-2-1",
            "DEVICE": "cuda",  # or "cpu"
            "temperature_list": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],  # List of temperatures to try
            "num_inference_steps": 200,
            "step_sizes": [10, 20, 28],  # Example step sizes
        }

        prompt_schedules = [
            [
                "A surreal landscape with giant mushrooms scattered across a rolling hill, a winding path leading through, and a distant castle on a hilltop.",  # Basic layout
                "A surreal landscape with giant purple mushrooms scattered across a rolling hill, a winding cobblestone path leading through, and a distant fairy-tale castle on a hilltop.",  # Add mushroom color
                "A surreal landscape with giant purple mushrooms scattered across a rolling hill, a winding cobblestone path leading through, a distant fairy-tale castle on a hilltop, and butterflies fluttering around.",  # Add butterflies
                "A surreal landscape with giant purple mushrooms scattered across a rolling hill, a winding cobblestone path leading through, a distant fairy-tale castle on a hilltop, butterflies fluttering, and a rainbow arching across the sky.",  # Add rainbow
                "A surreal landscape with giant purple mushrooms scattered across a rolling hill, a winding cobblestone path leading through, a distant fairy-tale castle on a hilltop, butterflies fluttering, a rainbow arching across the sky, and whimsical creatures peeking out from behind the mushrooms."  # Add creatures
            ],
            [
                "A cosmic café floating in space, with planets visible through the large windows, and colorful chairs arranged around tables.",  # Basic layout
                "A cosmic café floating in space, with planets visible through the large windows, colorful chairs arranged around tables, and a barista serving drinks behind the counter.",  # Add barista
                "A cosmic café floating in space, with planets visible through the large windows, colorful chairs arranged around tables, a barista serving drinks behind the counter, and patrons enjoying their drinks while gazing at the stars.",  # Add patrons
                "A cosmic café floating in space, with planets visible through the large windows, colorful chairs arranged around tables, a barista serving drinks behind the counter, patrons enjoying their drinks, and neon lights illuminating the café.",  # Add lights
                "A cosmic café floating in space, with planets visible through the large windows, colorful chairs arranged around tables, a barista serving drinks behind the counter, patrons enjoying their drinks, neon lights illuminating the café, and spaceships flying by outside."  # Add spaceships
            ],
            [
                "An enchanted library with towering shelves filled with books, a grand staircase, and a large stained-glass window.",  # Basic layout
                "An enchanted library with towering shelves filled with colorful books, a grand wooden staircase, and a large stained-glass window casting vibrant colors.",  # Add book details
                "An enchanted library with towering shelves filled with colorful books, a grand wooden staircase, a large stained-glass window casting vibrant colors, and a cozy reading nook with plush chairs.",  # Add reading nook
                "An enchanted library with towering shelves filled with colorful books, a grand wooden staircase, a large stained-glass window casting vibrant colors, a cozy reading nook with plush chairs, and a cat lounging on the windowsill.",  # Add cat
                "An enchanted library with towering shelves filled with colorful books, a grand wooden staircase, a large stained-glass window casting vibrant colors, a cozy reading nook with plush chairs, a cat lounging on the windowsill, and soft candlelight illuminating the space."  # Add candlelight
            ],
            [
                "A surreal dreamscape with floating islands, colorful clouds, and a giant moon hanging low in the sky.",  # Basic layout
                "A surreal dreamscape with floating islands covered in lush greenery, colorful clouds swirling around, and a giant glowing moon hanging low in the sky.",  # Add island details
                "A surreal dreamscape with floating islands covered in lush greenery, colorful clouds swirling around, a giant glowing moon hanging low in the sky, and fantastical creatures flying between the islands.",  # Add creatures
                "A surreal dreamscape with floating islands covered in lush greenery, colorful clouds swirling around, a giant glowing moon hanging low in the sky, fantastical creatures flying between the islands, and shimmering stars twinkling in the background.",  # Add stars
                "A surreal dreamscape with floating islands covered in lush greenery, colorful clouds swirling around, a giant glowing moon hanging low in the sky, fantastical creatures flying between the islands, shimmering stars twinkling in the background, and soft music echoing through the air."  # Add music
            ],
            [
                "A view of a large city square with a tall monument in the center and a road circling it.",  # Basic layout
                "A view of a large city square with a tall stone monument in the center, a road circling it, and trees lining the perimeter.",  # Add trees
                "A view of a large city square with a tall stone monument, a road circling it, trees lining the perimeter, and benches scattered around.",  # Add benches
                "A view of a large city square with a tall stone monument, a road circling it, trees lining the perimeter, benches scattered around, and people walking by.",  # Add people
                "A view of a large city square with a tall stone monument, a road circling it, trees, benches, people walking, and cars driving around the square."  # Add cars
            ],
            [
                "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees and distant rocky cliffs.",  # Basic layout
                "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees and distant rocky cliffs, and patches of golden sand.",  # Add sand
                "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees reflecting in the water, distant rocky cliffs, and patches of golden sand.",  # Add reflections
                "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees reflecting in the water, distant rocky cliffs, patches of golden sand, and vibrant green vegetation nearby.",  # Add vegetation
                "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees reflecting in the water and gently swaying, distant rocky cliffs, patches of golden sand, vibrant green vegetation nearby, and soft clouds drifting across the clear blue sky."  # Add clouds
            ],
            [
                "A view of Venice from a boat on the river, with tall buildings on both sides and a bridge ahead.",  # Basic layout
                "A view of Venice from a boat on the river, with tall red buildings on both sides, and a stone bridge ahead with people walking.",  # Add color and people
                "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, and soft lanterns lining the riverbank.",  # Add river color and lanterns
                "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, soft lanterns lining the riverbank, and colorful carnival decorations along the buildings.",  # Add carnival decorations
                "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, soft lanterns lining the riverbank, colorful carnival decorations along the buildings, and several boats drifting down the river toward the horizon."  # Add boats
            ],
            [
                "A bustling city square with tall modern buildings and a central fountain.",  # Basic layout
                "A bustling city square with tall modern glass buildings, a central fountain, and trees lining the streets.",  # Add trees
                "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, and people sitting on benches.",  # Add benches and people
                "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, people sitting on benches, and shopfronts in the background.",  # Add shopfronts
                "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, people sitting on benches, shopfronts in the background, and sunlight reflecting off the windows."  # Add sunlight
            ],
        ]


        for exp_id, prompt_schedule_list in enumerate(prompt_schedules):
            config_temperature["prompt_schedule"] = prompt_schedule_list
            exp_seed = SCoPE_Exp_Temperature(config_temperature, args.exp_name, str(exp_id))
            exp_seed.run()


    else:
        logger.error("Invalid experiment type selected.")



