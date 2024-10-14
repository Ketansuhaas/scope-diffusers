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
            "MODEL_ID": "stabilityai/stable-diffusion-2-1",
            "DEVICE": "cuda",  # or "cpu"
            "seed_list": [42, 123, 999],
            "num_inference_steps": 200,
            "step_sizes": [10, 20, 28],  # Example step sizes
        }

        # # Define prompt schedule based on step sizes
        # config_seed["prompt_schedule"] = prompt_schedule = [
        #     "An astronaut riding a horse on a barren landscape",  # Basic layout
        #     # "An astronaut riding a horse on a barren, dusty landscape with stars visible in the distance",  # Add stars in the distance
        #     # "An astronaut riding a horse on a barren, dusty landscape, with stars and a faint view of a distant planet in the background",  # Add distant planet
        #     # "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a faint view of a distant planet, the astronaut's visor reflecting starlight",  # Add starlight reflection
        #     "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a distant planet in the background, the astronaut's visor reflecting starlight, as comet trails streak across the sky"  # Add comet trails
        # ]
        prompt_schedules = [
            [
                "A serene lakeside at sunset, the sky's vibrant colors reflected on the water",  # Basic layout
                "A serene lakeside at sunset, the sky's vibrant colors reflected in the water, with surrounding trees casting long shadows",  # Add shadows
                "A serene lakeside at sunset, vibrant sky colors reflected on the still water, with surrounding trees and distant mountains casting long shadows",  # Add distant mountains
                "A serene lakeside at sunset, vibrant sky colors reflected in still waters, surrounded by trees and distant mountains with soft golden light",  # Full details
                "A serene lakeside at sunset, vibrant sky colors reflected in still waters, with surrounding trees, distant mountains, and small birds flying over the lake as the golden light fades"  # Extra detail: birds and fading light
            ],
            [
                "An ancient, overgrown temple in a dense jungle, with soft light breaking through the canopy",  # Basic layout
                "An ancient, overgrown temple in a dense jungle, the soft morning light illuminating vines creeping over stone ruins",  # Add vines
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, vines and moss covering the crumbling stone structure",  # Add moss
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines and moss overtaking the ancient stone ruins as the jungle thrives",  # Full details
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines, moss, and exotic flowers overtaking the ancient stone ruins as the jungle teems with life"  # Extra detail: exotic flowers and life
            ],
            [
                "A bustling cyberpunk cityscape at night, bathed in neon lights",  # Basic layout
                "A bustling cyberpunk cityscape at night, bathed in neon lights, with flying cars zooming past skyscrapers",  # Add flying cars
                "A bustling cyberpunk cityscape at night, glowing with neon signs, flying cars zooming past towering skyscrapers, streets crowded with people and robots",  # Add streets and people
                "A bustling cyberpunk cityscape at night, bathed in neon lights, flying cars soaring past skyscrapers, crowds of people and robots filling the neon-lit streets below",  # Full details
                "A bustling cyberpunk cityscape at night, neon lights reflected in rain-soaked streets, flying cars zooming past skyscrapers as crowds of people and robots navigate the wet city"  # Extra detail: rain-soaked streets
            ],
            [
                "A majestic mountain range under the cloak of night, lit only by a full moon",  # Basic layout
                "A majestic mountain range under the night sky, lit by a full moon with stars visible above",  # Add stars
                "A majestic mountain range under the night sky, illuminated by a full moon, stars twinkling above and reflecting on a calm lake below",  # Add lake reflection
                "A majestic mountain range under a starlit night sky, illuminated by a full moon, stars reflected on a calm lake, with shadows cast by the peaks",  # Full details
                "A majestic mountain range under a starlit sky, illuminated by a full moon, stars reflected on a calm lake, with mist rising from the water and shadows dancing on the peaks"  # Extra detail: mist and shadows
            ],
            [
                "A floral arrangement of soft pink roses and white Chinese peony in a pink nickel mug",  # Basic layout
                "A floral arrangement of soft pink roses, white Chinese peony, and eucalyptus leaves in a pink nickel mug",  # Add eucalyptus
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink mug, sitting on a thick white book",  # Add apple blossoms
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, in bright sunlight",  # Full details
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, bathed in bright sunlight, with a soft shadow cast on a nearby table"  # Extra detail: shadow
            ],
            [
                "An astronaut standing on Mars during sunset",  # Basic layout
                "An astronaut on Mars at sunset, with the Martian landscape stretching into the horizon",  # Add landscape
                "An astronaut on Mars at sunset, standing in a rocky terrain, with orange-red hues across the sky",  # Add sky hues
                "An astronaut on Mars during sunset, standing in a rocky landscape under the orange-red sky, distant mountains visible in the background",  # Full details
                "An astronaut on Mars at sunset, standing in a rocky landscape under a vivid orange-red sky, distant mountains in the background, and dust trails swirling in the wind"  # Extra detail: dust trails
            ],
            [
                "A minimalist logo featuring a surreal cityscape during a rainy night",  # Basic layout
                "A minimalist logo showing a surreal cityscape at night, wet streets reflecting neon signs",  # Add neon reflections
                "A minimalist logo of a surreal cityscape at night, with wet pavements, glowing neon signs, and reflections in puddles",  # Add puddles
                "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflecting in puddles, wet streets lined with towering skyscrapers",  # Full details
                "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflected in puddles, towering skyscrapers fading into the mist, with distant lights visible in the fog"  # Extra detail: mist and fog
            ],
            [
                "A small cozy house in the redwoods on a mountain",  # Basic layout
                "A small cozy house in the redwoods, with solar panels and a garage on a mountainside",  # Add solar panels
                "A small cozy modern house in the redwoods, with solar panels, a garage, and a driveway overlooking the mountain",  # Add driveway
                "A small modern house in the redwoods, with solar panels, a garage, driveway, and a great view of the mountains in the sunshine",  # Full details
                "A small modern house in the redwoods, with solar panels, a garage, driveway, and a stunning view of the mountains, bathed in sunshine, with a small garden blooming with flowers"  # Extra detail: blooming garden
            ],
            [
                "A new town square in Cambridge, with a big traditional museum",  # Basic layout
                "A new town square in Cambridge, with a big traditional museum, a fountain in the center",  # Add fountain
                "A new town square in Cambridge, featuring a traditional museum, a large fountain, and surrounding trees",  # Add trees
                "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square",  # Full details
                "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square, as people stroll along the pathways under the dappled sunlight"  # Extra detail: people and sunlight
            ],
            [
                "A breathtaking cityscape at dusk, illuminated by a warm glow",  # Basic layout
                "A breathtaking cityscape at dusk, with a skyline glowing in the golden light of the sunset",  # Add sunset light
                "A cityscape at dusk, the skyline bathed in golden light with silhouettes of skyscrapers and distant clouds",  # Add silhouettes
                "A breathtaking cityscape at dusk, with the skyline illuminated by a warm, golden glow, silhouetted skyscrapers rising against the glowing sky",  # Full details
                "A breathtaking cityscape at dusk, illuminated by a warm, golden glow, with silhouetted skyscrapers rising against the glowing sky, while soft clouds roll in the background"  # Extra detail: clouds rolling
            ]
        ]

        for exp_id, prompt_schedule_list in enumerate(prompt_schedules):
            config_seed["prompt_schedule"] = prompt_schedule_list
            exp_seed = SCoPE_Exp_Seed(config_seed, args.exp_name, str(exp_id))
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
            "MODEL_ID": "CompVis/stable-diffusion-v1-4",
            "DEVICE": "cuda",  # or "cpu"
            "temperature_list": [0, 0.1, 0.5, 1.0, 1.5, 2.0],  # List of temperatures to try
            "num_inference_steps": 200,
            "step_sizes": [10, 20, 28],  # Example step sizes
        }

        # # Define prompt schedule based on step sizes
        # config_temperature["prompt_schedule"] = prompt_schedule = [
        #     "An astronaut riding a horse on a barren landscape",  # Basic layout
        #     "An astronaut riding a horse on a barren, dusty landscape with stars visible in the distance",  # Add stars in the distance
        #     "An astronaut riding a horse on a barren, dusty landscape, with stars and a faint view of a distant planet in the background",  # Add distant planet
        #     "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a faint view of a distant planet, the astronaut's visor reflecting starlight",  # Add starlight reflection
        #     "An astronaut riding a horse on a barren, dusty landscape under a starlit sky, with a distant planet in the background, the astronaut's visor reflecting starlight, as comet trails streak across the sky"  # Add comet trails
        # ]
        prompt_schedules = [
            [
                "A serene lakeside at sunset, the sky's vibrant colors reflected on the water",  # Basic layout
                "A serene lakeside at sunset, the sky's vibrant colors reflected in the water, with surrounding trees casting long shadows",  # Add shadows
                "A serene lakeside at sunset, vibrant sky colors reflected on the still water, with surrounding trees and distant mountains casting long shadows",  # Add distant mountains
                "A serene lakeside at sunset, vibrant sky colors reflected in still waters, surrounded by trees and distant mountains with soft golden light",  # Full details
                "A serene lakeside at sunset, vibrant sky colors reflected in still waters, with surrounding trees, distant mountains, and small birds flying over the lake as the golden light fades"  # Extra detail: birds and fading light
            ],
            [
                "An ancient, overgrown temple in a dense jungle, with soft light breaking through the canopy",  # Basic layout
                "An ancient, overgrown temple in a dense jungle, the soft morning light illuminating vines creeping over stone ruins",  # Add vines
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, vines and moss covering the crumbling stone structure",  # Add moss
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines and moss overtaking the ancient stone ruins as the jungle thrives",  # Full details
                "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines, moss, and exotic flowers overtaking the ancient stone ruins as the jungle teems with life"  # Extra detail: exotic flowers and life
            ],
            [
                "A bustling cyberpunk cityscape at night, bathed in neon lights",  # Basic layout
                "A bustling cyberpunk cityscape at night, bathed in neon lights, with flying cars zooming past skyscrapers",  # Add flying cars
                "A bustling cyberpunk cityscape at night, glowing with neon signs, flying cars zooming past towering skyscrapers, streets crowded with people and robots",  # Add streets and people
                "A bustling cyberpunk cityscape at night, bathed in neon lights, flying cars soaring past skyscrapers, crowds of people and robots filling the neon-lit streets below",  # Full details
                "A bustling cyberpunk cityscape at night, neon lights reflected in rain-soaked streets, flying cars zooming past skyscrapers as crowds of people and robots navigate the wet city"  # Extra detail: rain-soaked streets
            ],
            [
                "A majestic mountain range under the cloak of night, lit only by a full moon",  # Basic layout
                "A majestic mountain range under the night sky, lit by a full moon with stars visible above",  # Add stars
                "A majestic mountain range under the night sky, illuminated by a full moon, stars twinkling above and reflecting on a calm lake below",  # Add lake reflection
                "A majestic mountain range under a starlit night sky, illuminated by a full moon, stars reflected on a calm lake, with shadows cast by the peaks",  # Full details
                "A majestic mountain range under a starlit sky, illuminated by a full moon, stars reflected on a calm lake, with mist rising from the water and shadows dancing on the peaks"  # Extra detail: mist and shadows
            ],
            [
                "A floral arrangement of soft pink roses and white Chinese peony in a pink nickel mug",  # Basic layout
                "A floral arrangement of soft pink roses, white Chinese peony, and eucalyptus leaves in a pink nickel mug",  # Add eucalyptus
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink mug, sitting on a thick white book",  # Add apple blossoms
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, in bright sunlight",  # Full details
                "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, bathed in bright sunlight, with a soft shadow cast on a nearby table"  # Extra detail: shadow
            ],
            [
                "An astronaut standing on Mars during sunset",  # Basic layout
                "An astronaut on Mars at sunset, with the Martian landscape stretching into the horizon",  # Add landscape
                "An astronaut on Mars at sunset, standing in a rocky terrain, with orange-red hues across the sky",  # Add sky hues
                "An astronaut on Mars during sunset, standing in a rocky landscape under the orange-red sky, distant mountains visible in the background",  # Full details
                "An astronaut on Mars at sunset, standing in a rocky landscape under a vivid orange-red sky, distant mountains in the background, and dust trails swirling in the wind"  # Extra detail: dust trails
            ],
            [
                "A minimalist logo featuring a surreal cityscape during a rainy night",  # Basic layout
                "A minimalist logo showing a surreal cityscape at night, wet streets reflecting neon signs",  # Add neon reflections
                "A minimalist logo of a surreal cityscape at night, with wet pavements, glowing neon signs, and reflections in puddles",  # Add puddles
                "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflecting in puddles, wet streets lined with towering skyscrapers",  # Full details
                "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflected in puddles, towering skyscrapers fading into the mist, with distant lights visible in the fog"  # Extra detail: mist and fog
            ],
            [
                "A small cozy house in the redwoods on a mountain",  # Basic layout
                "A small cozy house in the redwoods, with solar panels and a garage on a mountainside",  # Add solar panels
                "A small cozy modern house in the redwoods, with solar panels, a garage, and a driveway overlooking the mountain",  # Add driveway
                "A small modern house in the redwoods, with solar panels, a garage, driveway, and a great view of the mountains in the sunshine",  # Full details
                "A small modern house in the redwoods, with solar panels, a garage, driveway, and a stunning view of the mountains, bathed in sunshine, with a small garden blooming with flowers"  # Extra detail: blooming garden
            ],
            [
                "A new town square in Cambridge, with a big traditional museum",  # Basic layout
                "A new town square in Cambridge, with a big traditional museum, a fountain in the center",  # Add fountain
                "A new town square in Cambridge, featuring a traditional museum, a large fountain, and surrounding trees",  # Add trees
                "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square",  # Full details
                "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square, as people stroll along the pathways under the dappled sunlight"  # Extra detail: people and sunlight
            ],
            [
                "A breathtaking cityscape at dusk, illuminated by a warm glow",  # Basic layout
                "A breathtaking cityscape at dusk, with a skyline glowing in the golden light of the sunset",  # Add sunset light
                "A cityscape at dusk, the skyline bathed in golden light with silhouettes of skyscrapers and distant clouds",  # Add silhouettes
                "A breathtaking cityscape at dusk, with the skyline illuminated by a warm, golden glow, silhouetted skyscrapers rising against the glowing sky",  # Full details
                "A breathtaking cityscape at dusk, illuminated by a warm, golden glow, with silhouetted skyscrapers rising against the glowing sky, while soft clouds roll in the background"  # Extra detail: clouds rolling
            ]
        ]

        for exp_id, prompt_schedule_list in enumerate(prompt_schedules):
            config_seed["prompt_schedule"] = prompt_schedule_list
            exp_seed = SCoPE_Exp_Seed(config_seed, args.exp_name, str(exp_id))
            exp_seed.run()


    else:
        logger.error("Invalid experiment type selected.")