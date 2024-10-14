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
            "seed_list": [42,43,44],#, 123, 999],
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
        # prompt_schedules = [
        #     [
        #         "A serene lakeside at sunset, the sky's vibrant colors reflected on the water",  # Basic layout
        #         "A serene lakeside at sunset, the sky's vibrant colors reflected in the water, with surrounding trees casting long shadows",  # Add shadows
        #         "A serene lakeside at sunset, vibrant sky colors reflected on the still water, with surrounding trees and distant mountains casting long shadows",  # Add distant mountains
        #         "A serene lakeside at sunset, vibrant sky colors reflected in still waters, surrounded by trees and distant mountains with soft golden light",  # Full details
        #         "A serene lakeside at sunset, vibrant sky colors reflected in still waters, with surrounding trees, distant mountains, and small birds flying over the lake as the golden light fades"  # Extra detail: birds and fading light
        #     ],
        #     [
        #         "An ancient, overgrown temple in a dense jungle, with soft light breaking through the canopy",  # Basic layout
        #         "An ancient, overgrown temple in a dense jungle, the soft morning light illuminating vines creeping over stone ruins",  # Add vines
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, vines and moss covering the crumbling stone structure",  # Add moss
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines and moss overtaking the ancient stone ruins as the jungle thrives",  # Full details
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines, moss, and exotic flowers overtaking the ancient stone ruins as the jungle teems with life"  # Extra detail: exotic flowers and life
        #     ],
        #     [
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights",  # Basic layout
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights, with flying cars zooming past skyscrapers",  # Add flying cars
        #         "A bustling cyberpunk cityscape at night, glowing with neon signs, flying cars zooming past towering skyscrapers, streets crowded with people and robots",  # Add streets and people
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights, flying cars soaring past skyscrapers, crowds of people and robots filling the neon-lit streets below",  # Full details
        #         "A bustling cyberpunk cityscape at night, neon lights reflected in rain-soaked streets, flying cars zooming past skyscrapers as crowds of people and robots navigate the wet city"  # Extra detail: rain-soaked streets
        #     ],
        #     [
        #         "A majestic mountain range under the cloak of night, lit only by a full moon",  # Basic layout
        #         "A majestic mountain range under the night sky, lit by a full moon with stars visible above",  # Add stars
        #         "A majestic mountain range under the night sky, illuminated by a full moon, stars twinkling above and reflecting on a calm lake below",  # Add lake reflection
        #         "A majestic mountain range under a starlit night sky, illuminated by a full moon, stars reflected on a calm lake, with shadows cast by the peaks",  # Full details
        #         "A majestic mountain range under a starlit sky, illuminated by a full moon, stars reflected on a calm lake, with mist rising from the water and shadows dancing on the peaks"  # Extra detail: mist and shadows
        #     ],
        #     [
        #         "A floral arrangement of soft pink roses and white Chinese peony in a pink nickel mug",  # Basic layout
        #         "A floral arrangement of soft pink roses, white Chinese peony, and eucalyptus leaves in a pink nickel mug",  # Add eucalyptus
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink mug, sitting on a thick white book",  # Add apple blossoms
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, in bright sunlight",  # Full details
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, bathed in bright sunlight, with a soft shadow cast on a nearby table"  # Extra detail: shadow
        #     ],
        #     [
        #         "An astronaut standing on Mars during sunset",  # Basic layout
        #         "An astronaut on Mars at sunset, with the Martian landscape stretching into the horizon",  # Add landscape
        #         "An astronaut on Mars at sunset, standing in a rocky terrain, with orange-red hues across the sky",  # Add sky hues
        #         "An astronaut on Mars during sunset, standing in a rocky landscape under the orange-red sky, distant mountains visible in the background",  # Full details
        #         "An astronaut on Mars at sunset, standing in a rocky landscape under a vivid orange-red sky, distant mountains in the background, and dust trails swirling in the wind"  # Extra detail: dust trails
        #     ],
        #     [
        #         "A minimalist logo featuring a surreal cityscape during a rainy night",  # Basic layout
        #         "A minimalist logo showing a surreal cityscape at night, wet streets reflecting neon signs",  # Add neon reflections
        #         "A minimalist logo of a surreal cityscape at night, with wet pavements, glowing neon signs, and reflections in puddles",  # Add puddles
        #         "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflecting in puddles, wet streets lined with towering skyscrapers",  # Full details
        #         "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflected in puddles, towering skyscrapers fading into the mist, with distant lights visible in the fog"  # Extra detail: mist and fog
        #     ],
        #     [
        #         "A small cozy house in the redwoods on a mountain",  # Basic layout
        #         "A small cozy house in the redwoods, with solar panels and a garage on a mountainside",  # Add solar panels
        #         "A small cozy modern house in the redwoods, with solar panels, a garage, and a driveway overlooking the mountain",  # Add driveway
        #         "A small modern house in the redwoods, with solar panels, a garage, driveway, and a great view of the mountains in the sunshine",  # Full details
        #         "A small modern house in the redwoods, with solar panels, a garage, driveway, and a stunning view of the mountains, bathed in sunshine, with a small garden blooming with flowers"  # Extra detail: blooming garden
        #     ],
        #     [
        #         "A new town square in Cambridge, with a big traditional museum",  # Basic layout
        #         "A new town square in Cambridge, with a big traditional museum, a fountain in the center",  # Add fountain
        #         "A new town square in Cambridge, featuring a traditional museum, a large fountain, and surrounding trees",  # Add trees
        #         "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square",  # Full details
        #         "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square, as people stroll along the pathways under the dappled sunlight"  # Extra detail: people and sunlight
        #     ],
        #     [
        #         "A breathtaking cityscape at dusk, illuminated by a warm glow",  # Basic layout
        #         "A breathtaking cityscape at dusk, with a skyline glowing in the golden light of the sunset",  # Add sunset light
        #         "A cityscape at dusk, the skyline bathed in golden light with silhouettes of skyscrapers and distant clouds",  # Add silhouettes
        #         "A breathtaking cityscape at dusk, with the skyline illuminated by a warm, golden glow, silhouetted skyscrapers rising against the glowing sky",  # Full details
        #         "A breathtaking cityscape at dusk, illuminated by a warm, golden glow, with silhouetted skyscrapers rising against the glowing sky, while soft clouds roll in the background"  # Extra detail: clouds rolling
        #     ]
        # ]

        # prompt_schedules = [
        #     [
        #         "A serene lakeside at sunset",  # Basic layout
        #         "A serene lakeside at sunset with reflections of the sky on the water",  # Add reflections
        #         "A serene lakeside at sunset with reflections of the sky on the water, surrounded by trees",  # Add trees
        #         "A serene lakeside at sunset, with reflections of the sky on the water, trees, and distant mountains",  # Add mountains
        #         "A serene lakeside at sunset, with reflections of the sky on the water, trees, distant mountains, and birds flying"  # Add birds
        #     ],
        #     [
        #         "An ancient temple in a dense jungle",  # Basic layout
        #         "An ancient temple in a dense jungle, with vines growing on it",  # Add vines
        #         "An ancient temple in a dense jungle, with vines and moss growing on the stone",  # Add moss
        #         "An ancient temple in a dense jungle, with vines, moss, and plants covering the stone",  # Add plants
        #         "An ancient temple in a dense jungle, with vines, moss, plants, and sunlight shining through the trees"  # Add sunlight
        #     ],
        #     [
        #         "A cyberpunk city at night with neon lights",  # Basic layout
        #         "A cyberpunk city at night with neon lights and flying cars",  # Add flying cars
        #         "A cyberpunk city at night with neon lights, flying cars, and people in the streets",  # Add people
        #         "A cyberpunk city at night with neon lights, flying cars, people, and robots in the streets",  # Add robots
        #         "A cyberpunk city at night with neon lights, flying cars, people, robots, and rain on the streets"  # Add rain
        #     ],
        #     [
        #         "A mountain range at night with a full moon",  # Basic layout
        #         "A mountain range at night with a full moon and stars in the sky",  # Add stars
        #         "A mountain range at night with a full moon, stars, and a lake",  # Add lake
        #         "A mountain range at night with a full moon, stars, a lake, and trees by the shore",  # Add trees
        #         "A mountain range at night with a full moon, stars, a lake, trees, and mist rising from the water"  # Add mist
        #     ],
        #     [
        #         "A floral arrangement of pink roses and white peonies",  # Basic layout
        #         "A floral arrangement of pink roses, white peonies, and eucalyptus leaves",  # Add eucalyptus
        #         "A floral arrangement of pink roses, white peonies, eucalyptus leaves, and apple blossoms",  # Add apple blossoms
        #         "A floral arrangement of pink roses, white peonies, eucalyptus leaves, apple blossoms in a pink mug",  # Add mug
        #         "A floral arrangement of pink roses, white peonies, eucalyptus leaves, apple blossoms in a pink mug on a white book"  # Add book
        #     ],
        #     [
        #         "An astronaut on Mars at sunset",  # Basic layout
        #         "An astronaut on Mars at sunset with rocky terrain",  # Add rocky terrain
        #         "An astronaut on Mars at sunset with rocky terrain and mountains in the background",  # Add mountains
        #         "An astronaut on Mars at sunset with rocky terrain, mountains, and red sky",  # Add sky color
        #         "An astronaut on Mars at sunset with rocky terrain, mountains, red sky, and dust blowing in the wind"  # Add dust
        #     ],
        #     [
        #         "A minimalist logo of a cityscape at night",  # Basic layout
        #         "A minimalist logo of a cityscape at night with neon lights",  # Add neon lights
        #         "A minimalist logo of a cityscape at night with neon lights and wet streets",  # Add wet streets
        #         "A minimalist logo of a cityscape at night with neon lights, wet streets, and reflections in puddles",  # Add puddles
        #         "A minimalist logo of a cityscape at night with neon lights, wet streets, reflections in puddles, and tall buildings"  # Add buildings
        #     ],
        #     [
        #         "A cozy house in the redwoods",  # Basic layout
        #         "A cozy house in the redwoods with solar panels",  # Add solar panels
        #         "A cozy house in the redwoods with solar panels and a driveway",  # Add driveway
        #         "A cozy house in the redwoods with solar panels, a driveway, and a garage",  # Add garage
        #         "A cozy house in the redwoods with solar panels, a driveway, a garage, and a view of the mountains"  # Add view
        #     ],
        #     [
        #         "A town square with a large museum",  # Basic layout
        #         "A town square with a large museum and a fountain",  # Add fountain
        #         "A town square with a large museum, a fountain, and trees",  # Add trees
        #         "A town square with a large museum, a fountain, trees, and benches",  # Add benches
        #         "A town square with a large museum, a fountain, trees, benches, and people walking"  # Add people
        #     ],
        #     [
        #         "A cityscape at dusk with a glowing sky",  # Basic layout
        #         "A cityscape at dusk with a glowing sky and tall buildings",  # Add buildings
        #         "A cityscape at dusk with a glowing sky, tall buildings, and clouds in the sky",  # Add clouds
        #         "A cityscape at dusk with a glowing sky, tall buildings, clouds, and lights turning on",  # Add lights
        #         "A cityscape at dusk with a glowing sky, tall buildings, clouds, lights, and the reflection of the sunset on windows"  # Add reflections
        #     ]
        # ]

        # prompt_schedules = [
        #     [
        #         "A peaceful meadow at dawn, the grass covered in morning dew",  # Basic layout
        #         "A peaceful meadow at dawn, the grass covered in morning dew, with tall wildflowers swaying gently",  # Add wildflowers
        #         "A peaceful meadow at dawn, with dew-covered grass, tall wildflowers, and distant rolling hills visible under the soft morning light",  # Add rolling hills
        #         "A peaceful meadow at dawn, with dew-covered grass, wildflowers, and distant hills bathed in soft light, as the sky begins to warm with hints of orange",  # Full details
        #         "A peaceful meadow at dawn, with dew-covered grass, colorful wildflowers, and distant hills, as the early morning sun rises, casting a warm glow over the landscape"  # Extra detail: warm sunlight
        #     ],
        #     [
        #         "A quiet cobblestone street in an old European town, lined with lantern-lit buildings",  # Basic layout
        #         "A quiet cobblestone street in an old European town, with lantern-lit buildings and ivy creeping up the walls",  # Add ivy
        #         "A quiet cobblestone street in an old European town, lanterns glowing softly, ivy-covered walls, with flower boxes hanging from windows",  # Add flower boxes
        #         "A quiet cobblestone street in an old European town, lanterns glowing, ivy-covered walls, flower boxes, and a distant view of a grand clock tower",  # Full details
        #         "A quiet cobblestone street in an old European town, lantern-lit buildings with ivy, flower boxes, and a grand clock tower, as the soft evening light fades into twilight"  # Extra detail: twilight
        #     ],
        #     [
        #         "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees",  # Basic layout
        #         "A tranquil desert oasis at midday, with a still pool of water surrounded by palm trees and patches of golden sand",  # Add sand
        #         "A tranquil desert oasis at midday, with palm trees reflecting in the still water, surrounded by golden sand and distant rocky cliffs",  # Add distant cliffs
        #         "A tranquil desert oasis at midday, with palm trees, still waters, golden sand, and rocky cliffs under the clear blue sky",  # Full details
        #         "A tranquil desert oasis at midday, with palm trees swaying gently, still waters reflecting the sky, golden sand, and distant rocky cliffs bathed in warm sunlight"  # Extra detail: sunlight on cliffs
        #     ],
        #     [
        #         "A snow-covered village in the mountains at twilight, smoke rising from chimneys",  # Basic layout
        #         "A snow-covered village in the mountains at twilight, with smoke rising from chimneys and warm lights glowing inside the houses",  # Add glowing lights
        #         "A snow-covered village in the mountains at twilight, smoke from chimneys, glowing house lights, and snow-covered pine trees surrounding the village",  # Add snow-covered trees
        #         "A snow-covered village in the mountains at twilight, with smoke rising from chimneys, glowing lights in the houses, snow-covered pine trees, and a clear starry sky above",  # Full details
        #         "A snow-covered village in the mountains at twilight, with glowing house lights, smoke from chimneys, snow-covered trees, and a clear sky full of stars, as the first moonlight begins to appear"  # Extra detail: moonlight
        #     ],
        #     [
        #         "A calm beach at sunrise, with gentle waves lapping at the shore",  # Basic layout
        #         "A calm beach at sunrise, with gentle waves lapping at the shore and footprints in the sand",  # Add footprints
        #         "A calm beach at sunrise, gentle waves, footprints in the sand, and seashells scattered along the shoreline",  # Add seashells
        #         "A calm beach at sunrise, with soft waves, footprints, seashells on the shore, and the sky painted in soft pink and orange hues",  # Full details
        #         "A calm beach at sunrise, with soft waves, seashells, footprints in the sand, and the sky glowing in pink and orange, as the rising sun's first rays touch the horizon"  # Extra detail: first rays of sunlight
        #     ]
        # ]
        
        # prompt_schedules = [
        #     [
        #         "A serene lakeside at sunset, with a calm lake and distant mountains in the background",  # Basic layout
        #         "A serene lakeside at sunset, with a calm lake, distant mountains, and trees along the shore",  # Add trees
        #         "A serene lakeside at sunset, with a calm lake, distant mountains, trees along the shore, and soft golden light reflecting on the water",  # Add golden light
        #         "A serene lakeside at sunset, with a calm lake, distant mountains, trees along the shore, soft golden light, and small birds flying over the water",  # Full details
        #         "A serene lakeside at sunset, with a calm lake, distant mountains, trees along the shore, soft golden light reflecting on the water, and small birds flying as the sun dips below the horizon"  # Extra detail: sunset and birds
        #     ],
        #     [
        #         "An ancient, overgrown temple in a dense jungle, surrounded by tall trees and thick vines",  # Basic layout
        #         "An ancient, overgrown temple in a dense jungle, with tall trees, thick vines, and stone steps leading to the entrance",  # Add stone steps
        #         "An ancient, overgrown temple in a dense jungle, surrounded by tall trees, thick vines, stone steps, and a soft morning light illuminating the scene",  # Add morning light
        #         "An ancient, overgrown temple in a dense jungle, surrounded by tall trees, thick vines, stone steps, and a soft morning light highlighting the intricate carvings on the stone",  # Full details
        #         "An ancient, overgrown temple in a dense jungle, surrounded by tall trees, thick vines, stone steps, and a soft morning light revealing intricate carvings and colorful flowers blooming around the base"  # Extra detail: flowers
        #     ],
        #     [
        #         "A grand castle on a hill, overlooking a vast landscape",  # Basic layout
        #         "A grand castle on a hill, with tall towers, surrounded by lush greenery and a clear blue sky",  # Add towers and greenery
        #         "A grand castle on a hill, with tall towers, lush greenery, and a winding path leading up to the entrance",  # Add winding path
        #         "A grand castle on a hill, with tall towers, lush greenery, a winding path, and colorful flowers lining the way",  # Full details
        #         "A grand castle on a hill, with tall towers, lush greenery, a winding path lined with colorful flowers, and soft clouds drifting in the blue sky"  # Extra detail: clouds
        #     ],
        #     [
        #         "A magnificent lighthouse standing on a rugged cliff by the ocean",  # Basic layout
        #         "A magnificent lighthouse standing on a rugged cliff by the ocean, with waves crashing against the rocks below",  # Add crashing waves
        #         "A magnificent lighthouse on a rugged cliff by the ocean, with crashing waves, a clear sky, and seagulls flying around",  # Add seagulls
        #         "A magnificent lighthouse on a rugged cliff by the ocean, with crashing waves, a clear sky, and seagulls flying as the sun begins to set",  # Full details
        #         "A magnificent lighthouse on a rugged cliff by the ocean, with crashing waves, a clear sky, seagulls flying, and the warm colors of sunset painting the sky"  # Extra detail: sunset colors
        #     ],
        #     [
        #         "A bustling market square with large stalls and colorful banners",  # Basic layout
        #         "A bustling market square filled with large stalls, colorful banners, and a crowd of people shopping",  # Add crowd
        #         "A bustling market square with large stalls, colorful banners, a crowd of people, and various goods displayed prominently",  # Add goods
        #         "A bustling market square with large stalls, colorful banners, a lively crowd of people, and vibrant goods displayed in an inviting manner",  # Full details
        #         "A bustling market square with large stalls, colorful banners, a lively crowd, and vibrant goods displayed attractively, as the sun shines down on the cheerful scene"  # Extra detail: sunlight
        #     ]
        # ]
        # prompt_schedules = [
        #     [
        #         "A busy street with cars parked along the sidewalk and tall buildings in the background",  # Basic layout
        #         "A busy street with cars parked along the sidewalk, tall buildings, and trees lining the street",  # Add trees
        #         "A busy street with cars parked, tall buildings, trees, and pedestrians walking along the sidewalk",  # Add pedestrians
        #         "A busy street with cars parked, tall buildings, trees lining the street, pedestrians walking, and shopfronts on both sides",  # Full details
        #         "A busy street with cars parked, tall buildings, trees, pedestrians walking, shopfronts on both sides, and sunlight reflecting off the windows"  # Extra detail: sunlight reflections
        #     ],
        #     [
        #         "A living room with a large sofa and a coffee table in front of it",  # Basic layout
        #         "A living room with a large sofa, a coffee table, and a television mounted on the wall",  # Add television
        #         "A living room with a large sofa, coffee table, television, and a bookshelf in the corner",  # Add bookshelf
        #         "A living room with a large sofa, coffee table, television, bookshelf, and floor lamp near the sofa",  # Full details
        #         "A living room with a large sofa, coffee table, television, bookshelf, floor lamp, and warm sunlight streaming through the window"  # Extra detail: sunlight
        #     ],
        #     [
        #         "A dining table set with plates and silverware, surrounded by chairs",  # Basic layout
        #         "A dining table set with plates, silverware, and a vase with flowers in the center",  # Add vase with flowers
        #         "A dining table set with plates, silverware, a vase with flowers, and glasses of water",  # Add glasses of water
        #         "A dining table set with plates, silverware, a vase with flowers, glasses of water, and a bowl of fruit in the middle",  # Full details
        #         "A dining table set with plates, silverware, a vase with flowers, glasses of water, a bowl of fruit, and soft light from a chandelier overhead"  # Extra detail: chandelier light
        #     ],
        #     [
        #         "A city park with large trees and open grassy areas",  # Basic layout
        #         "A city park with large trees, open grassy areas, and a few benches scattered around",  # Add benches
        #         "A city park with large trees, open grassy areas, benches, and a walking path winding through",  # Add walking path
        #         "A city park with large trees, open grassy areas, benches, a walking path, and a pond with ducks",  # Full details
        #         "A city park with large trees, open grassy areas, benches, walking path, a pond with ducks, and soft afternoon light filtering through the leaves"  # Extra detail: afternoon light
        #     ],
        #     [
        #         "A large kitchen with modern cabinets and a kitchen island in the middle",  # Basic layout
        #         "A large kitchen with modern cabinets, a kitchen island, and stainless steel appliances",  # Add appliances
        #         "A large kitchen with modern cabinets, kitchen island, stainless steel appliances, and a bowl of fruit on the counter",  # Add bowl of fruit
        #         "A large kitchen with modern cabinets, kitchen island, stainless steel appliances, bowl of fruit, and a window with sunlight streaming in",  # Full details
        #         "A large kitchen with modern cabinets, kitchen island, stainless steel appliances, bowl of fruit, sunlight streaming in, and hanging lights above the island"  # Extra detail: hanging lights
        #     ]
        # ]

        # prompt_schedules = [
        #     [
        #         "A view of Venice from a boat on the river, with tall buildings on both sides and a bridge ahead.",  # Basic layout
        #         "A view of Venice from a boat on the river, with tall red buildings on both sides, and a stone bridge ahead with people walking.",  # Add color and people
        #         "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, and soft lanterns lining the riverbank.",  # Add river color and lanterns
        #         "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, soft lanterns lining the riverbank, and colorful carnival decorations along the buildings.",  # Add carnival decorations
        #         "A view of Venice from a boat on the green river, with tall red buildings on both sides, a stone bridge ahead with people walking, soft lanterns lining the riverbank, colorful carnival decorations along the buildings, and several boats drifting down the river toward the horizon."  # Add boats
        #     ],
        #     [
        #         "A small village at the base of a mountain, with cottages around an open field.",  # Basic layout
        #         "A small village at the base of a mountain, with thatched cottages around an open field and tall pine trees at the village edge.",  # Add pine trees
        #         "A small village at the base of a snow-covered mountain, with thatched cottages around an open field, tall pine trees at the village edge, and a river running beside the village.",  # Add snow and river
        #         "A small village at the base of a snow-covered mountain, with thatched cottages around an open field, tall pine trees at the village edge, a river running beside the village, and smoke rising from the chimneys.",  # Add smoke
        #         "A small village at the base of a snow-covered mountain, with thatched cottages around an open field, tall pine trees at the village edge, a river running beside the village, smoke rising from chimneys, and birds flying in the clear sky."  # Add birds
        #     ],
        #     [
        #         "A bustling city square with tall modern buildings and a central fountain.",  # Basic layout
        #         "A bustling city square with tall modern glass buildings, a central fountain, and trees lining the streets.",  # Add trees
        #         "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, and people sitting on benches.",  # Add benches and people
        #         "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, people sitting on benches, and shopfronts in the background.",  # Add shopfronts
        #         "A bustling city square with tall modern glass buildings, a central fountain, trees lining the streets, people sitting on benches, shopfronts in the background, and sunlight reflecting off the windows."  # Add sunlight
        #     ],
        #     [
        #         "A large forest clearing with a river running through the center, surrounded by tall trees.",  # Basic layout
        #         "A large forest clearing with a river running through the center, surrounded by tall pine trees and large rocks near the riverbank.",  # Add rocks
        #         "A large forest clearing with a river running through the center, surrounded by tall pine trees, large rocks near the riverbank, and a small wooden cabin by the water.",  # Add cabin
        #         "A large forest clearing with a river running through the center, surrounded by tall pine trees, large rocks near the riverbank, a small wooden cabin by the water, and patches of wildflowers scattered across the grass.",  # Add wildflowers
        #         "A large forest clearing with a river running through the center, surrounded by tall pine trees, large rocks near the riverbank, a small wooden cabin by the water, patches of wildflowers scattered across the grass, and soft sunlight filtering through the trees."  # Add sunlight
        #     ]
        # ]

        # prompt_schedules = [
        #     [
        #         "A large city park with a wide open field and a pond in the middle.",  # Basic layout
        #         "A large city park with a wide green field, a pond in the middle, and tall trees surrounding it.",  # Add trees
        #         "A large city park with a wide green field, a pond in the middle, tall trees surrounding it, and ducks swimming in the water.",  # Add ducks
        #         "A large city park with a wide green field, a pond in the middle, tall trees, ducks swimming, and people sitting by the water.",  # Add people
        #         "A large city park with a wide green field, a pond in the middle, tall trees, ducks swimming, people sitting by the water, and the sunlight glistening on the pond’s surface."  # Add sunlight
        #     ],
        #     [
        #         "A view of a sandy beach with waves crashing and a lighthouse in the distance.",  # Basic layout
        #         "A view of a sandy beach with gentle waves crashing, a white lighthouse in the distance, and seagulls flying overhead.",  # Add seagulls
        #         "A view of a sandy beach with gentle waves, a white lighthouse in the distance, seagulls flying, and a wooden pier stretching into the water.",  # Add pier
        #         "A view of a sandy beach with gentle waves, a white lighthouse in the distance, seagulls flying, a wooden pier, and colorful beach umbrellas scattered along the shore.",  # Add umbrellas
        #         "A view of a sandy beach with gentle waves, a white lighthouse in the distance, seagulls flying, a wooden pier, colorful beach umbrellas scattered along the shore, and the sun setting over the horizon casting a golden glow."  # Add sunset
        #     ],
        #     [
        #         "A mountain village with small houses and a tall clock tower in the center.",  # Basic layout
        #         "A mountain village with stone houses, a tall clock tower in the center, and snow-capped peaks in the background.",  # Add mountains
        #         "A mountain village with stone houses, a tall clock tower in the center, snow-capped peaks in the background, and narrow cobblestone streets winding through the village.",  # Add streets
        #         "A mountain village with stone houses, a tall clock tower in the center, snow-capped peaks, cobblestone streets winding through the village, and villagers walking around.",  # Add people
        #         "A mountain village with stone houses, a tall clock tower in the center, snow-capped peaks, cobblestone streets winding through the village, villagers walking around, and smoke rising from chimneys."  # Add smoke
        #     ]
        # ]
        # prompt_schedules = [
        #     [
        #         "A view of a large city square with a tall monument in the center and a road circling it.",  # Basic layout
        #         "A view of a large city square with a tall stone monument in the center, a road circling it, and trees lining the perimeter.",  # Add trees
        #         "A view of a large city square with a tall stone monument, a road circling it, trees lining the perimeter, and benches scattered around.",  # Add benches
        #         "A view of a large city square with a tall stone monument, a road circling it, trees lining the perimeter, benches scattered around, and people walking by.",  # Add people
        #         "A view of a large city square with a tall stone monument, a road circling it, trees, benches, people walking, and cars driving around the square."  # Add cars
        #     ],
        #     [
        #         "A dense forest with tall trees and a small cabin in the distance.",  # Basic layout
        #         "A dense forest with tall pine trees, a small wooden cabin in the distance, and a narrow dirt path leading to the cabin.",  # Add path
        #         "A dense forest with tall pine trees, a small wooden cabin, a dirt path leading to it, and soft sunlight filtering through the leaves.",  # Add sunlight
        #         "A dense forest with tall pine trees, a small wooden cabin, a dirt path, sunlight filtering through, and smoke rising from the cabin chimney.",  # Add smoke
        #         "A dense forest with tall pine trees, a small wooden cabin, a dirt path, sunlight filtering through, smoke from the chimney, and birds flying between the trees."  # Add birds
        #     ],
        #     [
        #         "A medieval castle on a hill, with large stone walls and a moat around it.",  # Basic layout
        #         "A medieval castle on a hill with large stone walls, a moat around it, and tall towers at each corner.",  # Add towers
        #         "A medieval castle on a hill with large stone walls, a moat, tall towers, and a drawbridge over the moat.",  # Add drawbridge
        #         "A medieval castle on a hill with large stone walls, a moat, tall towers, a drawbridge, and flags flying from the towers.",  # Add flags
        #         "A medieval castle on a hill with large stone walls, a moat, tall towers, a drawbridge, flags flying, and knights standing guard at the entrance."  # Add knights
        #     ],
        #     [
        #         "A coastal town with small houses on cliffs overlooking the ocean.",  # Basic layout
        #         "A coastal town with small white houses on cliffs, a vast blue ocean below, and boats floating near the shore.",  # Add boats
        #         "A coastal town with small white houses, cliffs, ocean, boats floating near the shore, and palm trees along the coastline.",  # Add palm trees
        #         "A coastal town with small white houses, cliffs, ocean, boats, palm trees, and people walking along a sandy beach.",  # Add people
        #         "A coastal town with small white houses, cliffs, ocean, boats, palm trees, people on the beach, and the sun setting over the water."  # Add sunset
        #     ],
        #     [
        #         "A wide desert with sand dunes stretching to the horizon and a lone camel walking in the distance.",  # Basic layout
        #         "A wide desert with tall sand dunes, a lone camel walking, and the sun high in the sky.",  # Add sun
        #         "A wide desert with tall sand dunes, a lone camel walking, the sun high, and cactus plants scattered across the sand.",  # Add cactus
        #         "A wide desert with tall sand dunes, a lone camel walking, sun high, cactus plants, and a caravan of travelers far behind the camel.",  # Add caravan
        #         "A wide desert with tall sand dunes, a lone camel walking, sun high, cactus plants, a caravan of travelers, and heat waves shimmering on the horizon."  # Add heat waves
        #     ],
        #     [
        #         "A snowy mountain range with towering peaks and a cabin nestled in the valley.",  # Basic layout
        #         "A snowy mountain range with towering white peaks, a wooden cabin in the valley, and a frozen lake nearby.",  # Add lake
        #         "A snowy mountain range with towering peaks, a wooden cabin in the valley, a frozen lake, and snow-covered trees.",  # Add trees
        #         "A snowy mountain range with towering peaks, a wooden cabin, frozen lake, snow-covered trees, and skiers moving down the slopes.",  # Add skiers
        #         "A snowy mountain range with towering peaks, a wooden cabin, frozen lake, snow-covered trees, skiers, and soft sunlight reflecting off the snow."  # Add sunlight
        #     ],
        #     [
        #         "A tropical rainforest with dense trees and a waterfall in the distance.",  # Basic layout
        #         "A tropical rainforest with dense green trees, a tall waterfall in the distance, and a river flowing from the waterfall.",  # Add river
        #         "A tropical rainforest with dense trees, a waterfall, a river flowing, and colorful birds flying above.",  # Add birds
        #         "A tropical rainforest with dense trees, a waterfall, a river, birds flying, and large tropical flowers scattered throughout the forest.",  # Add flowers
        #         "A tropical rainforest with dense trees, a waterfall, a river, birds, tropical flowers, and mist rising from the waterfall."  # Add mist
        #     ],
        #     [
        #         "A wide-open meadow with tall grass and mountains in the background.",  # Basic layout
        #         "A wide-open meadow with tall green grass, mountains in the background, and a dirt path winding through the field.",  # Add path
        #         "A wide-open meadow with tall grass, mountains, a dirt path, and wildflowers scattered throughout the meadow.",  # Add wildflowers
        #         "A wide-open meadow with tall grass, mountains, a dirt path, wildflowers, and a lone tree standing in the middle of the field.",  # Add tree
        #         "A wide-open meadow with tall grass, mountains, dirt path, wildflowers, a lone tree, and clouds floating lazily above the mountains."  # Add clouds
        #     ],
        #     [
        #         "A small harbor with boats docked along wooden piers and a lighthouse on the shore.",  # Basic layout
        #         "A small harbor with colorful boats docked along wooden piers, a white lighthouse on the shore, and seagulls flying above.",  # Add seagulls
        #         "A small harbor with colorful boats, wooden piers, a white lighthouse, seagulls flying, and fishermen working on the docks.",  # Add fishermen
        #         "A small harbor with colorful boats, wooden piers, white lighthouse, seagulls flying, fishermen working, and waves gently lapping at the boats.",  # Add waves
        #         "A small harbor with colorful boats, wooden piers, white lighthouse, seagulls flying, fishermen working, waves lapping, and the sun low on the horizon casting a golden glow."  # Add sunset
        #     ],
        #     [
        #         "A busy marketplace with stalls selling fruits and vegetables, and a clock tower in the background.",  # Basic layout
        #         "A busy marketplace with colorful stalls selling fruits and vegetables, a tall clock tower in the background, and people browsing the stalls.",  # Add people
        #         "A busy marketplace with colorful stalls, fruits and vegetables, clock tower, people browsing, and vendors shouting out to attract customers.",  # Add vendors
        #         "A busy marketplace with colorful stalls, fruits and vegetables, clock tower, people browsing, vendors shouting, and a cobblestone street running through the market.",  # Add street
        #         "A busy marketplace with colorful stalls, fruits and vegetables, clock tower, people browsing, vendors shouting, cobblestone street, and sunlight filtering through the stalls."  # Add sunlight
        #     ]
        # ]
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
            "MODEL_ID": "stabilityai/stable-diffusion-2-1",
            "DEVICE": "cuda",  # or "cpu"
            "temperature_list": [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],  # List of temperatures to try
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
        # prompt_schedules = [
        #     [
        #         "A serene lakeside at sunset, the sky's vibrant colors reflected on the water",  # Basic layout
        #         "A serene lakeside at sunset, the sky's vibrant colors reflected in the water, with surrounding trees casting long shadows",  # Add shadows
        #         "A serene lakeside at sunset, vibrant sky colors reflected on the still water, with surrounding trees and distant mountains casting long shadows",  # Add distant mountains
        #         "A serene lakeside at sunset, vibrant sky colors reflected in still waters, surrounded by trees and distant mountains with soft golden light",  # Full details
        #         "A serene lakeside at sunset, vibrant sky colors reflected in still waters, with surrounding trees, distant mountains, and small birds flying over the lake as the golden light fades"  # Extra detail: birds and fading light
        #     ],
        #     [
        #         "An ancient, overgrown temple in a dense jungle, with soft light breaking through the canopy",  # Basic layout
        #         "An ancient, overgrown temple in a dense jungle, the soft morning light illuminating vines creeping over stone ruins",  # Add vines
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, vines and moss covering the crumbling stone structure",  # Add moss
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines and moss overtaking the ancient stone ruins as the jungle thrives",  # Full details
        #         "An ancient, overgrown temple in a dense jungle, bathed in soft morning light, with vines, moss, and exotic flowers overtaking the ancient stone ruins as the jungle teems with life"  # Extra detail: exotic flowers and life
        #     ],
        #     [
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights",  # Basic layout
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights, with flying cars zooming past skyscrapers",  # Add flying cars
        #         "A bustling cyberpunk cityscape at night, glowing with neon signs, flying cars zooming past towering skyscrapers, streets crowded with people and robots",  # Add streets and people
        #         "A bustling cyberpunk cityscape at night, bathed in neon lights, flying cars soaring past skyscrapers, crowds of people and robots filling the neon-lit streets below",  # Full details
        #         "A bustling cyberpunk cityscape at night, neon lights reflected in rain-soaked streets, flying cars zooming past skyscrapers as crowds of people and robots navigate the wet city"  # Extra detail: rain-soaked streets
        #     ],
        #     [
        #         "A majestic mountain range under the cloak of night, lit only by a full moon",  # Basic layout
        #         "A majestic mountain range under the night sky, lit by a full moon with stars visible above",  # Add stars
        #         "A majestic mountain range under the night sky, illuminated by a full moon, stars twinkling above and reflecting on a calm lake below",  # Add lake reflection
        #         "A majestic mountain range under a starlit night sky, illuminated by a full moon, stars reflected on a calm lake, with shadows cast by the peaks",  # Full details
        #         "A majestic mountain range under a starlit sky, illuminated by a full moon, stars reflected on a calm lake, with mist rising from the water and shadows dancing on the peaks"  # Extra detail: mist and shadows
        #     ],
        #     [
        #         "A floral arrangement of soft pink roses and white Chinese peony in a pink nickel mug",  # Basic layout
        #         "A floral arrangement of soft pink roses, white Chinese peony, and eucalyptus leaves in a pink nickel mug",  # Add eucalyptus
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink mug, sitting on a thick white book",  # Add apple blossoms
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, in bright sunlight",  # Full details
        #         "A floral arrangement of pink roses, Chinese peony, apple blossoms, and eucalyptus in a pink nickel mug on a thick white book with golden cover, bathed in bright sunlight, with a soft shadow cast on a nearby table"  # Extra detail: shadow
        #     ],
        #     [
        #         "An astronaut standing on Mars during sunset",  # Basic layout
        #         "An astronaut on Mars at sunset, with the Martian landscape stretching into the horizon",  # Add landscape
        #         "An astronaut on Mars at sunset, standing in a rocky terrain, with orange-red hues across the sky",  # Add sky hues
        #         "An astronaut on Mars during sunset, standing in a rocky landscape under the orange-red sky, distant mountains visible in the background",  # Full details
        #         "An astronaut on Mars at sunset, standing in a rocky landscape under a vivid orange-red sky, distant mountains in the background, and dust trails swirling in the wind"  # Extra detail: dust trails
        #     ],
        #     [
        #         "A minimalist logo featuring a surreal cityscape during a rainy night",  # Basic layout
        #         "A minimalist logo showing a surreal cityscape at night, wet streets reflecting neon signs",  # Add neon reflections
        #         "A minimalist logo of a surreal cityscape at night, with wet pavements, glowing neon signs, and reflections in puddles",  # Add puddles
        #         "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflecting in puddles, wet streets lined with towering skyscrapers",  # Full details
        #         "A minimalist logo of a surreal cityscape during a rainy night, glowing neon signs reflected in puddles, towering skyscrapers fading into the mist, with distant lights visible in the fog"  # Extra detail: mist and fog
        #     ],
        #     [
        #         "A small cozy house in the redwoods on a mountain",  # Basic layout
        #         "A small cozy house in the redwoods, with solar panels and a garage on a mountainside",  # Add solar panels
        #         "A small cozy modern house in the redwoods, with solar panels, a garage, and a driveway overlooking the mountain",  # Add driveway
        #         "A small modern house in the redwoods, with solar panels, a garage, driveway, and a great view of the mountains in the sunshine",  # Full details
        #         "A small modern house in the redwoods, with solar panels, a garage, driveway, and a stunning view of the mountains, bathed in sunshine, with a small garden blooming with flowers"  # Extra detail: blooming garden
        #     ],
        #     [
        #         "A new town square in Cambridge, with a big traditional museum",  # Basic layout
        #         "A new town square in Cambridge, with a big traditional museum, a fountain in the center",  # Add fountain
        #         "A new town square in Cambridge, featuring a traditional museum, a large fountain, and surrounding trees",  # Add trees
        #         "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square",  # Full details
        #         "A new town square in Cambridge, with a traditional museum, fountain, classical design, and trees lining the square, as people stroll along the pathways under the dappled sunlight"  # Extra detail: people and sunlight
        #     ],
        #     [
        #         "A breathtaking cityscape at dusk, illuminated by a warm glow",  # Basic layout
        #         "A breathtaking cityscape at dusk, with a skyline glowing in the golden light of the sunset",  # Add sunset light
        #         "A cityscape at dusk, the skyline bathed in golden light with silhouettes of skyscrapers and distant clouds",  # Add silhouettes
        #         "A breathtaking cityscape at dusk, with the skyline illuminated by a warm, golden glow, silhouetted skyscrapers rising against the glowing sky",  # Full details
        #         "A breathtaking cityscape at dusk, illuminated by a warm, golden glow, with silhouetted skyscrapers rising against the glowing sky, while soft clouds roll in the background"  # Extra detail: clouds rolling
        #     ]
        # ]
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