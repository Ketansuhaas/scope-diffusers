import torch
from scope_diffuser import SCoPEDiffusionPipeline as sdp_scope
from diffusers import StableDiffusionPipeline as sdp
import matplotlib.pyplot as plt
import numpy as np
from ezcolorlog import root_logger as logger
from config import config

# Load the model
pipe = sdp_scope.from_pretrained(
    config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
)
pipe = pipe.to(config["DEVICE"])

image_scope_list = []

for step_size in config["step_sizes"]:

    logger.info(f"Running with step size: {step_size}")

    prompt_schedule = [
        (0, "A marketplace at night", None),
        (step_size, "An Indian marketplace at night with small shops", None),
        (
            step_size * 2,
            "An Indian marketplace at night with small shops, and people walking around in groups",
            None,
        ),
        (
            step_size * 3,
            "An Indian marketplace at night with small shops selling vegetables, and people walking around in groups ",
            None,
        ),
        (
            step_size * 4,
            "An Indian marketplace at night under a full moon with small shops selling vegetables, and people walking around in groups taking on phones",
            None,
        ),
    ]

    logger.info(f"Running SCoPE Diffusion on the prompt schedule: {prompt_schedule}")

    torch.manual_seed(config["seed"])
    image = pipe(
        prompt_schedule,
        num_inference_steps=config["num_inference_steps"],
        callback=None,
        callback_steps=1,
    ).images[0]

    # Convert PIL Image to numpy array
    image_scope_list.append(np.array(image))


logger.info("Running normal Stable Diffusion")
# Load the normal Stable Diffusion model
pipe = sdp.from_pretrained(
    config["MODEL_ID"], torch_dtype=torch.float16, low_cpu_mem_usage=True
)
pipe = pipe.to(config["DEVICE"])
torch.manual_seed(config["seed"])
image = pipe(
    prompt_schedule[-1][1],
    num_inference_steps=config["num_inference_steps"],
    callback=None,
    callback_steps=1,
).images[0]

# Convert PIL Image to numpy array
image_normal = np.array(image)

# Plotting
plt.figure(figsize=(20, 8))
for idx, image_scope in enumerate(image_scope_list):
    plt.subplot(1, len(config["step_sizes"]) + 1, idx + 1)
    plt.axis("off")
    plt.imshow(image_scope)
    plt.title(f"Step size = {config['step_sizes'][idx]}")

plt.subplot(1, len(config["step_sizes"]) + 1, len(config["step_sizes"]) + 1)
plt.axis("off")
plt.imshow(image_normal)
plt.title("Normal image")

# Adjust layout to make space for text
plt.tight_layout()

# Add the text below the plots
plt.figtext(0.5, 0.1, prompt_schedule[-1][1], ha="center", fontsize=15, wrap=True)

# Display the plot with text
plt.savefig("output.png")
