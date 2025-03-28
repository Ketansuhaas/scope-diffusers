import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

from interpolator.interpolator import get_interpolator
from helpers import build_step_callback

# === Main Example: Hooking NLerp into StableDiffusionPipeline ===

# -------------------------------
# Main: Run Pipeline with Hook
# -------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the Stable Diffusion Pipeline from diffusers.
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    
    # Define a progressive prompt schedule.
    prompt_schedule_list = [
        "a photo of a cat",
        "a photo of a cat in a hat",
        "a photo of a cat in a hat on a mat",
        "a photo of a cat in a hat on a mat with a bat",
        "a photo of a cat in a hat on a mat with a bat eating a rat"
    ]
    # Define stage times corresponding to each prompt (e.g., diffusion steps 0, 25, 50).
    stage_times = [None, None, None, None, 25]
    
    # Encode prompt embeddings for each stage.
    # Use the internal _encode_prompt API (subject to change in diffusers).
    prompt_embeddings = []
    for prompt in prompt_schedule_list:
        embeds = pipe._encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )  # Expected shape: (batch, seq_len, embed_dim)
        prompt_embeddings.append(embeds)
    # Stack embeddings: shape becomes (num_stages, batch, seq_len, embed_dim)
    prompt_embeddings = torch.stack(prompt_embeddings, dim=0)
    interpolator = get_interpolator(prompt_embeddings, stage_times, method="nlerp_og_dynamic_stdev", std_dev=3.0, device=device)

    # Build the callback using our interpolator.
    step_callback = build_step_callback(interpolator)
    
    # Use the final prompt as a fallback.
    final_prompt = prompt_schedule_list[-1]
    
    # Run the pipeline with the callback.
    # The callback_on_step_end hook will update "prompt_embeds" on every diffusion step.
    output = pipe(
        prompt=final_prompt,
        num_inference_steps=50,
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["prompt_embeds"]
    )
    
    image = output.images[0]
    image.save("output_nlerp_hook.png")
    print("Image saved as output_nlerp_hook.png")


if __name__ == "__main__":
    main()