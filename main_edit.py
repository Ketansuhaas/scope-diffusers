import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from interpolator.interpolator import get_interpolator
from helpers import build_step_callback
import os

def encode_prompt_schedule(pipe, prompts, device):
    """
    Encodes a list of text prompts into embeddings with classifier-free guidance on.
    Returns a stacked tensor of shape [num_prompts, 2, seq_len, embed_dim],
    where dimension 1 is (negative_embeds, positive_embeds).
    """
    prompt_embeddings = []
    for prompt in prompts:
        embeds = pipe._encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True
        )
        prompt_embeddings.append(embeds)
    return torch.stack(prompt_embeddings, dim=0)

def run_coarse_to_fine_edit(
    model_name="runwayml/stable-diffusion-v1-5",
    prompt_schedule=[
    "a fluffy cat sitting on a wooden floor",
    "a fluffy cat sitting on a wooden floor next to a houseplant",
    "a fluffy cat sitting near a window on a wooden floor with soft daylight and houseplants around",
    "a fluffy cat with green eyes sitting on a cozy rug in a sunlit room with wooden floors, large windows, and several potted plants"
    ],
    interpolation_method="nlerp_og",
    num_steps=50,
    strength=0.8,
    seed=42,
    save_dir="img2img_scope_test"
):
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Generate initial image from first prompt
    pipe_txt2img = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    init_result = pipe_txt2img(prompt_schedule[0], num_inference_steps=50).images[0]
    init_path = os.path.join(save_dir, "init_cat.png")
    init_result.save(init_path)
    print(f"Initial image saved to {init_path}")

    # Step 2: Encode prompt schedule
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    prompt_embeddings = encode_prompt_schedule(pipe_img2img, prompt_schedule, device)

    # Step 3: Build interpolator
    interpolator_cls = get_interpolator(interpolation_method)
    interpolator = interpolator_cls(
        embeddings=prompt_embeddings,
        interpolation_period=21,
        device=device
    )
    step_callback = build_step_callback(interpolator)

    # Step 4: Generate edited image using img2img
    init_image = init_result.resize((512, 512)).convert("RGB")
    initial_embed = interpolator(0)
    pos_embed = initial_embed[1].unsqueeze(0)
    neg_embed = initial_embed[0].unsqueeze(0)

    result = pipe_img2img(
        image=init_image,
        strength=strength,
        num_inference_steps=num_steps,
        prompt_embeds=pos_embed,
        negative_prompt_embeds=neg_embed,
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["prompt_embeds"]
    )

    result_image = result.images[0]
    result_path = os.path.join(save_dir, "scope_edit.png")
    result_image.save(result_path)
    print(f"Edited image saved to {result_path}")

if __name__ == "__main__":
    run_coarse_to_fine_edit()