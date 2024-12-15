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
from encode_prompt import encode_prompt

from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)


import random
import torch


def attention_interpolate_embeddings(embeddings, times, time_i):

        device = embeddings.device
        
        if time_i >= times[-1]:
            return embeddings[-1]

        import numpy as np
        q = len(times)-1

        tau = times[1]*(1 - (time_i/times[-1])) + 0.1 

        distances = np.zeros((embeddings.shape[2],q))
        for idx in range(embeddings.shape[0]-1):
            e1 = embeddings[idx].detach().cpu().numpy()
            e2 = embeddings[idx+1].detach().cpu().numpy()
            for i in range(e1.shape[1]):  # Iterate over the second dimension (77)
                euclidean_distance = np.linalg.norm(e1[-1,i,:] - e2[-1,i,:])                    
                distances[i][idx] = euclidean_distance
        # Normalize each row by dividing by its sum
        row_sums = distances.sum(axis=1, keepdims=True)
        distances_normalized = distances / (row_sums+1e-5)
        # Multiply all elements by times[-1]
        distances = distances_normalized * times[-1]
        distances = np.cumsum(distances, axis=1)
        zero_column = np.zeros((distances.shape[0], 1))
        # Concatenate the zero column with the cumulative sum
        times = np.hstack((zero_column, distances))
        #============ABOVE IS OPTIONAL============================================ (times[i]->times)

        # Initialize the interpolated embedding with the same shape as input embeddings
        interpolated_embedding = embeddings[0].clone().detach()            
        # Perform weighted sum token-wise across prompts
        for i in range(embeddings[0].shape[1]):  # Handle all weights zero due to similar tokens
            if row_sums[i]==0:
                continue
            # Calculate weights using the exponential function for all points
            # print("times: ",times[i])
            weights = torch.tensor([np.exp(-((t - time_i)/ (tau))**2/2) for t in times[i]], device=device)
            # Normalize weights
            # print("weights before norm", weights)
            weights /= weights.sum()
            # print("weights after norm", weights)
            weights = weights.unsqueeze(1)
            # print(f"token {i}")
            token_feature_values = torch.stack([embeddings[k,-1, i, :] for k in range(embeddings.shape[0])])  # same token from all prompts
            # print("all prompts: ",token_feature_values.shape)
            interpolated_value = torch.sum(token_feature_values * weights,dim=0)
            # print(f"interpolated token {interpolated_value.shape}")
            interpolated_embedding[-1, i, :] = interpolated_value
            # print(f"replaced in the final embeddings: {interpolated_embedding.shape}")
            original_magnitude = torch.norm(embeddings[0][-1, i, :])
            current_magnitude = torch.norm(interpolated_embedding[-1, i, :])
            # print(f"original norm {original_magnitude}, current norm {current_magnitude}")
            interpolated_embedding[-1, i, :] *= (original_magnitude / current_magnitude)
            # print(interpolated_embedding[-1,i,:])
            # print('corrected: ',torch.norm(interpolated_embedding[-1,i,:]))


        return interpolated_embedding.to(device)


from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

with torch.no_grad():
    

    @torch.no_grad()
    #@replace_example_docstring(EXAMPLE_DOC_STRING)
    def get_image(
        pipe,
        prompt_schedule = None,
        prompt: Union[str, List[str]] = "This is a placeholder",
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or pipe.default_sample_size * pipe.vae_scale_factor
        width = width or pipe.default_sample_size * pipe.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        pipe.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        pipe._guidance_scale = guidance_scale
        pipe._guidance_rescale = guidance_rescale
        pipe._clip_skip = clip_skip
        pipe._cross_attention_kwargs = cross_attention_kwargs
        pipe._denoising_end = denoising_end
        pipe._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = pipe._execution_device

        # 3. Encode input prompt
        lora_scale = (
            pipe.cross_attention_kwargs.get("scale", None) if pipe.cross_attention_kwargs is not None else None
        )

        # set prompt stage timings
        stage_times = [st for (st, _) in prompt_schedule]
        prompt_embeddings = []
        for _, prompt in prompt_schedule:
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = encode_prompt(
                pipe=pipe,
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                negative_prompt=None,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=lora_scale,
                clip_skip=pipe.clip_skip,
            )

            prompt_embeddings.append(prompt_embeds)

        prompt_embeddings = torch.stack(prompt_embeddings, dim=0)                    

        prompt_embeds = attention_interpolate_embeddings(              #interpolate positive prompts
            prompt_embeddings, stage_times, 0
        )
        # print(prompt_embeds.shape)
        # exit()

    
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

        add_time_ids = pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if pipe.do_classifier_free_guidance:
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)


        if ip_adapter_image is not None:
            image_embeds = pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            pipe.denoising_end is not None
            and isinstance(pipe.denoising_end, float)
            and pipe.denoising_end > 0
            and pipe.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    pipe.scheduler.config.num_train_timesteps
                    - (pipe.denoising_end * pipe.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(pipe.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = pipe.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        pipe._num_timesteps = len(timesteps)
        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                #---------------------------------------------------------
                prompt_embeds = attention_interpolate_embeddings(
                    prompt_embeddings, stage_times, i
                )
                #---------------------------------------------------------

                if pipe.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

                prompt_embeds = prompt_embeds.to(device)
                #---------------------------------------------------------------------------------------
                if pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if pipe.do_classifier_free_guidance else latents

                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                    
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if pipe.do_classifier_free_guidance and pipe.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=pipe.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(pipe.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast

            if needs_upcasting:
                pipe.upcast_vae()
                latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)

            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                pipe.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if pipe.watermark is not None:
                image = pipe.watermark.apply_watermark(image)

            image = pipe.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)