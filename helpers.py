# === Callback Function to Inject Interpolated Embeddings ===

def build_step_callback(interpolator):
    """
    Returns a callback function that updates the prompt embeddings at each diffusion step.
    """
    def callback(pipeline, step, timestep, callback_kwargs):
        # Update prompt_embeds using the interpolator for the current step.
        callback_kwargs["prompt_embeds"] = interpolator(step)
        return callback_kwargs
    return callback

def build_step_callback_sdxl(interpolator, pooled_interpolator, interpolate_prompt_embeds=True, interpolate_pooled_prompt_embeds=True):
    def callback(pipeline, step, timestep, callback_kwargs):
        if interpolate_prompt_embeds:
            callback_kwargs["prompt_embeds"] = interpolator(step)
        if interpolate_pooled_prompt_embeds:
            callback_kwargs["add_text_embeds"] = pooled_interpolator(step).squeeze(0)
        return callback_kwargs
    return callback

def build_step_callback_flux(interpolator, pooled_interpolator):
    """
    Returns a callback function for Flux pipelines that updates the prompt embeddings
    as well as the pooled prompt embeddings.

    Flux's allowed callback tensor inputs only include "prompt_embeds", so we update that key
    with the main (non-pooled) interpolated embeddings, and then directly set an attribute
    (e.g. pipeline.pooled_prompt_embeds) with the interpolated pooled embeddings.
    """
    def callback(pipeline, step, timestep, callback_kwargs):
        # Update the main prompt embeddings via the interpolator.
        callback_kwargs["prompt_embeds"] = interpolator(step)
        # Directly set the pipeline's pooled prompt embeddings.
        pipeline.pooled_prompt_embeds = pooled_interpolator(step)
        return callback_kwargs
    return callback

import itertools

def get_all_hparam_combinations(interpolator_cls):
    grid = interpolator_cls.hparam_grid()
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]