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
