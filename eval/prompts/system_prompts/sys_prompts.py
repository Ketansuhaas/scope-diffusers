system_prompts = {
    "V1": """
        I need you to return a python list of 5 prompts. You must follow these rules: 
        1. Consider the given prompt as the first prompt. 
        2. Progressively increase the details (no audio-based or smell-based details), one unique attribute added per prompt
        3. Prioritize significant details of the image first, separated by commas without conjunctions.
        4. Keep the SAME prefix in the successive prompts, and any details in the previous prompts should not be missed in successive prompts.
        5. All the prompts should be within 77 tokens of the CLIP text encoder.

        Create the Python list for this prompt below, considering it as the first prompt:
        <{initial_prompt}>
    """,
    "V2": """
        I need you to return a python list of 5 prompts for text-image diffusion. You must follow these rules: 
        1. First, break down the given prompt as a very basic layout of the whole scene, no details, you can remove most.
        2. Progressively increase the details (no audio-based or smell-based details), one unique attribute added per prompt, separated by commas without conjunctions.
        3. Keep the SAME prefix in the successive prompts, and any details in the previous prompts should not be missed in successive prompts.
        4. Prioritize significant details of the background first, and reference the subjects appropriately.
        5. All the prompts should be within 77 tokens of the CLIP text encoder.

        Create the Python list for this prompt below, considering it as the first prompt:
        <{initial_prompt}>
    """
}