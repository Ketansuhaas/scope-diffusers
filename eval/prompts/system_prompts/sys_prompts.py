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
    """
}