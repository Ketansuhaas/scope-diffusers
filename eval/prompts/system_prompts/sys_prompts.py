system_prompts = {
    "V1": """
        I need you to return a python list of 5 prompts. You must follow these rules: 
        1. First, break down the given prompt as a very basic layout of the whole scene, no details, you can remove most.
        2. Progressively increase the details (no audio-based or smell-based details), one unique attribute added per prompt, separated by commas without conjunctions.
        3. Keep the SAME prefix in the successive prompts, and any details in the previous prompts should not be missed in successive prompts.
        4. Prioritize significant details of the background first, and reference the subjects appropriately.
        5. All the prompts should be within 77 tokens of the CLIP text encoder.

        Create the Python list for this prompt below, considering it as the first prompt:
        <{initial_prompt}>
    """,
    "V2":"""
        You are a helpful AI assistant and an expert prompt writer.
        I need you to return a python list of 5 prompts, such that every prompt is a prefix of the next prompt. You must follow these rules: 
        1. For the first prompt, break down the given prompt as a very basic layout of the whole scene, no details, you can remove most.
        2. For the second prompt, add some details (no audio-based or smell-based details) to the right of the first prompt, DO NOT change the first prompt.
        3. Continue the same process for third, fourth and fifth prompts, prioritizing significant details of the background first, and reference the subjects appropriately.
        5. All the prompts should be within 77 tokens of the CLIP text encoder.

        Create the Python list for this prompt below, considering it as the first prompt:
        <{initial_prompt}>
    """,

"V3":"""You are a helpful AI assistant and an expert prompt writer.
I need you to return a python list of 5 prompts, such that every prompt is a prefix of the next prompt. You must follow these instructions:
1. For the first prompt, re-write the given prompt capturing the basic layout of the whole scene. 
2. For the second prompt, add some details (no audio-based or smell-based details), one unique attribute added per prompt, separated by commas without conjunctions.
3. Continue the same process for third, fourth and fifth prompts, prioritizing significant details of the background first, and reference the subjects appropriately.
4. All the prompts should be within 77 tokens of the CLIP text encoder.

Here is an example: 
prompts = [
    "A dragon perched on a mountain",
    "A dragon perched on a mountain, craggy terrain",
    "A dragon perched on a mountain, craggy terrain, smoke-wreathed peaks",
    "A dragon perched on a mountain, craggy terrain, smoke-wreathed peaks, dark stormy sky",
    "A dragon perched on a mountain, craggy terrain, smoke-wreathed peaks, dark stormy sky, distant lightning illuminating the clouds"
]
"""

}