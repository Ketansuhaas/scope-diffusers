from openai import OpenAI
import pandas as pd
import json
from dotenv import load_dotenv
from dataset import GenAIDataset
import os

# Load environment variables from .env (ensure your API key is stored there)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to call OpenAI API for each prompt
def get_progressive_prompts(initial_prompt):
    SYSTEM_PROMPT = f"""
    I need you to return a python list of 5 prompts. You must follow these rules: 
    1. Consider the given prompt as the first prompt. 
    2. Progressively increase the details (no audio-based or smell-based details), one unique attribute added per prompt
    3. Prioritize significant details of the image first, separated by commas without conjunctions.
    4. Keep the SAME prefix in the successive prompts, and any details in the previous prompts should not be missed in successive prompts.
    5. All the prompts should be within 77 tokens of the CLIP text encoder.

    Create the Python list for this prompt below, considering it as the first prompt:
    <{initial_prompt}>
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

# Initialize the dataset object and create DataFrame
genai_dataset = GenAIDataset()
df = genai_dataset.create_dataframe()

# Get the prompts with top 50 most number of adjectives
df = df.sort_values(by='Num_Adjectives', ascending=False)
prompts = df['Prompt'].head(50).tolist()

# Create an empty list to store the responses in JSON format
json_format_responses = []

# Call OpenAI for each prompt and save the response
for prompt in prompts:
    progressive_prompts = get_progressive_prompts(prompt)
    json_format_responses.append({
        "initial_prompt": prompt,
        "progressive_prompts": progressive_prompts
    })

# Save the results into a JSON file for future use
with open('scope_prompts_responses.json', 'w') as f:
    json.dump(json_format_responses, f, indent=4)

print("Results saved in scope_prompts_responses.json")