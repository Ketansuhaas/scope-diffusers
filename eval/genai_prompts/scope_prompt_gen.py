from openai import OpenAI
import pandas as pd
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env (ensure your API key is stored there)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to call OpenAI API for each prompt
def get_progressive_prompts(sys_prompt, initial_prompt):
    SYSTEM_PROMPT = sys_prompt
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create the Python list for this prompt below, considering it as the first prompt: {initial_prompt}"}
        ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    print(messages)
    
    return response.choices[0].message.content.strip()


def get_progressive_prompts_from_scratch(sys_prompt):
    SYSTEM_PROMPT = sys_prompt
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    print(messages)
    
    return response.choices[0].message.content.strip()