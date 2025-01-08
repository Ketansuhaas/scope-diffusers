import openai
import time, sys


def openai_completion(prompt, engine="gpt-4o-mini", max_tokens=2048, temperature=0):
    client = openai.OpenAI()
    
    resp =  client.chat.completions.create(
        model=engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        )
    
    return resp.choices[0].message.content



