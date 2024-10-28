import pandas as pd
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

def get_prompt_schedule(prompt_schedule_text):
    match = re.search(r'\[([\s\S]*?)\]', prompt_schedule_text)
    prompt_schedule_match = match.group(1)
    prompt_schedule_list = [prompt.strip().strip('"') for prompt in prompt_schedule_match.split('",')]
    prompt_schedule_list = [prompt for prompt in prompt_schedule_list if prompt]  # Remove empty prompts
    return prompt_schedule_list

def get_preprocessed_prompt_lists(param = 'num_nouns', count = 10,ascending=True):
    df = pd.read_csv('genai_dataset_preprocessed_stats.csv')
    # Sort the DataFrame by 'num_nouns' in descending order
    df_sorted = df.sort_values(by=param, ascending=ascending)
    prompts = []
    for i in range(count):
        ps = get_prompt_schedule(df_sorted.iloc[i]['prompt_schedule'])
        prompts.append(ps)
    return prompts