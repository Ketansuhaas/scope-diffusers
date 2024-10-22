import pandas as pd
from datasets import load_dataset
from transformers import CLIPTokenizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os
from tqdm import tqdm


# Set up a custom download directory for NLTK data
nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Set the NLTK data path explicitly
nltk.data.path.append(nltk_data_dir)

# Download the required resources, specifying the download directory
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)


class GenAIDataset:
    def __init__(self):
        # Load the GenAI-Bench dataset
        self.dataset = load_dataset("BaiqiL/GenAI-Bench")
        # Initialize the CLIP tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def get_columns(self):
        # Return the columns in the dataset
        return self.dataset['train'].column_names
    
    def get_prompt(self, index):
        # Retrieve the prompt by index
        return self.dataset['train']['Prompt'][index]

    def get_human_ratings(self, index):
        # Retrieve human ratings for each model at a specific index
        return self.dataset['train']['HumanRatings'][index]

    def analyze_prompt(self, prompt):
        # Tokenize using NLTK to count the actual number of words
        words = word_tokenize(prompt)
        sentences = sent_tokenize(prompt)
        
        # POS tagging for counting adjectives, nouns, verbs, etc.
        pos_tags = nltk.pos_tag(words)

        # Use CLIP tokenizer to tokenize the prompt and count tokens
        clip_tokens = self.tokenizer.tokenize(prompt)

        # Count different parts of speech
        num_adjectives = len([word for word, pos in pos_tags if pos.startswith('JJ')])
        num_nouns = len([word for word, pos in pos_tags if pos.startswith('NN')])
        num_verbs = len([word for word, pos in pos_tags if pos.startswith('VB')])
        num_adverbs = len([word for word, pos in pos_tags if pos.startswith('RB')])

        # Tokenized prompt contains the token count and sentence count based on the tokenizer
        analysis = {
            'num_words': len(words),           # Actual number of words using NLTK
            'num_sentences': len(sentences),   # Number of sentences
            'num_tokens': len(clip_tokens),    # Number of tokens (from CLIP tokenizer)
            'num_adjectives': num_adjectives,  # Number of adjectives
            'num_nouns': num_nouns,            # Number of nouns
            'num_verbs': num_verbs,            # Number of verbs
            'num_adverbs': num_adverbs         # Number of adverbs
        }
        return analysis

    def create_dataframe(self, num_samples=None):
        # Initialize lists for dataframe
        indices = []
        prompts = []
        num_words = []
        num_sentences = []
        num_tokens = []
        num_adjectives = []
        num_nouns = []
        num_verbs = []
        num_adverbs = []
        human_ratings = []

        # Iterate over dataset to gather data
        for i in range(len(self.dataset['train']) if not num_samples else num_samples):
            indices.append(i)  # Store the dataset index
            prompt = self.get_prompt(i)
            prompts.append(prompt)

            # Analyze the prompt using NLTK and CLIP tokenizer
            analysis = self.analyze_prompt(prompt)
            num_words.append(analysis['num_words'])
            num_sentences.append(analysis['num_sentences'])
            num_tokens.append(analysis['num_tokens'])
            num_adjectives.append(analysis['num_adjectives'])
            num_nouns.append(analysis['num_nouns'])
            num_verbs.append(analysis['num_verbs'])
            num_adverbs.append(analysis['num_adverbs'])

            # Get human ratings
            ratings = self.get_human_ratings(i)
            human_ratings.append(ratings)

        # Create DataFrame
        df = pd.DataFrame({
            'Index': indices,                 # Include the dataset index
            'Prompt': prompts,
            'Num_Words': num_words,           # Actual word count
            'Num_Sentences': num_sentences,   # Sentence count
            'Num_Tokens': num_tokens,         # Token count (from CLIP tokenizer)
            'Num_Adjectives': num_adjectives, # Count of adjectives
            'Num_Nouns': num_nouns,           # Count of nouns
            'Num_Verbs': num_verbs,           # Count of verbs
            'Num_Adverbs': num_adverbs,       # Count of adverbs
            'Human_Ratings': human_ratings    # Human evaluation scores
        })

        return df
