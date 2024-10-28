import pandas as pd
import ast

def filter_dataset(dataset, filter_tags: dict, num_filter: int, filter_by: str):
    """
    Filters the dataset based on specified tags in the 'basic' category only 
    and sorts it based on a specified criterion.
    """
    # Make a copy of the dataset to avoid modifying the original one
    df = dataset.copy()

    # Convert 'Tags' column to a dictionary if it is in string format
    if isinstance(df['Tags'].iloc[0], str):
        df['Tags'] = df['Tags'].apply(ast.literal_eval)

    # Function to check if a row matches the given filter tags in the 'basic' category only
    def match_basic_tags(entry):
        # Ensure all specified tags in the 'basic' category are present
        basic_tags = filter_tags.get('basic', [])
        if basic_tags:
            return all(tag in entry.get('basic', []) for tag in basic_tags)
        return True  # If no basic tags to filter, return True

    # Apply the filtering function
    df_filtered = df[df['Tags'].apply(match_basic_tags)]

    # Sorting the dataframe based on the filter_by criteria
    if filter_by in ['Num_Tokens', 'Num_Words', 'Num_Sentences','Num_Adjectives']:
        df_filtered = df_filtered.sort_values(by=filter_by, ascending=False)
    else:
        raise ValueError(f"Invalid filter_by value: {filter_by}")

    # Return the top num_filter entries
    return df_filtered.head(num_filter)
