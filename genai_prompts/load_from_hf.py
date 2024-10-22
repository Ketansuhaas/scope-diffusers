from datasets import load_dataset

# Load the GenAI-Bench dataset
dataset = load_dataset("BaiqiL/GenAI-Bench")

# Print the column names
columns = dataset['train'].column_names
print("Columns: ", columns)
