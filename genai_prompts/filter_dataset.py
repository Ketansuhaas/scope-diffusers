import wandb 
from dataset import GenAIDataset

wandb.init(
        project="SCoPE_Diffusion_Project",  # Your main project name
        group=f"genai_bench",  # Group by experiment type and name
        name=f"filtered_df",  # Unique run name
    )
# Initialize the dataset object
genai_dataset = GenAIDataset()

# Create a DataFrame for the first 100 samples (or any number you prefer)
df = genai_dataset.create_dataframe()

df.to_csv('genai_dataset.csv', index=False)

table = wandb.Table(dataframe=df)

# Add the table to W&B
wandb.log({"genai_bench_table": table})

# Finish the W&B run
wandb.finish()

# Print the DataFrame (optional)
print(df.head())
