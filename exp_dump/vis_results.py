import gradio as gr
import os
import random

# Base directory for your images and text files
BASE_DIR = "spat_rel50/num_inference_200_TEMP_1.0_STEP_SIZE_5_SEED_42/prompt_exp_V3_filter_advanced_none_basic_Spatial_Relation_num_prompts_50_filter_Num_Tokens"

def get_available_indices(base_dir):
    """Get all subdirectories in base_dir that have a numeric name."""
    indices = sorted([int(folder) for folder in os.listdir(base_dir) if folder.isdigit()])
    print(f"Available indices: {indices}")
    return indices

# Function to extract indices
available_indices = get_available_indices(BASE_DIR)

def display_and_select(idx, mix_up=False):
    """Display the images and return the prompt and image paths."""
    print(f"Displaying index: {idx}")
    
    # Construct file paths
    normal_image_path = os.path.join(BASE_DIR, str(idx), "normal_image.png")
    scope_image_path = os.path.join(BASE_DIR, str(idx), "scope_image.png")
    text_path = os.path.join(BASE_DIR, str(idx), "prompt_schedule.txt")
    
    # Check if the image and text files exist
    if not os.path.exists(normal_image_path) or not os.path.exists(scope_image_path):
        print(f"One or both images not found for index {idx}: {normal_image_path}, {scope_image_path}")
        return "One or both images not found!", None, None, idx
    if not os.path.exists(text_path):
        print(f"Prompt text not found for index {idx}")
        return None, "Prompt not found!", None, idx
    
    # Read the text
    with open(text_path, "r") as file:
        prompt = file.read()
    
    # No mix-up logic in Viewer Mode
    left_image, right_image = normal_image_path, scope_image_path

    return prompt, left_image, right_image, idx

def next_index(current_idx):
    """Increment the index based on the list of available indices."""
    try:
        current_pos = available_indices.index(current_idx)
        new_pos = min(current_pos + 1, len(available_indices) - 1)
        new_idx = available_indices[new_pos]
    except ValueError:
        new_idx = available_indices[0]  # Default to the first available index if not found
    print(f"Next index: {new_idx}")
    return new_idx

def previous_index(current_idx):
    """Decrement the index based on the list of available indices."""
    try:
        current_pos = available_indices.index(current_idx)
        new_pos = max(current_pos - 1, 0)
        new_idx = available_indices[new_pos]
    except ValueError:
        new_idx = available_indices[0]  # Default to the first available index if not found
    print(f"Previous index: {new_idx}")
    return new_idx

# Create Gradio interface
with gr.Blocks() as gr_interface:
    gr.Markdown("# Viewer Mode")

    # Dropdown input for selecting image pair index without interaction
    viewer_idx_input = gr.Number(label="Image Index (idx)", value=available_indices[0], interactive=True)

    # Display elements for viewer mode
    viewer_prompt_display = gr.Textbox(label="Prompt", interactive=False)
    
    with gr.Row():
        viewer_normal_image_display = gr.Image(type="filepath", label="Normal Image", interactive=False)
        viewer_scope_image_display = gr.Image(type="filepath", label="Scope Image", interactive=False)

    # Arrow buttons to navigate between images
    with gr.Row():
        previous_button = gr.Button("⬅️ Previous")
        next_button = gr.Button("Next ➡️")

    # Function to update displayed content for viewer mode
    def update_viewer_display(idx):
        prompt, normal_image, scope_image, idx = display_and_select(idx)
        print(f"Updated display for index {idx}")
        return prompt, normal_image, scope_image, idx

    # Bind components for viewer mode
    previous_button.click(fn=previous_index, inputs=viewer_idx_input, outputs=viewer_idx_input).then(
        fn=update_viewer_display, inputs=viewer_idx_input, outputs=[viewer_prompt_display, viewer_normal_image_display, viewer_scope_image_display, viewer_idx_input]
    )
    next_button.click(fn=next_index, inputs=viewer_idx_input, outputs=viewer_idx_input).then(
        fn=update_viewer_display, inputs=viewer_idx_input, outputs=[viewer_prompt_display, viewer_normal_image_display, viewer_scope_image_display, viewer_idx_input]
    )

# Launch the app
gr_interface.launch(share=True, server_port=7861)
