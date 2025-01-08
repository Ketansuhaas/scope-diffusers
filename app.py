import streamlit as st
import json
import os
from PIL import Image

# Load JSON data
clip_json_file = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/clip_scores.json"  # Path for CLIP scores JSON
tifa_json_file = "/projectnb/ivc-ml/xthomas/cs791/scope-diffusers/tifa/tifa_scores.json"  # Path for TIFA scores JSON

with open(clip_json_file, "r") as file:
    clip_data = json.load(file)
with open(tifa_json_file, "r") as file:
    tifa_data = json.load(file)

# Merge CLIP and TIFA data by image_id
def merge_data(clip_data, tifa_data):
    merged = []
    tifa_dict = {item["image_id"]: item for item in tifa_data}
    for clip_item in clip_data:
        image_id = clip_item["image_id"]
        tifa_item = tifa_dict.get(image_id, {})
        merged.append({**clip_item, **tifa_item})
    return merged

# Merge datasets
images_metadata = merge_data(clip_data, tifa_data)
num_images = len(images_metadata)

# Main App
st.title("Image Viewer with CLIP and TIFA Scores")

# Initialize session state for navigation
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Navigation Buttons
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("⬅️ Previous"):
        st.session_state.current_index = (st.session_state.current_index - 1) % num_images

with col3:
    if st.button("Next ➡️"):
        st.session_state.current_index = (st.session_state.current_index + 1) % num_images

# Fetch current image and metadata
current_image_data = images_metadata[st.session_state.current_index]

# Display Image
image_path = current_image_data.get("best_path", "")
if os.path.exists(image_path):
    image_path = os.path.join(image_path, "scope_image.png")
    st.image(Image.open(image_path), caption=f"Image {current_image_data['image_id']}")
else:
    st.warning("Image path does not exist.")

# Display TIFA Scores
st.subheader("TIFA Scores")
if "normal_tifa_score" in current_image_data and "best_scope_tifa_score" in current_image_data:
    st.write(f"**Normal TIFA Score**: {current_image_data['normal_tifa_score']:.2f}")
    st.write(f"**Best Scope TIFA Score**: {current_image_data['best_scope_tifa_score']:.2f}")
    st.write(f"**TIFA Difference**: {current_image_data['difference']:.4f}")
if "scope_tifa_scores" in current_image_data:
    st.subheader("Scope TIFA Scores by Step")
    st.json(current_image_data["scope_tifa_scores"])

# Display CLIP Scores
st.subheader("CLIP Scores")
if "normal_clip_score" in current_image_data and "best_scope_clip_score" in current_image_data:
    st.write(f"**Normal CLIP Score**: {current_image_data['normal_clip_score']:.4f}")
    st.write(f"**Best Scope CLIP Score**: {current_image_data['best_scope_clip_score']:.4f}")
    st.write(f"**CLIP Difference**: {current_image_data['difference']:.4f}")
if "scope_clip_scores" in current_image_data:
    st.subheader("Scope CLIP Scores by Step")
    st.json(current_image_data["scope_clip_scores"])

# Question Details Table
if "best_scope_tifa_dets" in current_image_data:
    st.subheader("Question Details")
    question_details = current_image_data["best_scope_tifa_dets"].get("question_details", {})

    data_table = []
    for question, details in question_details.items():
        data_table.append([
            question,
            details.get("answer", "N/A"),
            details.get("element_type", "N/A"),
            details.get("scores", 0),
        ])

    st.table(
        {"Question": [row[0] for row in data_table],
         "Answer": [row[1] for row in data_table],
         "Element Type": [row[2] for row in data_table],
         "Score": [row[3] for row in data_table]}
    )

# Page Indicator
st.write(f"**Page {st.session_state.current_index + 1} of {num_images}**")
