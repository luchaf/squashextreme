import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Initialize or load session state to store label-color mapping
if "color_to_label" not in st.session_state:
    st.session_state["color_to_label"] = {}

# Load the background image
bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

if bg_image:

    # Label selection
    label = st.selectbox("Select Label:", ["row", "column"])

    # Set the fill color based on the label selection
    if label == "column":
        fill_color = "rgba(255, 255, 0, 0.3)"  # Yellow with opacity
    else:
        fill_color = "rgba(255, 0, 0, 0.3)"  # Red with opacity

    # Drawing mode selection
    mode = "transform" if st.checkbox("Move ROIs", False) else "rect"

    bg_image = Image.open(bg_image)
    img_width, img_height = bg_image.size

    # Canvas for annotation
    canvas_result = st_canvas(
        fill_color=fill_color,
        stroke_width=0.5,
        background_image=bg_image,
        height=img_height,
        width=img_width,
        drawing_mode=mode,
        key="color_annotation_app",
    )

    # Process and display annotations
    if canvas_result.json_data is not None:
        df = pd.json_normalize(canvas_result.json_data["objects"])
        if len(df) == 0:
            st.warning("No annotations available.")
        else:
            # Map label color to label text
            st.session_state["color_to_label"][fill_color] = label
            df["label"] = df["fill"].map(st.session_state["color_to_label"])

            # Display the annotated table
            st.dataframe(df[["top", "left", "width", "height", "label"]])

            st.write(canvas_result.json_data["objects"])

