# Copyright (C) 2023, Lucas Hafner.
# Code adapted from doctr
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import tempfile
from collections import defaultdict
import pandas as pd
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import time
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from match_results.match_results_utils import SquashMatchDatabase, TableImageProcessor
from datetime import datetime, date
from doctr.io import DocumentFile
from doctr.models.predictor import OCRPredictor
from doctr.models import ocr_predictor, db_resnet50, parseq

# Set the page to wide mode
st.set_page_config(layout="wide")

# Constants file
EXTENSION_TO_FORMAT = {
    'jpg': 'JPEG',
    'jpeg': 'JPEG',
    'png': 'PNG',
    'pdf': 'PDF'
}

# Utility Functions
def get_file_extension(filename):
    return os.path.splitext(filename)[1].lstrip('.').lower()

def correct_image_orientation(image):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = image._getexif()
    if exif:
        exif = dict(exif.items())
        orientation = exif.get(orientation)
        if orientation == 3:
            return image.rotate(180, expand=True)
        elif orientation == 6:
            return image.rotate(270, expand=True)
        elif orientation == 8:
            return image.rotate(90, expand=True)
    return image


def upload_page_fixed():

    # Initialize session state for tracking progress
    if 'step' not in st.session_state:
        st.session_state['step'] = 'upload'
    if 'editable_df' not in st.session_state:
        st.session_state['editable_df'] = pd.DataFrame()

    if 'color_to_label' not in st.session_state:
        st.session_state['color_to_label'] = {}

    if 'selected_label' not in st.session_state:
        st.session_state['selected_label'] = "row"

    if "current_script_path" not in st.session_state:
        st.session_state["current_script_path"] = os.path.dirname(__file__)
        current_script_path = st.session_state["current_script_path"]

    if "parent_directory" not in st.session_state:
        st.session_state["parent_directory"] = os.path.join(current_script_path, os.pardir)
        parent_directory = st.session_state["parent_directory"]

    if "target_folder" not in st.session_state:
        st.session_state["target_folder"] = os.path.normpath(os.path.join(parent_directory, 'table_structure_recognition/data/images'))
        target_folder = st.session_state["target_folder"]
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)


    # First step: Uploading the image
    if st.session_state['step'] == 'upload':
        with st.form("upload_form"):
            uploaded_file = st.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
            upload_button = st.form_submit_button('Upload Image')

        if upload_button and uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['step'] = 'crop'
            st.experimental_rerun()  # Force a rerun to update the page immediately

    # Second step: Cropping the image
    elif st.session_state['step'] == 'crop' and 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']
        file_name = uploaded_file.name

        file_extension = get_file_extension(uploaded_file.name)
        st.session_state['image_format'] = EXTENSION_TO_FORMAT.get(file_extension, file_extension.upper())
        orig_img = Image.open(uploaded_file)
        orig_img = correct_image_orientation(orig_img)

        # Define the path for the folder one level above the current script
        current_script_path = st.session_state["current_script_path"]
        parent_directory = st.session_state["parent_directory"]
        target_folder = st.session_state["target_folder"]
        save_path = os.path.join(target_folder, uploaded_file.name)

        orig_img.save(save_path)
        st.session_state['orig_image'] = orig_img

        width, height = orig_img.size
        
        label = st.radio(
            "Select Label:",
            ["table", "row", "column"],
            key='label',
        )

        cropped_img_dims = {"top":0,"left":0,"height":height,"width":width}
        # Update the session state based on the label selection
        st.session_state['selected_label'] = label

        # Set the fill color based on the updated session state
        if st.session_state['selected_label'] == "column":
            fill_color = "rgba(255, 255, 0, 0.3)"  # Yellow for columns
        elif st.session_state['selected_label'] == "row":
            fill_color = "rgba(255, 0, 0, 0.3)"  # Red for rows
        elif st.session_state['selected_label'] == "table":
            fill_color = "rgba(255, 192, 203, 0.6)"  # Green for table

        # Drawing mode selection
        mode = "transform" if st.checkbox("Move ROIs", False) else "rect"
        # Update the session state based on the label selection
        st.session_state['mode'] = mode
        # Canvas for annotation
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=0.5,
            background_image=orig_img,
            height=height/4,
            width=width/4,
            drawing_mode=st.session_state['mode'],
            key="color_annotation_app",
        )
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            if len(df) == 0:
                st.warning("No annotations available.")
            else:
                # Update color to label mapping here
                st.session_state['color_to_label'][fill_color] = st.session_state['selected_label']
                df["label"] = df["fill"].map(st.session_state["color_to_label"])

                # Display the annotated table
                st.dataframe(df[["top", "left", "width", "height", "label"]])
                
                numerical_columns = ["top", "left", "width", "height"]
                df[numerical_columns] = df[numerical_columns].multiply(4)

                df_table = df[df["label"]=="table"].copy()
                df_table = df_table[["top", "left", "height", "width"]].copy()
                st.dataframe(df_table)
                cropped_img_dims = df_table.to_dict(orient='records')[0]

        with st.form("crop_form"):
            crop_button = st.form_submit_button('Crop Image')

        if crop_button:
            df.to_parquet(os.path.join(target_folder, "okok.parquet"))
            crop_area = (cropped_img_dims["left"], cropped_img_dims["top"], cropped_img_dims["left"] + cropped_img_dims["width"], cropped_img_dims["top"] + cropped_img_dims["height"])
            cropped_img = orig_img.crop(crop_area)

            st.image(cropped_img, caption='Cropped Image', use_column_width=True)
            st.session_state['cropped_img'] = cropped_img
            st.session_state['step'] = 'process'
            st.experimental_rerun()  # Force a rerun to update the page immediately

    # Third step: Process and display the cropped image and table
    elif st.session_state['step'] == 'process':
        if 'uploaded_file' in st.session_state and 'cropped_img' in st.session_state and 'image_format' in st.session_state:
            uploaded_file = st.session_state['uploaded_file']
            cropped_img = st.session_state['cropped_img']
            image_format = st.session_state['image_format']

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                cropped_img.save(tmp_file, format=image_format)
                tmp_file_path = tmp_file.name

            doc = DocumentFile.from_images(tmp_file_path)
            page = doc[0]

            # Create an instance of the TableImageProcessor class with the image located at tmp_file_path. 
            # During initialization, the image is processed, and the necessary models
            #  (DETR feature extractor, table transformer, recognition model, OCR predictor) are loaded.
            processor = TableImageProcessor(tmp_file_path)

            # Get the bounding boxes of the table cells using the table transformer model. 
            # This information is crucial to understand the layout and structure of the table in the image.
            processor.compute_boxes()
            
            # Apply the OCR model to extract words from the image. 
            # Then, map these words to their respective bounding boxes, preparing the data for insertion into a DataFrame.
            processor.extract_and_map_words()

            # This step takes the processed information (words and their locations) and maps them into a structured format, 
            # creating a pandas DataFrame. Each cell in the DataFrame corresponds to a cell in the table image, 
            # filled with the extracted text.
            df_from_table_transformer  = processor.map_values_to_dataframe()
            
            # Wrap table editing and submission button in a form
            with st.form("edit_table_form"):
                col1, col2, col3 = st.columns(3)

                if cropped_img:
                    img_width, img_height = cropped_img.size
                    aspect_ratio = img_height / img_width
                    canvas_width = 600
                    canvas_height = int(canvas_width * aspect_ratio)
                else:
                    canvas_width, canvas_height = 600, 600

                with col1:
                    st.write("Table transformer + Text recognition result:")
                    match_day_date = st.date_input('Date', value=date.today())
                    match_day_date = match_day_date.strftime('%Y%m%d')
                    edited_df = st.data_editor(
                        df_from_table_transformer , 
                        num_rows="dynamic", 
                        height=750, 
                        use_container_width=True)
                    # Submission button for the form
                    submit_edits_button = st.form_submit_button('Confirm Edits and Save Table')

                with col2:
                    st.write("Cropped Image")
                    st.image(cropped_img, use_column_width=True)

                with col3:
                    st.write("Table transformer + Text recognition visualized:")
                    boxes_fig = processor.plot_boxes()
                    st.pyplot(boxes_fig, use_container_width=True)

            if submit_edits_button:
                processor.save_corrected_data_for_retraining(edited_df)
                
                # Set the first row as the header
                edited_df.columns = edited_df.iloc[0]
                # Drop the first row
                edited_df = edited_df.drop(edited_df.index[0])           

                edited_df["date"] = match_day_date
                edited_df["match_number_day"] = range(1, len(edited_df) + 1)

                st.dataframe(edited_df)

                # db = SquashMatchDatabase()
                # db.insert_df_into_db(edited_df) # Insert a Pandas DataFrame
                # db.update_csv_file() # Update CSV file with current DB data

                st.write("Table Saved Successfully!")
                st.session_state['step'] = 'upload'
                st.experimental_rerun()  # Force a rerun to update the page immediately


def enter_table_manually():
    with st.form("add match results to table"):
        df_manual_add = pd.DataFrame(
        {
        'Player1': ["Friede"],
        'Score1': [1],
        'Player2': ["Peter"],
        'Score2': [1],
        })
        match_day_date = st.date_input('Date', value=date.today())
        match_day_date = match_day_date.strftime('%Y%m%d')
        edited_df = st.data_editor(
                                df_manual_add, 
                                column_config={
                                     "Player1": st.column_config.SelectboxColumn(
                                        "Player1", 
                                        options=["Friede", "Lucas", "Peter", "Simon", "Tobias"],
                                        default="Friede",
                                        required=True,
                                        ),
                                    "Score1": st.column_config.NumberColumn(
                                                "Score2",
                                                help="Achieved score of Player1",
                                                min_value=0,
                                                max_value=100,
                                                step=1,
                                                format="%d",
                                            ),
                                     "Player2": st.column_config.SelectboxColumn(
                                        "Player2", 
                                        options=["Friede", "Lucas", "Peter", "Simon", "Tobias"],
                                        default="Peter",
                                        required=True,
                                        ),
                                    "Score2": st.column_config.NumberColumn(
                                                "Score2",
                                                help="Achieved score of Player2",
                                                min_value=0,
                                                max_value=100,
                                                step=1,
                                                format="%d",
                                            ),
                                },
                                num_rows="dynamic", 
                                use_container_width=True,
                                column_order=("Player1", "Score1", "Player2", "Score2")
                                )
        # Submission button for the form
        submit_edits_button = st.form_submit_button('Confirm Edits and Save Table')

    if submit_edits_button:
        edited_df["date"] = match_day_date
        edited_df["match_number_day"] = range(1, len(edited_df) + 1)
        db = SquashMatchDatabase()
        db.insert_df_into_db(edited_df) # Insert a Pandas DataFrame
        db.update_csv_file() # Update CSV file with current DB data                            
        st.write("Table Saved Successfully!")


# Create tab names and corresponding functions
tab_names = [
    "Digital - Enter match results directly in the online form!",
    "Analog - Upload photo of your match results and let the AI extract the match results.",
]

tab_functions = [
    enter_table_manually,    
    upload_page_fixed,
]

# Create tabs dynamically
selected_tab = st.radio("How do you want to track your match results?", tab_names)
tab_index = tab_names.index(selected_tab)
selected_function = tab_functions[tab_index]

# Execute the selected function
selected_function()
