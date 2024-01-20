# Copyright (C) 2023, Lucas Hafner.
# Code adapted from doctr
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# import streamlit as st
# import sys
# sys.path.append('/teamspace/studios/this_studio/squashextreme')

# st.write(sys.path)

# import os
# st.write(os.getcwd())


import io
import os
import tempfile
from collections import defaultdict
import pandas as pd
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn, parseq
import torch
from sklearn.cluster import DBSCAN
import time
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from match_results.match_results_utils import SquashMatchDatabase, TableImageProcessor


# Set the page to wide mode
st.set_page_config(layout="wide")



def load_predictor(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    bin_thresh: float,
    device: torch.device,
) -> OCRPredictor:
    """Load a predictor from doctr.models

    Args:
    ----
        det_arch: detection architecture
        reco_arch: recognition architecture
        assume_straight_pages: whether to assume straight pages or not
        straighten_pages: whether to straighten rotated pages or not
        bin_thresh: binarization threshold for the segmentation map
        device: torch.device, the device to load the predictor on

    Returns:
    -------
        instance of OCRPredictor
    """
    predictor = ocr_predictor(
        det_arch,
        reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=straighten_pages,
        detect_orientation=not assume_straight_pages,
    ).to(device)
    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    return predictor


def forward_image(predictor: OCRPredictor, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Forward an image through the predictor

    Args:
    ----
        predictor: instance of OCRPredictor
        image: image to process
        device: torch.device, the device to process the image on

    Returns:
    -------
        segmentation map
    """
    with torch.no_grad():
        processed_batches = predictor.det_predictor.pre_processor([image])
        out = predictor.det_predictor.model(processed_batches[0].to(device), return_model_output=True)
        seg_map = out["out_map"].to("cpu").numpy()

    return seg_map


# Define functions for each tab
def show_me_the_list():
    db = SquashMatchDatabase()
    df = db.get_match_results_from_db() # Get match results as a DataFrame
    st.dataframe(df)
    return df

def online_form():
    # Define a list of player names
    player_names = ["Friedemann", "Lucas", "Peter", "Simon", "Tobias"]

    def reset_session_state():
        """Helper function to reset session state."""
        st.session_state['player1_name'] = ''
        st.session_state['player1_score'] = None
        st.session_state['player2_name'] = ''
        st.session_state['player2_score'] = None
        st.session_state['matchday_input'] = None
        st.session_state['show_confirm'] = False
        st.session_state['data_written'] = False
        
    def display_enter_match_results():
        # Initialize session state values if not already set
        if 'player1_name' not in st.session_state:
            st.session_state['player1_name'] = ''
        if 'player1_score' not in st.session_state:
            st.session_state['player1_score'] = None
        if 'player2_name' not in st.session_state:
            st.session_state['player2_name'] = ''
        if 'player2_score' not in st.session_state:
            st.session_state['player2_score'] = None
        if 'matchday_input' not in st.session_state:
            st.session_state['matchday_input'] = None
        if 'show_confirm' not in st.session_state:
            st.session_state['show_confirm'] = False
        if 'data_written' not in st.session_state:
            st.session_state['data_written'] = False

        if st.session_state['data_written']:
            st.success("Successfully wrote match result to database. Do you want to enter a new match result?")
            if st.button("Enter New Match Result"):
                reset_session_state()
                st.experimental_rerun()
        else:
            st.title("Racquet Records: Document your match results")
        
            st.write("Log your praiseworthy or pitiful match results here:")
            
            # Use selectbox for player names with an option to add a new player
            selected_player1 = st.selectbox("Player 1 Name", [''] + player_names + ['Add New Player'])
        
            if selected_player1 == 'Add New Player':
                new_player_name = st.text_input("Enter New Player Name")
                if new_player_name.strip() != '':
                    player_names.append(new_player_name.strip())
                    selected_player1 = new_player_name.strip()

            if selected_player1 != '':
                # Use number input for player 1 score with a default value of 0
                st.session_state['player1_name'] = selected_player1
                st.session_state['player1_score'] = st.number_input("Player 1 Score", min_value=0, value=st.session_state.get('player1_score', 0), step=1)
                
                if st.session_state['player1_score'] is not None:
                    # Use selectbox for player 2 name with an option to add a new player
                    selected_player2 = st.selectbox("Player 2 Name", [''] + player_names + ['Add New Player'])
        
                    if selected_player2 == 'Add New Player':
                        new_player_name = st.text_input("Enter New Player Name")
                        if new_player_name.strip() != '':
                            player_names.append(new_player_name.strip())
                            selected_player2 = new_player_name.strip()

                    if selected_player2 != '':
                        # Use number input for player 2 score with a default value of 0
                        st.session_state['player2_name'] = selected_player2
                        st.session_state['player2_score'] = st.number_input("Player 2 Score", min_value=0, value=st.session_state.get('player2_score', 0), step=1)
                        if st.session_state['player2_score'] is not None:
                            st.session_state['matchday_input'] = st.date_input("Matchday", st.session_state['matchday_input'] if st.session_state['matchday_input'] else None)
        
            if st.session_state['matchday_input'] and (st.session_state['player1_name'] or st.session_state['player2_name']):
                if st.button("Preview"):
                    st.subheader("Confirm the following match result:")
                    st.write(f"{st.session_state['player1_name']}: {st.session_state['player1_score']} - {st.session_state['player2_name']}: {st.session_state['player2_score']} on {st.session_state['matchday_input']}")
                    st.session_state['show_confirm'] = True
        
            if st.session_state['show_confirm']:
                if st.button("Confirm"):
                    # Determine match_number_day by loading the existing database....
                    db = SquashMatchDatabase()
                    df_get_date_match_num = db.get_match_results_from_db()
                    date_to_filter_on = str(st.session_state['matchday_input'].strftime('%Y%m%d'))                   
                    match_number_day_to_use = df_get_date_match_num[df_get_date_match_num["date"]==date_to_filter_on]["match_number_day"].max()+1
                    if np.isnan(match_number_day_to_use):
                        match_number_day_to_use = 1

                    df_add = pd.DataFrame({
                        'Player1': [st.session_state['player1_name']],
                        'Score1': [st.session_state['player1_score']],
                        'Player2': [st.session_state['player2_name']],
                        'Score2': [st.session_state['player2_score']],
                        'date': [int(st.session_state['matchday_input'].strftime('%Y%m%d'))],
                        'match_number_day':[match_number_day_to_use],
                    })
                    db.insert_df_into_db(df_add) # Insert new data to databse
                    db.update_csv_file() # update the associated .csv file

                    # Clear the inputs and flag data as written
                    st.session_state['data_written'] = True               
                    # Clear the inputs
                    st.session_state['player1_name'] = ''
                    st.session_state['player1_score'] = None
                    st.session_state['player2_name'] = ''
                    st.session_state['player2_score'] = None
                    st.session_state['matchday_input'] = None
                    st.session_state['show_confirm'] = False
        
                    st.experimental_rerun()  # This will rerun the script and update the UI with cleared inputs

    display_enter_match_results()


def upload_page_fixed():

    # Initialize session state for tracking progress
    if 'step' not in st.session_state:
        st.session_state['step'] = 'upload'
    if 'editable_df' not in st.session_state:
        st.session_state['editable_df'] = pd.DataFrame()

    DET_ARCHS = ["db_resnet50"]
    RECO_ARCHS = ["parseq"]
    forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        # Map file extensions to PIL format strings
        EXTENSION_TO_FORMAT = {
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'png': 'PNG',
            'pdf': 'PDF'
        }

        file_extension = os.path.splitext(uploaded_file.name)[1].lstrip('.').lower()
        st.session_state['image_format'] = EXTENSION_TO_FORMAT.get(file_extension, file_extension.upper())

        orig_img = Image.open(uploaded_file)
        # Check and correct the orientation if needed
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = orig_img._getexif()
        if exif:
            exif = dict(exif.items())
            orientation = exif.get(orientation)
            if orientation == 3:
                orig_img = orig_img.rotate(180, expand=True)
            elif orientation == 6:
                orig_img = orig_img.rotate(270, expand=True)
            elif orientation == 8:
                orig_img = orig_img.rotate(90, expand=True)

        # Define the path for the folder one level above the current script
        current_script_path = os.path.dirname(__file__)
        parent_directory = os.path.join(current_script_path, os.pardir)
        target_folder = os.path.normpath(os.path.join(parent_directory, 'data/images'))
        # Make sure the target directory exists, if not, create it
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        # Define the full path for the new file
        save_path = os.path.join(target_folder, uploaded_file.name)
        orig_img.save(save_path)

        width, height = orig_img.size
        border_pixels = 25
        cropped_img_dims = st_cropper(orig_img, realtime_update=True, box_color='#0000FF', return_type="box", aspect_ratio=None, default_coords=(border_pixels, width - border_pixels, border_pixels, height - border_pixels))

        with st.form("crop_form"):
            crop_button = st.form_submit_button('Crop Image')

        if crop_button:
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

            if uploaded_file.name.endswith(".pdf"):
                doc = DocumentFile.from_pdf(tmp_file_path)
            else:
                doc = DocumentFile.from_images(tmp_file_path)

            page = doc[0]
            processor = TableImageProcessor(tmp_file_path)
            processor.compute_boxes()
            processor.extract_and_map_words()
            df = processor.map_values_to_dataframe()

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
                    edited_df = st.data_editor(
                        df, 
                        num_rows="dynamic", 
                        height=1400, 
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
                st.write("Table Saved Successfully!")
                st.session_state['step'] = 'upload'
                st.experimental_rerun()  # Force a rerun to update the page immediately

# Create tab names and corresponding functions
tab_names = [
    "Pointless list of recorded matches",
    "Pointless online form",
    "Pointless upload page",
]

tab_functions = [
    show_me_the_list,
    online_form,
    upload_page_fixed,
]

# Create tabs dynamically
selected_tab = st.selectbox("Select an option to enter your match result", tab_names)
tab_index = tab_names.index(selected_tab)
selected_function = tab_functions[tab_index]

# Execute the selected function
selected_function()



# # pip install pascal_voc_writer




