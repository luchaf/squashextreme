# Copyright (C) 2023, Lucas Hafner.
# Code adapted from doctr
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import tempfile

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from utils import extract_data_from_games
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
import io
import torch
from doctr.models import ocr_predictor, db_resnet50, crnn_vgg16_bn, parseq
from collections import defaultdict
from sklearn.cluster import DBSCAN
import time
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection
#import psutil
from sklearn.cluster import DBSCAN


import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from streamlit_dimensions import st_dimensions

from streamlit_img_label import st_img_label


import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager


def run_jo(img_dir='/teamspace/studios/this_studio/pointless/squashextreme/data/images', labels=["table row", "table column"]):

    idm = ImageDirManager(img_dir)

    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])
    
    def refresh():
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning('This is the last image.')

    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning('This is the first image.')

    def next_annotate_file():
        image_index = st.session_state["image_index"]
        next_image_index = idm.get_next_annotation_image(image_index)
        if next_image_index:
            st.session_state["image_index"] = idm.get_next_annotation_image(image_index)
        else:
            st.warning("All images are annotated.")
            next_image()

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])
    st.sidebar.write("Total files:", n_files)
    st.sidebar.write("Total annotate files:", n_annotate_files)
    st.sidebar.write("Remaining files:", n_files - n_annotate_files)

    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)
    st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)
    st.sidebar.button(label="Refresh", on_click=refresh)

    # Main content: annotate images
    img_file_name = idm.get_image(st.session_state["image_index"])
    img_path = os.path.join(img_dir, img_file_name)
    im = ImageManager(img_path)
    img = im.get_img()
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()
    rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

    def annotate():
        im.save_annotation()
        image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        next_annotate_file()

    if rects:
        st.button(label="Save", on_click=annotate)
        preview_imgs = im.init_annotation(rects)

        for i, prev_img in enumerate(preview_imgs):
            prev_img[0].thumbnail((200, 200))
            col1, col2 = st.columns(2)
            with col1:
                col1.image(prev_img[0])
            with col2:
                default_index = 0
                if prev_img[1]:
                    default_index = labels.index(prev_img[1])

                select_label = col2.selectbox(
                    "Label", labels, key=f"label_{i}", index=default_index
                )
                im.set_annotation(i, select_label)



def run_func(labels):

    # Upload file
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Read the file and set it in the session state
        bytes_data = uploaded_file.read()
        st.session_state["image"] = bytes_data
        image = Image.open(io.BytesIO(bytes_data))

        # Main content: annotate images
        resized_img = image.resize((600, 400))  # Resize for easier handling
        rects = st_img_label(resized_img, box_color="red")

        def annotate():
            if 'image' in st.session_state and rects:
                # Construct a simple string to represent the annotations
                annotation_data = f"Image: {uploaded_file.name}\n"
                for i, rect in enumerate(rects):
                    label = st.session_state.get(f"label_{i}", "No Label")
                    annotation_data += f"Box {i+1}: {rect}, Label: {label}\n"

                # Save to a file
                with open("annotations.txt", "a") as file:
                    file.write(annotation_data)

                st.success("Annotation saved!")


        if rects:
            st.button(label="Save", on_click=annotate)

            for i, rect in enumerate(rects):
                # Display the bounding box coordinates (or add any other logic you need)
                st.write(f"Bounding Box {i+1}: {rect}")

                # If you want to label the bounding boxes, you can add a select box like this
                label = st.selectbox("Label", labels, key=f"label_{i}")
                # Add logic to handle the selected label


# Set the page to wide mode
st.set_page_config(layout="wide")


# def editable_table(df):
#     # Create a copy of the DataFrame to store updates
#     new_df = df.copy()
    
#     # Iterate over each cell and create an input field for user interaction
#     for i, row in df.iterrows():
#         for col, value in row.items():
#             new_value = st.text_input(f"{col}_{i}", value)  # Create an input field for each cell
#             new_df.at[i, col] = new_value  # Update the new DataFrame with the input data
    
#     return new_df


class TableImageProcessor:
    """
    A class used to encapsulate all methods related to processing an image of a table and extracting its data.

    Attributes:
        image_path (str): The path to the image file to be processed.
        feature_extractor (DetrFeatureExtractor): The feature extractor for DETR (DEtection TRansformer).
        tatr_model (TableTransformerForObjectDetection): The table transformer model for object detection.
        reco_model (torch.nn.Module): The recognition model from the doctr package.
        model_ocr (OCRPredictor): The OCR predictor model for text detection and recognition.
        image (PIL.Image.Image): The processed PIL image.
        cell_locations (list): List of cell bounding box coordinates.
        value_boxes_absolute (list): List of word bounding boxes and their respective values.
    """

    def __init__(self, image_path, model_path='/teamspace/studios/this_studio/pointless/squashextreme/models/parseq_20240102-232334.pt'):
        """
        The constructor for TableImageProcessor class.

        Parameters:
            image_path (str): The path to the image file to be processed.
            model_path (str): The path to the pretrained model.
        """
        self.image_path = image_path
        self.feature_extractor, self.tatr_model, self.reco_model, self.model_ocr = self.initialize_models(model_path)
        self.image = self.process_image(image_path)
        self.cell_locations = []
        self.value_boxes_absolute = []
        self.value_boxes_relative = []

    def initialize_models(self, model_path):
        """
        Initializes and loads the models required for image processing and OCR.

        Parameters:
            model_path (str): The path to the pretrained model.

        Returns:
            tuple: Tuple containing:
                    - feature_extractor (DetrFeatureExtractor): The feature extractor for DETR.
                    - tatr_model (TableTransformerForObjectDetection): The table transformer model for object detection.
                    - reco_model (torch.nn.Module): The recognition model from the doctr package.
                    - model_ocr (OCRPredictor): The OCR predictor model for text detection and recognition.
        """
        # Initialize the feature extractor for DETR (DEtection TRansformer)
        feature_extractor_detr = DetrFeatureExtractor()

        # Initialize the table transformer model for object detection
        tatr_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

        # Initialize the recognition model from the doctr package
        reco_model = parseq(pretrained=False, pretrained_backbone=False)
        reco_params = torch.load(model_path, map_location="cpu")
        reco_model.load_state_dict(reco_params)

        # Initialize the OCR predictor model for text detection and recognition
        model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch=reco_model, pretrained=True)

        return feature_extractor_detr, tatr_model, reco_model, model_ocr

    def process_image(self, image_path):
        """
        Processes the image by opening and potentially rotating it for correct orientation.

        Parameters:
            image_path (str): The path to the image file to be processed.

        Returns:
            PIL.Image.Image: The processed image.
        """
        image = Image.open(image_path).convert("RGB")

        # # If the image is in landscape mode (width > height), rotate it
        # if image.width > image.height:
        #     image = image.transpose(Image.ROTATE_270)

        # # Save the processed image back to the path
        # image.save(image_path)

        return image

    def compute_boxes(self):
        """
        Computes the bounding boxes for table cells using the table transformer model.
        """
        # Extract the width and height of the image
        width, height = self.image.size

        # Prepare the image for the model
        encoding = self.feature_extractor(self.image, return_tensors="pt")

        # Compute the model output
        with torch.no_grad():
            outputs = self.tatr_model(**encoding)

        # Post-process the model output to get the bounding boxes and labels
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]
        boxes, labels = results['boxes'].tolist(), results['labels'].tolist()

        # Extract the cell locations based on the labels
        self.cell_locations = [(box_col[0], box_row[1], box_col[2], box_row[3]) for box_row, label_row in zip(boxes, labels) if label_row == 2 for box_col, label_col in zip(boxes, labels) if label_col == 1]
        self.cell_locations.sort(key=lambda x: (x[1], x[0]))

    def extract_and_map_words(self):
        """
        Extracts words from the image using the OCR model and maps them to their respective bounding boxes.
        """
        # Read the image using OpenCV to get its dimensions
        cv_image = cv2.imread(self.image_path)
        image_height, image_width, _ = cv_image.shape

        # Extract words from the image using the OCR model
        self.value_boxes_relative = self.get_word_list(self.model_ocr(DocumentFile.from_images(self.image_path)).export()["pages"][0])

        # Convert the relative coordinates of the bounding boxes to absolute pixel values
        self.value_boxes_absolute = [{'value': box['value'], 'geometry': [[int(box['geometry'][0][0] * image_width), 
                                                                          int(box['geometry'][0][1] * image_height)], 
                                                                         [int(box['geometry'][1][0] * image_width), 
                                                                          int(box['geometry'][1][1] * image_height)]]}
                                    for box in self.value_boxes_relative]

    def get_word_list(self, data):
        """
        Extracts a list of words and their positions from the OCR output.

        Parameters:
            data (dict): The OCR output data.

        Returns:
            list: A list of words and their positions.
        """
        words_list = []
        # Iterate through the blocks, lines, and words in the OCR output
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    words_list.append(word)
        return words_list

    def plot_boxes(self):
        """
        Plots the bounding boxes for table cells and words on the image.
        """

        # Assuming self.image is a PIL Image, get its dimensions for aspect ratio
        image_width, image_height = self.image.size

        # Set the figure size based on the aspect ratio of the image
        fig_width = 12  # You can adjust this as necessary
        fig_height = fig_width * (image_height / image_width)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(1, figsize=(fig_width, fig_height))
        ax.imshow(self.image)

        # Turn off the axis
        ax.axis('off')

        # # Create a figure and axis for the plot
        # fig, ax = plt.subplots(1, figsize=(12, 12))
        # ax.imshow(self.image)

        # Plot the bounding boxes for the table cells
        for (x1, y1, x2, y2) in self.cell_locations:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot the bounding boxes for the words
        for box_info in self.value_boxes_absolute:
            x1, y1 = box_info['geometry'][0]
            x2, y2 = box_info['geometry'][1]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, box_info['value'], color='white', ha='center', va='center', fontsize=32)

        # Remove padding and margins from the figure
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        return fig
        # Display the plot
        #st.pyplot(fig)

    def map_values_to_dataframe(self):
        """
        Maps the extracted words to their respective table cells and constructs a DataFrame.

        Returns:
            pandas.DataFrame: The DataFrame constructed from the table data.
        """
        unique_rows = set()
        unique_columns = set()

        # Identify the unique rows and columns based on the cell locations
        for box in self.cell_locations:
            x1, y1, x2, y2 = box
            unique_rows.add(y1)
            unique_columns.add(x1)

        # Initialize the DataFrame with the correct dimensions
        num_rows = len(unique_rows)
        num_cols = len(unique_columns)
        df = pd.DataFrame(index=range(num_rows), columns=range(num_cols))

        # Map the values to the correct cells in the DataFrame
        for value_box in self.value_boxes_absolute:
            value = value_box['value']
            x_mid = (value_box['geometry'][0][0] + value_box['geometry'][1][0]) / 2
            y_mid = (value_box['geometry'][0][1] + value_box['geometry'][1][1]) / 2

            # Find the corresponding cell for each word and update the DataFrame
            for i, box in enumerate(self.cell_locations):
                if self.is_inside_box((x_mid, y_mid), box):
                    row_idx = sorted(list(unique_rows)).index(box[1])
                    col_idx = sorted(list(unique_columns)).index(box[0])
                    if pd.isnull(df.at[row_idx, col_idx]):
                        df.at[row_idx, col_idx] = value
                    else:
                        df.at[row_idx, col_idx] += value
                    break
        return df

    def is_inside_box(self, point, box):
        """
        Checks if a given point is inside a specified box.

        Parameters:
            point (tuple): The (x, y) coordinates of the point.
            box (tuple): The (x1, y1, x2, y2) coordinates of the box.

        Returns:
            bool: True if the point is inside the box, False otherwise.
        """
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    ##############################################################################
    ### rule based table extraction with a sprinkle of machine learning ###
    ##############################################################################

    def calculate_centroids(self, data):
        """
        Calculate the centroids of the geometry in the given data.

        Args:
            data (list): A list of dictionaries, where each dictionary contains the 'geometry' key.

        Returns:
            numpy.ndarray: An array of calculated centroids.
        """
        return np.array([
            [
                (item['geometry'][0][0] + item['geometry'][1][0]) / 2,  # x-coordinate
                (item['geometry'][0][1] + item['geometry'][1][1]) / 2   # y-coordinate
            ] for item in data
        ])

    def cluster_data(self, centroids, axis_index, eps):
        """
        Cluster the data using DBSCAN based on the specified axis and epsilon.

        Args:
            centroids (numpy.ndarray): The centroids of the data points.
            axis_index (int): The axis index (0 for x, 1 for y) to be considered for clustering.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.

        Returns:
            tuple: A tuple containing the labels for each point and the centroids of the clusters.
        """
        dbscan = DBSCAN(eps=eps, min_samples=4)
        dbscan.fit(centroids[:, axis_index].reshape(-1, 1))
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        centroid_points = np.array([np.mean(centroids[labels == label, axis_index]) for label in unique_labels if label != -1])
        return labels, centroid_points

    def get_min_max(self, data, index):
        """
        Get the minimum and maximum values for the specified axis of the geometry.

        Args:
            data (list): A list of dictionaries, where each dictionary contains the 'geometry' key.
            index (int): The axis index (0 for x, 1 for y) to consider.

        Returns:
            tuple: A tuple containing the minimum and maximum values.
        """
        return min([item['geometry'][0][index] for item in data]), max([item['geometry'][1][index] for item in data])

    def get_grid_position(self, x, y, min_x, min_y, cell_width, cell_height):
        """
        Calculate the row and column position of a point in the grid.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            min_x (float): The minimum x-coordinate of the grid.
            min_y (float): The minimum y-coordinate of the grid.
            cell_width (float): The width of a cell in the grid.
            cell_height (float): The height of a cell in the grid.

        Returns:
            tuple: A tuple containing the row and column indices.
        """
        col = int((x - min_x) / cell_width)
        row = int((y - min_y) / cell_height)
        return row, col

    def create_table(self, data, eps_x, eps_y):
        """
        Create a table structure from the given data using clustering for row and column determination.

        Args:
            data (list): A list of dictionaries, where each dictionary represents a word with its geometry and value.
            eps_x (float): Epsilon value for DBSCAN clustering along the x-axis.
            eps_y (float): Epsilon value for DBSCAN clustering along the y-axis.

        Returns:
            pandas.DataFrame: A DataFrame representing the structured table.
        """
        # Calculate centroids of the geometries
        centroids = self.calculate_centroids(data)

        # Cluster data along x and y axes
        labels_x, centroids_x = self.cluster_data(centroids, 0, eps_x)
        labels_y, centroids_y = self.cluster_data(centroids, 1, eps_y)

        # Determine the number of rows and columns
        n_rows, n_cols = len(np.unique(labels_y)), len(np.unique(labels_x))

        # Get minimum and maximum values for x and y coordinates
        min_x, max_x = self.get_min_max(data, 0)
        min_y, max_y = self.get_min_max(data, 1)

        # Calculate the width and height of each cell in the grid
        cell_width, cell_height = (max_x - min_x) / n_cols, (max_y - min_y) / n_rows

        cell_words = defaultdict(str)
        for item in data:
            # Calculate the average x and y coordinates for the item
            avg_x, avg_y = self.calculate_centroids([item])[0]
            # Determine the grid position for the item
            row, col = self.get_grid_position(avg_x, avg_y, min_x, min_y, cell_width, cell_height)
            # Append the item's value to the cell
            cell_words[(row, col)] = cell_words[(row, col)] + " " + item['value'] if cell_words[(row, col)] else item['value']

        # Initialize the table with empty values
        concatenated_table = [['' for _ in range(n_cols)] for _ in range(n_rows)]
        # Fill in the table with concatenated words
        for (row, col), value in cell_words.items():
            concatenated_table[row][col] = value.strip()

        # Convert the table to a DataFrame for better visualization and return
        return pd.DataFrame(concatenated_table)
    
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
    # Load data from database
    df = pd.DataFrame()
    df_sheet = pd.read_csv(st.secrets["public_gsheets_url"])
    df_sheet["date"] = df_sheet["date"].astype(str)
    list_of_available_dates = list(set(df_sheet["date"].tolist()))
    df_sheet['parsed_sheet_df'] = df_sheet.apply(lambda x: extract_data_from_games(x["games"], x["date"]), axis=1)
    df_tmp = pd.DataFrame()
    for _, row in df_sheet.iterrows():
        df_tmp = pd.concat([df_tmp, row["parsed_sheet_df"]])
    df_tmp = df_tmp[["First Name", "First Score", "Second Name", "Second Score", "date"]].reset_index(drop=True).copy()
    df_tmp = df_tmp.rename(columns={
        "First Name": "Player1", 
        "Second Name": "Player2", 
        "First Score": "Score1", 
        "Second Score": "Score2", 
    }).copy()
    st.dataframe(df_tmp)
    
    return df_tmp

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
                    spreadsheetId = st.secrets["public_gsheets_id"]
                    scope = ['https://www.googleapis.com/auth/spreadsheets']
                    google_creds_dict = dict(st.secrets["google_creds"])
                    # Create credentials from the dictionary
                    credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scope)
                    client = gspread.authorize(credentials)
                    spreadsheet = client.open_by_key(spreadsheetId)
                    worksheet = spreadsheet.sheet1
                    match_result = f"{st.session_state['player1_name']} - {st.session_state['player2_name']} {st.session_state['player1_score']}:{st.session_state['player2_score']}"
                    game_date = int(st.session_state['matchday_input'].strftime('%Y%m%d'))
                    # Append the new game and its date to the worksheet
                    worksheet.append_row([match_result, game_date])
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
                #st.session_state['step'] = 'upload'
                st.experimental_rerun()  # Force a rerun to update the page immediately

# Create tab names and corresponding functions
tab_names = [
    "Pointless list of recorded matches",
    "Pointless online form",
    "Pointless upload page",
    "Pointless tabel labeling",
]

tab_functions = [
    show_me_the_list,
    online_form,
    upload_page_fixed,
    run_jo
]

# Create tabs dynamically
selected_tab = st.selectbox("Select an option to enter your match result", tab_names)
tab_index = tab_names.index(selected_tab)
selected_function = tab_functions[tab_index]

# Execute the selected function
selected_function()



# # pip install pascal_voc_writer




