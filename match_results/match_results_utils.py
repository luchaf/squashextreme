import sqlite3
import pandas as pd
import csv
import os

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
from doctr.models import ocr_predictor, from_hub
from doctr.models.predictor import OCRPredictor
from doctr.models import ocr_predictor, db_resnet50, parseq, crnn, crnn_mobilenet_v3_small
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection

import io
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from collections import defaultdict
from PIL import Image, ExifTags
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
from datetime import datetime




class SquashMatchDatabase:
    """
    A class to handle database operations for storing and retrieving squash match results.

    Attributes:
        db_path (str): Path to the SQLite database file.
        csv_path (str): Path to the CSV file for exporting data.
    """

    def __init__(self, db_path=None, csv_path=None):
        """
        The constructor for SquashMatchDatabase class.

        Parameters:
            db_path (str): Path to the SQLite database file.
            csv_path (str): Path to the CSV file for exporting data.
        """
        #    if db_path is None:
        #        db_path = os.path.join('/teamspace/studios/this_studio/squashextreme/match_results', 'squash.db')
        #    if csv_path is None:
        #        csv_path = os.path.join('/teamspace/studios/this_studio/squashextreme/match_results', 'squash.csv')
        #    self.db_path = db_path
        #    self.csv_path = csv_path
        #    self.create_squash_db()

        if db_path is None:
            db_path = os.path.join('/mount/src/squashextreme/match_results', 'squash.db')
        if csv_path is None:
            csv_path = os.path.join('/mount/src/squashextreme/match_results', 'squash.csv')
        self.db_path = db_path
        self.csv_path = csv_path
        self.create_squash_db()



    def create_squash_db(self):
        """
        Creates the squash database and match_results table if it doesn't already exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS match_results (
            match_number_total INTEGER PRIMARY KEY AUTOINCREMENT,
            match_number_day INTEGER,
            Player1 TEXT,
            Score1 INTEGER,
            Player2 TEXT,
            Score2 INTEGER,
            date TEXT
        );
        '''
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()

    def get_match_results_from_db(self):
        """
        Retrieves all match results from the database and returns them as a Pandas DataFrame.

        Returns:
            DataFrame: A Pandas DataFrame containing all rows from the match_results table.
        """
        conn = sqlite3.connect(self.db_path)
        sql_query = 'SELECT * FROM match_results'
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df

    def insert_df_into_db(self, df_input):
        """
        Inserts data from a Pandas DataFrame into the match_results table in the database.

        Parameters:
            df_input (DataFrame): A Pandas DataFrame containing the data to be inserted.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for _, row in df_input.iterrows():
            cursor.execute("""
                INSERT INTO match_results (match_number_day, Player1, Score1, Player2, Score2, date)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (row['match_number_day'], row['Player1'], row['Score1'], row['Player2'], row['Score2'], row['date']))
        conn.commit()
        conn.close()

    def update_csv_file(self):
        """
        Exports data from the match_results table in the database to a CSV file.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM match_results")
        rows = cursor.fetchall()
        with open(self.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i[0] for i in cursor.description])
            writer.writerows(rows)
        cursor.close()
        conn.close()

# Usage
#db = SquashMatchDatabase()
# db.insert_df_into_db(your_dataframe) # Insert a Pandas DataFrame
# df = db.get_match_results_from_db() # Get match results as a DataFrame
# db.update_csv_file() # Update CSV file with current DB data


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

    def __init__(self, image_path, text_recognition_model_path='luchaf/crnn_mobilenet_v3_large_pointless_1'):
        """
        The constructor for TableImageProcessor class.

        Parameters:
            image_path (str): The path to the image file to be processed.
            text_recognition_model_path (str): The path to the pretrained text recognition model.
        """
        self.image_path = image_path
        self.feature_extractor, self.tatr_model, self.reco_model, self.model_ocr = self.initialize_models(text_recognition_model_path)
        self.image = self.process_image(image_path)
        self.cell_locations = []
        self.value_boxes_absolute = []
        self.value_boxes_relative = []

    def initialize_models(self, text_recognition_model_path):
        """
        Initializes and loads the models required for image processing and OCR.

        Parameters:
            text_recognition_model_path (str): The path to the pretrained text recognition model.

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
        #tatr_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
        tatr_model = TableTransformerForObjectDetection.from_pretrained("luchaf/table-transformer-structure-recognition-pointless")
        #tatr_model = TableTransformerForObjectDetection.from_pretrained("squashextreme/table_structure_recognition/tatr_model/best_model")

        # Initialize the recognition model
        reco_model = from_hub(text_recognition_model_path)

        # Initialize the OCR predictor model for text detection and recognition
        model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch=reco_model, pretrained=True)
        #model_ocr = ocr_predictor(det_arch='db_resnet50', reco_arch="parseq", pretrained=True)

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
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.9, target_sizes=[(height, width)])[0]
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
        self.df_from_table_transformer = df
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

    def save_corrected_data_for_retraining(self, edited_df, retraining_folder='squashextreme/text_recognition/data/user_entries/annotations'):
        if not os.path.exists(retraining_folder):
            os.makedirs(retraining_folder)
        #images_folder = os.path.join(retraining_folder, 'images')
        #if not os.path.exists(images_folder):
        #    os.makedirs(images_folder)

        labels = {}
        for row_idx, (original_row, edited_row) in enumerate(zip(self.df_from_table_transformer.itertuples(index=False), edited_df.itertuples(index=False))):
            for col_idx, (original_value, edited_value) in enumerate(zip(original_row, edited_row)):
                #if original_value != edited_value:
                # Get the bounding box for this cell
                box = self.cell_locations[row_idx * self.df_from_table_transformer.shape[1] + col_idx]
                x1, y1, x2, y2 = box

                # Crop and save image
                cropped_image = self.image.crop((x1, y1, x2, y2))
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
                if original_value != edited_value:
                    filename = f"img_{edited_value}_corrected_{timestamp}.png"
                else:
                    filename = f"img_{edited_value}_accepted_{timestamp}.png"
                filepath = os.path.join(retraining_folder, filename)
                cropped_image.save(filepath)

                # Use edited value as label
                labels[filename] = edited_value

        # Save labels to JSON file
        with open(os.path.join(retraining_folder, 'labels.json'), 'w') as label_file:
            json.dump(labels, label_file, indent=4)

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
        n_rows = 16
        n_cols = 4

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
    