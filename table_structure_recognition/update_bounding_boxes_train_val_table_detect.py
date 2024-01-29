import json
import os

def filter_annotations_and_categories(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Filter annotations for category_id = 0 (tables)
    filtered_annotations = [anno for anno in data['annotations'] if anno['category_id'] == 0]

    # Filter categories to keep only the 'table' category
    filtered_categories = [category for category in data['categories'] if category['id'] == 0]

    # Update the annotations and categories in the data object
    data['annotations'] = filtered_annotations
    data['categories'] = filtered_categories

    # Overwrite the original JSON file with modified data
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Paths to the train and val JSON files
train_json_path = 'data/images/table_detection_modeling_data/train/train.json'
val_json_path = 'data/images/table_detection_modeling_data/val/val.json'

# Apply the filtering to both files
filter_annotations_and_categories(train_json_path)
filter_annotations_and_categories(val_json_path)
