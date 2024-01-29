import os
import json
import random
import shutil

# Directory containing JSON and JPG files
directory = 'data/images'
structure_recognition_directory = 'data/images/table_structure_recognition_modeling_data'

# Make train and val directories for images
train_img_dir = os.path.join(structure_recognition_directory, 'train')
val_img_dir = os.path.join(structure_recognition_directory, 'val')
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

# Retrieve all JSON file names
all_json_files = [f for f in os.listdir(directory) if f.endswith('_output.json')]

# Shuffle and split the files into training (80%) and validation (20%)
random.shuffle(all_json_files)
split_index = int(0.8 * len(all_json_files))
train_files = all_json_files[:split_index]
val_files = all_json_files[split_index:]

def convert_id_to_int(id_str):
    try:
        return int(id_str)
    except ValueError:
        # Extract numeric part for filenames like 'IMG_3420.JPG'
        return int(''.join(filter(str.isdigit, id_str)))

def copy_images_and_combine_jsons(file_list, src_dir, dest_dir):
    combined = {'images': [], 'annotations': [], 'categories': []}
    for file_name in file_list:
        # Correctly generate the image file name
        base_name = file_name.split('_output')[0]
        image_name = f'IMG_{base_name}.JPG' # Adjusted image file name

        shutil.copy2(os.path.join(src_dir, image_name), dest_dir)

        # Combine JSON data
        with open(os.path.join(src_dir, file_name), 'r') as f:
            data = json.load(f)
            for image in data.get('images', []):
                image['id'] = convert_id_to_int(image['id'])
                combined['images'].append(image)
            for annotation in data.get('annotations', []):
                annotation['image_id'] = convert_id_to_int(annotation['image_id'])
                combined['annotations'].append(annotation)
            if 'categories' in data and not combined['categories']:
                combined['categories'] = data['categories']
    return combined

# Combine and save the training and validation JSON files, and copy images
train_json = copy_images_and_combine_jsons(train_files, directory, train_img_dir)
with open(os.path.join(train_img_dir, 'train.json'), 'w') as f:
    json.dump(train_json, f, indent=4)

val_json = copy_images_and_combine_jsons(val_files, directory, val_img_dir)
with open(os.path.join(val_img_dir, 'val.json'), 'w') as f:
    json.dump(val_json, f, indent=4)
