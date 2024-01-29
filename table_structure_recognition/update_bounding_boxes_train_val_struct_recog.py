import json
from PIL import Image
import os

def crop_and_update_bboxes(json_file, image_folder):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    updated_annotations = []
    updated_images = []

    for image_info in data['images']:
        image_id = image_info['id']
        image_path = os.path.join(image_folder, image_info['file_name'])

        # Find the table bounding box for this image
        table_bbox = None
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id and annotation['category_id'] == 0:
                table_bbox = annotation['bbox']
                break

        if table_bbox:
            # Crop the image to the table bounding box
            with Image.open(image_path) as img:
                left, top, width, height = table_bbox
                cropped_img = img.crop((left, top, left + width, top + height))
                cropped_img.save(image_path)  # Overwrite the original image

            # Update the image info
            updated_image_info = image_info
            updated_image_info['height'] = height
            updated_image_info['width'] = width
            updated_images.append(updated_image_info)

            # Update the coordinates of the other bounding boxes
            for annotation in data['annotations']:
                if annotation['image_id'] == image_id and annotation['category_id'] != 0:
                    x, y, w, h = annotation['bbox']
                    updated_bbox = [x - left, y - top, w, h]
                    annotation['bbox'] = updated_bbox
                    updated_annotations.append(annotation)
        else:
            # If no table is found, keep the original image and annotations
            updated_images.append(image_info)
            updated_annotations.extend([ann for ann in data['annotations'] if ann['image_id'] == image_id])

    # Save the updated data directly to the original JSON file
    updated_data = data
    updated_data['images'] = updated_images
    updated_data['annotations'] = updated_annotations
    with open(json_file, 'w') as outfile:
        json.dump(updated_data, outfile, indent=4)

# Usage examples
train_json = 'data/images/table_structure_recognition_modeling_data/train/train.json'
train_image_folder = 'data/images/table_structure_recognition_modeling_data/train'

val_json = 'data/images/table_structure_recognition_modeling_data/val/val.json'
val_image_folder = 'data/images/table_structure_recognition_modeling_data/val'

crop_and_update_bboxes(train_json, train_image_folder)
crop_and_update_bboxes(val_json, val_image_folder)
