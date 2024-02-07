import os
import json
import random
import shutil
from PIL import Image

class TatrImageDataProcessor:
    """
    A class to process image datasets for machine learning models, handling tasks like
    creating train/validation splits, copying images, combining JSON files, cropping images based on bounding boxes,
    and filtering annotations and categories in JSON files.

    Attributes:
        base_directory (str): The base directory where the image datasets are located.
    """

    def __init__(self, base_directory='data/images'):
        """
        Initializes the TatrImageDataProcessor with a specific base directory.

        Args:
            base_directory (str): The base directory where the image datasets are located.
        """
        self.base_directory = base_directory

    def get_train_val_split(self, file_extension='_output.json'):
        """
        Generates a train and validation split of the files in the base directory.

        Args:
            file_extension (str): The file extension to filter files by. Defaults to '_output.json'.

        Returns:
            tuple: A tuple containing two lists - the first for training files and the second for validation files.
        """
        # List all files in the base directory with the specified file extension
        all_json_files = [f for f in os.listdir(self.base_directory) if f.endswith(file_extension)]
        # Randomly shuffle the list of files to ensure random splits
        random.shuffle(all_json_files)
        # Calculate the index at which to split the files (80% train, 20% val)
        split_index = int(0.8 * len(all_json_files))
        # Split the files into training and validation sets
        train_files = all_json_files[:split_index]
        val_files = all_json_files[split_index:]
        return train_files, val_files

    def process_images(self, task_specific_directory, train_files, val_files):
        """
        Processes images for a specific task, including creating directories, copying images, and combining JSON files.

        Args:
            task_specific_directory (str): The directory for the specific task (e.g., recognition, detection).
            train_files (list): The list of training files.
            val_files (list): The list of validation files.
        """
        # Construct paths for the task-specific directories
        task_dir = os.path.join(self.base_directory, task_specific_directory)
        train_img_dir = os.path.join(task_dir, 'train')
        val_img_dir = os.path.join(task_dir, 'val')
        # Create the directories if they do not exist
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)

        # Process training and validation files separately
        train_json, _ = self.copy_images_and_combine_jsons(train_files, train_img_dir, 0)
        with open(os.path.join(train_img_dir, 'train.json'), 'w') as f:
            json.dump(train_json, f, indent=4)

        val_json, _ = self.copy_images_and_combine_jsons(val_files, val_img_dir, 0)
        with open(os.path.join(val_img_dir, 'val.json'), 'w') as f:
            json.dump(val_json, f, indent=4)


    def copy_images_and_combine_jsons(self, file_list, dest_dir, start_annotation_id):
        """
        Copies images to a destination directory and combines JSON data from multiple files into a single JSON object.

        This function iterates through a list of filenames, copies each image to the specified destination directory,
        and combines their corresponding JSON data (images, annotations, categories) into a single JSON object. 
        It ensures each image and annotation has a unique ID, starting from 0 for images and starting from 
        `start_annotation_id` for annotations. It also accurately captures and sets the dimensions (width and height) 
        of each image.

        Args:
            file_list (List[str]): A list of filenames to process. Each filename should correspond to a JSON file 
                that contains metadata (including annotations) for a single image.
            dest_dir (str): The destination directory path where the images should be copied to.
            start_annotation_id (int): The starting ID for annotations. This allows for sequential numbering 
                of annotations across multiple JSON files.

        Returns:
            Tuple[Dict, int]: A tuple containing two elements:
                - A dictionary object that combines all images, annotations, and categories from the processed files.
                - The next available annotation ID, which can be used as the starting ID for processing additional files.

        """
        combined_json = {'images': [], 'annotations': [], 'categories': []}
        image_id_counter = 0  # Initialize a counter for image IDs starting at 0
        annotation_id_counter = start_annotation_id  # Initialize annotation ID counter

        for file_name in file_list:
            base_name = file_name.split('_output')[0]
            image_name = f'IMG_{base_name}.JPG'
            image_path = os.path.join(self.base_directory, image_name)
            # Copy the image to the destination directory
            shutil.copy2(image_path, dest_dir)
            
            # Open the image to get its dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            with open(os.path.join(self.base_directory, file_name), 'r') as f:
                data = json.load(f)
                
                # Add image information with updated image ID and actual dimensions
                combined_json['images'].append({
                    "file_name": image_name,
                    "height": height,
                    "width": width,
                    "id": image_id_counter
                })
                
                # Assuming categories are consistent across files, update if needed
                if 'categories' in data and not combined_json['categories']:
                    combined_json['categories'] = data['categories']

                # Update annotations with new image_id and sequential annotation IDs
                for annotation in data['annotations']:
                    new_annotation = annotation.copy()
                    new_annotation['image_id'] = image_id_counter
                    new_annotation['id'] = annotation_id_counter
                    combined_json['annotations'].append(new_annotation)
                    annotation_id_counter += 1  # Increment annotation ID for the next annotation

            image_id_counter += 1  # Increment image ID for the next image

        return combined_json, annotation_id_counter


    @staticmethod
    def convert_id_to_int(id_str):
        """
        Converts an ID string to an integer, extracting digits if necessary.

        Args:
            id_str (str): The ID string to convert.

        Returns:
            int: The converted integer ID.
        """
        try:
            return int(id_str)
        except ValueError:
            # Extract digits from the string and convert to int
            return int(''.join(filter(str.isdigit, id_str)))

    def crop_and_update_bboxes(self, json_file, image_folder):
        """
        Crops images based on table bounding boxes and updates the bounding boxes of annotations accordingly.

        Args:
            json_file (str): The JSON file containing image and annotation data.
            image_folder (str): The folder where the images are stored.
        """
        with open(json_file, 'r') as file:
            data = json.load(file)
        updated_annotations = []
        updated_images = []

        for image_info in data['images']:
            image_id = image_info['id']
            image_path = os.path.join(image_folder, image_info['file_name'])
            # Find the table bounding box for this image
            table_bbox = next((anno['bbox'] for anno in data['annotations'] if anno['image_id'] == image_id and anno['category_id'] == 0), None)
            
            if table_bbox:
                # Crop the image and update annotations if table bbox is found
                self.crop_image(table_bbox, image_path)
                self.update_annotations(data, table_bbox, image_id, updated_annotations, updated_images, image_info)
            else:
                # Keep original image and annotations if no table bbox is found
                updated_images.append(image_info.copy())
                updated_annotations.extend([ann for ann in data['annotations'] if ann['image_id'] == image_id])
        
        # Save the updated data back to the JSON file
        updated_data = {'images': updated_images, 'annotations': updated_annotations, 'categories': data['categories']}
        with open(json_file, 'w') as outfile:
            json.dump(updated_data, outfile, indent=4)

    @staticmethod
    def crop_image(table_bbox, image_path):
        """
        Crops an image to the specified bounding box.

        Args:
            table_bbox (tuple): The bounding box to crop the image to (left, top, width, height).
            image_path (str): The path to the image file.
        """
        with Image.open(image_path) as img:
            left, top, width, height = table_bbox
            cropped_img = img.crop((left, top, left + width, top + height))
            cropped_img.save(image_path)  # Overwrite the original image with the cropped version

    @staticmethod
    def update_annotations(data, table_bbox, image_id, updated_annotations, updated_images, image_info):
        """
        Updates the annotations based on the new bounding box after an image has been cropped.

        Args:
            data (dict): The original data loaded from the JSON file.
            table_bbox (tuple): The bounding box of the table used for cropping.
            image_id (int): The ID of the current image being processed.
            updated_annotations (list): The list to append updated annotations to.
            updated_images (list): The list to append updated images to.
            image_info (dict): The original image info dictionary.
        """
        left, top, width, height = table_bbox
        updated_image_info = image_info.copy()
        updated_image_info.update({'height': height, 'width': width})
        updated_images.append(updated_image_info)
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                x, y, w, h = annotation['bbox']
                updated_bbox = [x - left, y - top, w, h]  # Adjust bbox coordinates based on the crop
                annotation['bbox'] = updated_bbox
                updated_annotations.append(annotation)


    def filter_annotations_and_categories(self, json_file_path):
        """
        Filters annotations and categories in a JSON file to include only those related to tables,
        and reassigns annotation IDs to ensure they are sequential starting from 0.

        Args:
            json_file_path (str): The path to the JSON file to filter.
        """
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Filter annotations and categories related to tables
        filtered_annotations = [anno for anno in data['annotations'] if anno['category_id'] == 0]
        filtered_categories = [category for category in data['categories'] if category['id'] == 0]
        
        # Reassign IDs to filtered annotations to ensure they are sequential starting from 0
        for i, annotation in enumerate(filtered_annotations):
            annotation['id'] = i  # Reassign ID

        # Update the data with filtered and updated annotations and categories
        data['annotations'] = filtered_annotations
        data['categories'] = filtered_categories
        
        with open(json_file_path, 'w') as file:
            json.dump(data, file, indent=4)


def main():
    """
    The main function to process image datasets. It sets up the TatrImageDataProcessor, creates consistent train/val splits,
    and processes images for both table recognition and detection tasks.
    """
    processor = TatrImageDataProcessor()
    # Generate a consistent train/val split for all tasks
    train_files, val_files = processor.get_train_val_split()

    tasks = [
        ('table_structure_recognition_modeling_data', processor.crop_and_update_bboxes),
        ('table_detection_modeling_data', processor.filter_annotations_and_categories)
    ]

    for task_directory, update_function in tasks:
        # Process images with the consistent train/val split
        processor.process_images(task_directory, train_files, val_files)

        train_json = f'{processor.base_directory}/{task_directory}/train/train.json'
        val_json = f'{processor.base_directory}/{task_directory}/val/val.json'
        train_img_folder = f'{processor.base_directory}/{task_directory}/train'
        val_img_folder = f'{processor.base_directory}/{task_directory}/val'

        if update_function == processor.crop_and_update_bboxes:
            update_function(train_json, train_img_folder)
            update_function(val_json, val_img_folder)
        else:
            update_function(train_json)
            update_function(val_json)

if __name__ == '__main__':
    main()
