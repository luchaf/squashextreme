import cv2
import os
import numpy as np
import random
import xml.etree.ElementTree as ET

def synthesize_table_and_create_xml(text_image_folder, number_image_folder, num_rows, table_image_path, xml_file_path):
    # Get all image filenames from the text and number image folders
    text_images = [os.path.join(text_image_folder, f) for f in os.listdir(text_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    number_images = [os.path.join(number_image_folder, f) for f in os.listdir(number_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not text_images or not number_images:
        raise ValueError("No images found in one or both specified folders.")

    num_cols = 4  # Number of columns in the table
    cell_size = (100, 50)  # Size of each cell
    table_width = cell_size[0] * num_cols
    table_height = cell_size[1] * num_rows
    table_img = np.zeros((table_height, table_width, 3), dtype=np.uint8)

    # Initialize XML structure
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = os.path.basename(table_image_path)
    ET.SubElement(root, "path").text = table_image_path
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(table_width)
    ET.SubElement(size, "height").text = str(table_height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    # Create table image and add XML objects for rows and columns
    for row in range(num_rows):
        for col in range(num_cols):
            x_start = col * cell_size[0]
            y_start = row * cell_size[1]

            image_path = random.choice(text_images if col % 2 == 0 else number_images)
            cell_img = cv2.imread(image_path)
            cell_img = cv2.resize(cell_img, cell_size)
            table_img[y_start:y_start + cell_size[1], x_start:x_start + cell_size[0]] = cell_img

    # Add bounding boxes for columns in XML
    for col in range(num_cols):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "table column"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(col * cell_size[0])
        ET.SubElement(bndbox, "ymin").text = "0"
        ET.SubElement(bndbox, "xmax").text = str((col + 1) * cell_size[0])
        ET.SubElement(bndbox, "ymax").text = str(num_rows * cell_size[1])

    # Add bounding boxes for rows in XML
    for row in range(num_rows):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "table row"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "0"
        ET.SubElement(bndbox, "ymin").text = str(row * cell_size[1])
        ET.SubElement(bndbox, "xmax").text = str(num_cols * cell_size[0])
        ET.SubElement(bndbox, "ymax").text = str((row + 1) * cell_size[1])

    # Save the table image
    cv2.imwrite(table_image_path, table_img)

    # Write XML to file
    tree = ET.ElementTree(root)
    with open(xml_file_path, "wb") as xml_file:
        tree.write(xml_file)

# Example usage
synthesize_table_and_create_xml('text_image_path', 'number_image_path', 10, 'table_image.jpg', 'table_image.xml')
