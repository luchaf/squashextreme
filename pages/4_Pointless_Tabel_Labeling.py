import streamlit as st
import os
from pathlib import Path
import subprocess
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager

def run(img_dir, labels):

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


def get_unique_filenames(source_dir, target_file):
    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        print(f"The source directory {source_dir} does not exist.")
        return

    # Get the list of files in the source directory
    files = os.listdir(source_dir)

    # Extract unique filenames without extensions
    unique_filenames = set()
    for file in files:
        name, ext = os.path.splitext(file)
        st.write(name)
        unique_filenames.add(name)

    # Ensure the target directory exists
    target_dir = os.path.dirname(target_file)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Write the unique filenames to the target file
    with open(target_file, 'w') as f:
        for name in sorted(unique_filenames):
            f.write(f"{name}\n")

    print(f"Unique filenames have been written to {target_file}")



if __name__ == "__main__":

    # Define the path for the folder one level above the current script
    current_script_path = os.path.dirname(__file__)
    parent_directory = os.path.join(current_script_path, os.pardir)
    source_folder = os.path.normpath(os.path.join(parent_directory, 'table_structure_recognition/data'))
    target_folder = os.path.normpath(os.path.join(parent_directory, 'table_structure_recognition/data/images'))

    run(target_folder, ["table row", "table column"])

    with st.form("voc2coco"):
        confirm_button = st.form_submit_button('voc2coco script')
        if confirm_button:
            # Define the path to the 'voc2coco.py' script
            voc2coco_script_path = os.path.join(source_folder, 'voc2coco.py')
            # Ensure that the script exists
            if not os.path.isfile(voc2coco_script_path):
                raise FileNotFoundError(f"voc2coco.py not found at {voc2coco_script_path}")

            # Make sure the target directory exists, if not, create it
            if not os.path.exists(source_folder):
                os.makedirs(source_folder)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            unique_filenames_path = os.path.join(source_folder, 'unique_filenames.txt')
            # Define the full path for the new file
            save_path = os.path.join(source_folder, unique_filenames_path)
            get_unique_filenames(target_folder, save_path)
                                       
            # Define other paths and ensure directories exist
            ann_dir = target_folder
            ann_ids = save_path
            labels = os.path.join(source_folder, 'labels.txt')
            output = os.path.join(source_folder, 'output.json')
            for path in [ann_dir, os.path.dirname(ann_ids), os.path.dirname(output)]:
                Path(path).mkdir(parents=True, exist_ok=True)

            # Define the command and arguments
            command = 'python'
            args = [
                '--ann_dir', ann_dir,
                '--ann_ids', ann_ids,
                '--labels', labels,
                '--output', output,
                '--ext', 'xml'
            ]

            # Combine the command, script path, and arguments into a single list
            cmd = [command, voc2coco_script_path] + args

            # Run the command
            subprocess.run(cmd)

