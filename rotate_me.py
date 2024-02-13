import streamlit as st
from PIL import Image
import os
import tempfile
from pointless_deskew import *


def display_image_grid(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.JPG', '.jpg', '.jpeg', '.png'))]
    images = [Image.open(os.path.join(folder_path, f)) for f in image_files]

    if 'selected_image_path' not in st.session_state:
        st.session_state['selected_image_path'] = None

    num_columns = 7
    num_rows = len(images) // num_columns + (1 if len(images) % num_columns > 0 else 0)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col, idx in zip(cols, range(row * num_columns, (row + 1) * num_columns)):
            if idx < len(images):
                with col:
                    st.image(images[idx], width=100, caption=image_files[idx])
                    if st.button('Select', key=image_files[idx]):
                        st.session_state['selected_image_path'] = os.path.join(folder_path, image_files[idx])
                        st.write(f"Selected: {image_files[idx]}")

def process_and_display_image(image_path_or_buffer, predictor, tokenizer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path_or_buffer)[1] if isinstance(image_path_or_buffer, str) else ".png") as tmp_file:
        if isinstance(image_path_or_buffer, str):
            # If image path is provided (selected image)
            img = Image.open(image_path_or_buffer)
        else:
            # If image buffer is provided (uploaded image)
            img = Image.open(image_path_or_buffer)
        img.save(tmp_file, format="PNG")
        tmp_file_path = tmp_file.name

    import time
    start_time = time.time()

    _, _, _, rotated_image = process_image(tmp_file_path, plot_visualization=True, image_scale_factor=0.5)
    _, rotated_image, _, _ = orientation_rotation_estimation(rotated_image, predictor, tokenizer)

    st.image(rotated_image, use_column_width=True)
    end_time = time.time()
    st.write(f"Execution time: {end_time - start_time} seconds")

def app():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Model selection
    model_choice = st.radio(
        "Select OCR model:",
        ('db_mobilenet_v3_large + crnn_mobilenet_v3_small', 'db_resnet50 + parseq')
    )

    if model_choice == 'db_mobilenet_v3_large + crnn_mobilenet_v3_small':
        predictor = ocr_predictor("db_mobilenet_v3_large", "crnn_mobilenet_v3_small", pretrained=True)
    else:  # db_resnet50 + parseq
        predictor = ocr_predictor("db_resnet50", "parseq", pretrained=True)

    st.write("Select one of these images")
    folder_path = 'squashextreme/rotation_test_images'  # Change this to your images folder path
    display_image_grid(folder_path)
    uploaded_image = st.file_uploader("...or upload your image", type=["JPG", "jpg", "jpeg", "png"])

    with st.form("upload_form"):
        image_to_process = None
        if 'selected_image_path' in st.session_state and st.session_state['selected_image_path']:
            selected_image = Image.open(st.session_state['selected_image_path'])
            st.image(selected_image, caption="Selected Image", use_column_width=True)
            image_to_process = st.session_state['selected_image_path']
        elif uploaded_image is not None:
            selected_image = Image.open(uploaded_image)
            st.image(selected_image, caption="Uploaded Image", use_column_width=True)
            image_to_process = uploaded_image

        upload_button = st.form_submit_button('Process Image')

    if upload_button and image_to_process:
        process_and_display_image(image_to_process, predictor, tokenizer)

if __name__ == "__main__":
    app()
