import streamlit as st
import tempfile
from transformers import BertTokenizer
from doctr.models import ocr_predictor
from pointless_deskew import *

with st.form("upload_form"):
    uploaded_file = st.file_uploader("Upload files", type=["png", "jpeg", "jpg", "JPG"])
    upload_button = st.form_submit_button('Upload Image')
    
if upload_button:
    orig_img = Image.open(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        orig_img.save(tmp_file) #, format=image_format)
        tmp_file_path = tmp_file.name

    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    predictor = ocr_predictor("db_mobilenet_v3_large", "crnn_mobilenet_v3_small", pretrained=True)
    #predictor = ocr_predictor("db_resnet50", "parseq", pretrained=True)

    import time
    start_time = time.time()

    skew_angle, _, _, rotated_image = process_image(tmp_file_path, plot_visualization= False, image_scale_factor= 0.5)
    orientation_angle, final_image  = orientation_rotation_estimation(rotated_image, predictor, tokenizer)

    st.image(final_image, use_column_width=True)
    end_time = time.time()
    st.write(f"Execution time: {end_time - start_time} seconds")

    