
# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from doctr.file_utils import is_tf_available
from doctr.io import DocumentFile
from doctr.utils.visualization import visualize_page
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
import torch

DET_ARCHS = [
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]
RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]


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



"""Build a streamlit layout"""
# Wide mode
st.set_page_config(layout="wide")

# Designing the interface
st.title("docTR: Document Text Recognition")
# For newline
st.write("\n")
# Instructions
st.markdown("*Hint: click on the top-right corner of an image to enlarge it!*")
# Set the columns
cols = st.columns((1, 1, 1, 1))
cols[0].subheader("Input page")
cols[1].subheader("Segmentation heatmap")
cols[2].subheader("OCR output")
cols[3].subheader("Page reconstitution")

# Sidebar
# File selection
st.sidebar.title("Document selection")
# Disabling warning
#st.set_option("deprecation.showfileUploaderEncoding", False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader("Upload files", type=["pdf", "png", "jpeg", "jpg"])
forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        doc = DocumentFile.from_pdf(uploaded_file.read())
    else:
        doc = DocumentFile.from_images(uploaded_file.read())
    page_idx = st.sidebar.selectbox("Page selection", [idx + 1 for idx in range(len(doc))]) - 1
    page = doc[page_idx]
    cols[0].image(page)

# Model selection
st.sidebar.title("Model selection")
st.sidebar.markdown("**Backend**: " + ("TensorFlow" if is_tf_available() else "PyTorch"))
det_arch = st.sidebar.selectbox("Text detection model", det_archs)
reco_arch = st.sidebar.selectbox("Text recognition model", reco_archs)

# For newline
st.sidebar.write("\n")
# Only straight pages or possible rotation
st.sidebar.title("Parameters")
assume_straight_pages = st.sidebar.checkbox("Assume straight pages", value=True)
st.sidebar.write("\n")
# Straighten pages
straighten_pages = st.sidebar.checkbox("Straighten pages", value=False)
st.sidebar.write("\n")
# Binarization threshold
bin_thresh = st.sidebar.slider("Binarization threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
st.sidebar.write("\n")

if st.sidebar.button("Analyze page"):
    if uploaded_file is None:
        st.sidebar.write("Please upload a document")

    else:
        with st.spinner("Loading model..."):
            predictor = load_predictor(
                det_arch, reco_arch, assume_straight_pages, straighten_pages, bin_thresh, forward_device
            )

        with st.spinner("Analyzing..."):
            # Forward the image to the model
            seg_map = forward_image(predictor, page, forward_device)
            seg_map = np.squeeze(seg_map)
            seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Plot the raw heatmap
            fig, ax = plt.subplots()
            ax.imshow(seg_map)
            ax.axis("off")
            cols[1].pyplot(fig)

            # Plot OCR output
            out = predictor([page])
            fig = visualize_page(out.pages[0].export(), out.pages[0].page, interactive=False, add_labels=False)
            cols[2].pyplot(fig)

            # Page reconsitution under input page
            page_export = out.pages[0].export()
            if assume_straight_pages or (not assume_straight_pages and straighten_pages):
                img = out.pages[0].synthesize()
                cols[3].image(img, clamp=True)

            # Display JSON
            st.markdown("\nHere are your analysis results in JSON format:")
            st.json(page_export)
