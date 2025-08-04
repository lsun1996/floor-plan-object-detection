import streamlit as st
import torch
orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load
from ultralytics import YOLO
import PIL
import helper
import setting
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import re
import fitz
import io


def run_detection(source_img_name, model, source_img, confidence, selected_labels):
    """
    Run object detection on the given image using the specified model.
    
    Args:
        model: YOLO model instance
        image: PIL Image object
        confidence: Confidence threshold for detection
        selected_labels: List of labels to filter detections
    
    Returns:
        List of detected boxes with labels and confidence scores
    """
    tiles = helper.split_image(source_img, tile_size=(1920, 1920))

    # run detection on each tile
    labeled_tiles = []
    tiled_labels = []

    all_filtered_boxes = [] # variable for counting detected objects

    for tile in tiles:
        res = model.predict(tile['image'], conf=confidence, imgsz=1920)
        filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels]
        res[0].boxes = filtered_boxes
        res_plotted = helper.draw_custom_labels(tile['image'], filtered_boxes, model)

        if isinstance(res_plotted, np.ndarray): # convert numpy array to PIL Image
            res_plotted = Image.fromarray(res_plotted)
            res_plotted = res_plotted.convert("RGB")

        labeled_tiles.append({
            'image': res_plotted,
            'x': tile['x'],
            'y': tile['y'],
        })

        tiled_labels.append({
            'boxes': filtered_boxes,
            'x': tile['x'],
            'y': tile['y'],
        })
        all_filtered_boxes.extend(filtered_boxes)

    # combine all labeled tiles into a single image
    combined_image = helper.combine_image_tiles(labeled_tiles, source_img.width, source_img.height)
    # combine all labels into a single array
    combined_labels = helper.combine_labels(tiled_labels, source_img.width, source_img.height)

    object_counts = helper.count_detected_objects(model, all_filtered_boxes)
    yolo_file = helper.download_yolo_labels(source_img_name, combined_labels)

    return combined_image, object_counts, yolo_file

def display_image_group(source_img, combined_image, object_counts, source_img_name, yolo_file):
    """
    Display a group of images and their detection results.
    
    Args:
        source_img: Original uploaded image
        combined_image: Processed image with detections
        object_counts: Dictionary of detected object counts
        combined_labels: Combined YOLO labels
        source_img_name: Name of the source image
        index: Group index for unique keys
        col1: First column for source image
        col2: Second column for results
    """
    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        # Adding the uploaded image to the page with a caption
        if source_img:
            st.image(source_img,caption="Uploaded Image",use_column_width=True)
        else:
            st.warning("Please upload an image.")
                
    # if st.sidebar.button('Detect Objects'):
    if not source_img:
        st.warning("Please upload an image before detecting objects.")
    else:
        with col2:
            st.image(combined_image, caption='Detected Image',use_column_width=True)
            st.write("\n\nDetected Objects and their Counts:")
            for label, count in object_counts.items():
                st.write(f"{label}: {count}")
            # Create two columns for horizontal alignment of buttons
            button_col1, button_col2 = st.columns(2)

            with button_col1:
                st.download_button(
                    label="Download YOLO Labels",
                    data=yolo_file,
                    file_name=f'{source_img_name}.txt',
                    mime='text/plain',
                    key=f'yolo_labels_{source_img_name}'
                )

            with button_col2:
                st.download_button(
                    label="Download Annotated Image",
                    data=helper.convert_image_to_bytes(combined_image),
                    file_name=f'{source_img_name}_annotated.png',
                    mime='image/png',
                    key=f'annotated_image_{source_img_name}'
                )

def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()
    
    model_path = "best.pt"
    model = YOLO(model_path)

    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        # uploaded_img = st.sidebar.file_uploader(
        #     "Choose an image...", type=("jpg", "jpeg", "png"))
        uploaded_pdf = st.sidebar.file_uploader(
            "Choose a PDF file...", type=("pdf"), key="pdf_uploader")
        
        if uploaded_pdf is not None:
            pdf_name = uploaded_pdf.name.split('.')[0]
            doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            num_pages = doc.page_count if doc else 0

        # Add page numbers input
        page_numbers_str = st.text_input(
            "Page Numbers (comma-separated, e.g., 1,2,3):", 
            value="1", 
            key="page_numbers_input"
        )

        if not re.fullmatch(r'\s*\d+(\s*,\s*\d+)*\s*', page_numbers_str):
            st.warning("Please enter valid page numbers (comma-separated integers).")
        else:
            page_numbers = [int(num) for num in re.findall(r'\d+', page_numbers_str)]

        if uploaded_pdf and page_numbers:
            source_imgs = []
            for page in page_numbers:
                if 1 <= page <= num_pages:
                    pdf_page = doc.load_page(page - 1)
                    pix = pdf_page.get_pixmap()
                    img = PIL.Image.open(io.BytesIO(pix.tobytes()))
                    source_imgs.append(img)
                    
        # # Extract image name without extension
        # source_img_name = uploaded_pdf.name.split('.')[0] if uploaded_pdf else None
        # source_img = PIL.Image.open(uploaded_pdf) if uploaded_pdf else None

        # Model Options
        confidence = setting.get_model_confidence()

        # Multiselect for selecting labels
        available_labels = list(model.names.values())
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")
    if not uploaded_pdf or not source_imgs:
        st.warning("Please upload a PDF file to start detection.\nIf you wish to detect all objects, leave the label selection empty.")

    detect_button_disabled = not uploaded_pdf or not source_imgs

    if st.sidebar.button('Detect Objects', disabled=detect_button_disabled):
        for index, source_img in zip(page_numbers, source_imgs):
            source_img_name = f"{pdf_name}_page_{index}"
            combined_image, object_counts, yolo_file = run_detection(source_img_name, model, source_img, confidence, selected_labels)
            display_image_group(source_img, combined_image, object_counts, source_img_name, yolo_file)

if __name__ == "__main__":
    main()
