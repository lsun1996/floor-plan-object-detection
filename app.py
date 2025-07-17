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
import pytesseract
import re


def clean_ocr_text(text: str) -> str:
    """Clean OCR text to keep only alphanumeric characters and spaces."""
    # Remove any non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned

def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()
    
    model = YOLO("/Users/lesun/floor-plan-object-detection/models/20250709_1920/best.pt")

    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        uploaded_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        # Model Options
        confidence = setting.get_model_confidence()

        # Multiselect for selecting labels
        available_labels = list(model.names.values())
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if uploaded_img:
            source_img_name = uploaded_img.name.split('.')[0]
            # Opening the uploaded image
            source_img = PIL.Image.open(uploaded_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,caption="Uploaded Image",use_column_width=True)
        else:
            st.warning("Please upload an image.")


    if st.sidebar.button('Detect Objects'):
        if not uploaded_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            # split image in col2 into tiles of 1920x1920
            tiles = helper.split_image(source_img, tile_size=(1920, 1920))

            # run detection on each tile
            labeled_tiles = []
            tiled_labels = []

            all_filtered_boxes = [] # variable for counting detected objects

            for tile in tiles:
                res = model.predict(tile['image'], conf=confidence, imgsz=1920)
                filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels]
                # crops = helper.crop_detected_objects(tile['image'], filtered_boxes)

                # for crop in crops:
                #     label_name = model.names[int(crop['label'])]
                #     ocr_text = pytesseract.image_to_string(crop['crop'])
                #     cleaned_text = clean_ocr_text(ocr_text)
                #     if cleaned_text:
                #         st.write(f"Detected {label_name}: {cleaned_text}")
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

            with col2:
                st.image(combined_image, caption='Detected Image',use_column_width=True)
                # Count detected objects and display counts
                object_counts = helper.count_detected_objects(model, all_filtered_boxes)
                st.write("\n\nDetected Objects and their Counts:")
                for label, count in object_counts.items():
                    st.write(f"{label}: {count}")

            # combine all detected boxes into a single list

            yolo_file = helper.download_yolo_labels(source_img_name, combined_labels)
            st.download_button(
                label="Download YOLO Labels",
                data=yolo_file,
                file_name=f'{source_img_name}.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()
