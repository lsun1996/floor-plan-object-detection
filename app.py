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

def split_image(image, tile_size=(1920, 1920)):
    """
    Split image into tiles and return them as PIL Image objects in memory.
    
    Args:
        image: PIL Image object or file-like object
        tile_size: tuple of (width, height) for each tile
    
    Returns:
        tuple: (num_tiles_x, num_tiles_y, tiles_list)
        tiles_list: list of dictionaries with 'image', 'x', 'y' keys
    """

    Image.MAX_IMAGE_PIXELS = None
    # base_name = os.path.basename(image_path).split('.')[0]

    try:
        image_width, image_height = image.size
        tile_width, tile_height = tile_size

        num_tiles_x = image_width // tile_width
        num_tiles_y = image_height // tile_height

        if image_width % tile_width != 0:
            num_tiles_x += 1
        if image_height % tile_height != 0:
            num_tiles_y += 1

        tiles = []
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                left = x * tile_width
                top = y * tile_height
                right = min(left + tile_width, image_width)
                bottom = min(top + tile_height, image_height)

                # crop and save image tile
                tile = image.crop((left, top, right, bottom))
                tiles.append({
                    'image': tile,
                    'x': left,
                    'y': top,
                    'position': (left, top, right, bottom)
                })
    except Exception as e:
        print(f"Error tiling image: {e}")
        return 0, 0, []

    return num_tiles_x, num_tiles_y, tiles

def combine_image_tiles(tiles, image_width, image_height):
    """
    Combine image tiles into a single image.
    
    Args:
        tiles: list of dictionaries with 'image', 'x', 'y' keys
        image_width: width of the final combined image
        image_height: height of the final combined image
    
    Returns:
        PIL Image object of the combined image
    """
    combined_image = Image.new('RGB', (image_width, image_height))

    for tile in tiles:
        x, y = tile['x'], tile['y']
        tile_image = tile['image']
        box = (x, y, x + tile_image.width, y + tile_image.height)

        # Ensure the box does not go out of bounds
        box = (
            max(0, box[0]),  # Ensure x is not negative
            max(0, box[1]),  # Ensure y is not negative
            min(image_width, box[2]),  # Ensure x + width is not beyond image
            min(image_height, box[3])  # Ensure y + height is not beyond image
        )

        combined_image.paste(tile_image, box)

    return combined_image

def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()
    
    model = YOLO("/Users/lesun/floor-plan-object-detection/models/20250708_1920/best.pt")

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
            num_tiles_x, num_tiles_y, tiles = split_image(source_img, tile_size=(1920, 1920))

            # run detection on each tile
            labeled_tiles = []
            all_filtered_boxes = []

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
                    'position': tile['position']
                })
                all_filtered_boxes.extend(filtered_boxes)

            # combine all labeled tiles into a single image
            combined_image = combine_image_tiles(labeled_tiles, source_img.width, source_img.height)
            with col2:
                st.image(combined_image, caption='Detected Image',use_column_width=True)
                # Count detected objects and display counts
                object_counts = helper.count_detected_objects(model, all_filtered_boxes)
                st.write("\n\nDetected Objects and their Counts:")
                for label, count in object_counts.items():
                    st.write(f"{label}: {count}")

            # todo: download the image tiles and labels for tiles, not the combined ones
            yolo_file = helper.download_yolo_labels(source_img_name, all_filtered_boxes)
            st.download_button(
                label="Download YOLO Labels",
                data=yolo_file,
                file_name=f'{source_img_name}.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()
