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

def main():
    """
    Main function for the Streamlit app.
    """
    setting.configure_page()
    
    model = YOLO("/Users/lesun/floor-plan-object-detection/models/20250625_1920/best.pt")

    # Creating sidebar
    with st.sidebar:
        st.header("Image Configuration")     # Adding header to sidebar
        # Adding file uploader to sidebar for selecting images
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        # Model Options
        confidence = setting.get_model_confidence()

        # Multiselect for selecting labels
        # available_labels = ['Column', 'Curtain Wall', 'Dimension', 'Door', 'Railing', 'Sliding Door', 'Stair Case', 'Wall', 'Window']
        available_labels = list(model.names.values())
        selected_labels = setting.select_labels(available_labels)

    # Creating main page heading
    st.title("Floor Plan Object Detection using YOLOv8")

    # Creating two columns on the main page
    col1, col2 = st.columns(2)

    # Adding image to the first column if image is uploaded
    with col1:
        if source_img:
            source_img_name = source_img.name.split('.')[0]
            # Opening the uploaded image
            uploaded_image = PIL.Image.open(source_img)
            # Adding the uploaded image to the page with a caption
            st.image(source_img,caption="Uploaded Image",use_column_width=True)
        else:
            st.warning("Please upload an image.")


    if st.sidebar.button('Detect Objects'):
        if not source_img:
            st.warning("Please upload an image before detecting objects.")
        else:
            res = model.predict(uploaded_image, conf=confidence, imgsz=1920)
            filtered_boxes = [box for box in res[0].boxes if model.names[int(box.cls)] in selected_labels]
            res[0].boxes = filtered_boxes
            # res_plotted = res[0].plot()[:, :, ::-1]
            res_plotted = helper.draw_custom_labels(uploaded_image, filtered_boxes, model)
            with col2:
                st.image(res_plotted, caption='Detected Image',use_column_width=True)
                # Count detected objects and display counts
                object_counts = helper.count_detected_objects(model, filtered_boxes)
                st.write("\n\nDetected Objects and their Counts:")
                for label, count in object_counts.items():
                    st.write(f"{label}: {count}")

            # # Generate and provide download link for CSV
            # csv_file = helper.generate_csv(object_counts)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv_file,
            #     file_name='detected_objects.csv',
            #     mime='text/csv'
            # )
            # Generate and provide download link for CSV
            yolo_file = helper.download_yolo_labels(source_img_name, filtered_boxes)
            st.download_button(
                label="Download YOLO Labels",
                data=yolo_file,
                file_name=f'{source_img_name}.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()
