import os
import torch
orig_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_torch_load(*args, **kwargs)
torch.load = patched_torch_load
from ultralytics import YOLO
import PIL
from tqdm import tqdm

def download_yolo_labels(filename, boxes, img_width, img_height):
    """
    Generate YOLO format labels from detected boxes.
    Each box is represented as [class_id, x_center, y_center, width, height].
    """
    yolo_labels = []
    for box in boxes:
        class_id = int(box.cls)
        x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2
        width = box.xyxy[0][2] - box.xyxy[0][0]
        height = box.xyxy[0][3] - box.xyxy[0][1]

        # Normalize coordinates
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    yolo_file = "\n".join(yolo_labels)
    return yolo_file

def annotate_image(image_path, model, confidence=0.4):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    uploaded_image = PIL.Image.open(image_path)
    img_width, img_height = uploaded_image.size
    res = model.predict(uploaded_image, conf=confidence, imgsz=1920)
    boxes = res[0].boxes
    return boxes, img_width, img_height

def main():
    """
    Main function for auto annotation.
    """
    input_dir = "/Users/lesun/BSL-floorplan-analysis/data/TrainingData_added/images"
    output_dir = "/Users/lesun/BSL-floorplan-analysis/data/TrainingData_added/labels"

    if not input_dir or not os.path.exists(input_dir):
        print("Invalid directory path. Please provide a valid path.")
        return
    if not output_dir or not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load the pretrained model
    model = YOLO("/Users/lesun/floor-plan-object-detection/models/20250625_1920/best.pt")

    for image_name in tqdm(image_files, desc="Annotating images"):
        # name the output file based on the input image name
        input_name = image_name.split('.')[0]
        label_path = os.path.join(output_dir, f"{input_name}.txt")

        # skip if the image is already annotated
        if os.path.exists(label_path):
            tqdm.write(f"Label file {label_path} already exists. Skipping.")
            continue
        # check if the image file exists
        image_path = os.path.join(input_dir, image_name)
        if not os.path.exists(image_path):
            tqdm.write(f"Image file {image_path} does not exist. Skipping.")
            continue
        
        # Annotate the image and get the bounding boxes
        boxes, img_width, img_height = annotate_image(image_path, model, confidence=0.4)
        if not boxes:
            tqdm.write(f"No objects detected in {image_name}. Skipping.")
            continue
        
        # save the labels in YOLO format
        yolo_file = download_yolo_labels(input_name, boxes, img_width, img_height)
        with open(label_path, 'w') as f:
            f.write(yolo_file)
        tqdm.write(f"Labels saved to {label_path}")

if __name__ == "__main__":
    main()
