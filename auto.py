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
import helper

def download_yolo_labels(filename, boxes, img_width, img_height):
    """
    Generate YOLO format labels from detected boxes.
    Each box is represented as [class_id, x_center, y_center, width, height].
    """
    yolo_labels = []
    for box in boxes:
        class_id, (x0, y0, x1, y1) = box
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2
        width = x1 - x0
        height = y1 - y0
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    yolo_file = "\n".join(yolo_labels)
    return yolo_file

def annotate_image(image_path, model, confidence=0.4):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    uploaded_image = PIL.Image.open(image_path)
    img_width, img_height = uploaded_image.size
    tiles = helper.split_image(uploaded_image, tile_size=(1920, 1920))
    
    # labeled_tiles = []
    tiled_labels = []
    for tile in tiles:
        res = model.predict(tile['image'], conf=confidence, imgsz=1920)
        boxes = res[0].boxes
        # labeled_tiles.extend(boxes)
        tiled_labels.append({
            'boxes': boxes,
            'x': tile['x'],
            'y': tile['y'],
        })
    # Combine all labels into a single array
    boxes = helper.combine_labels(tiled_labels, img_width, img_height)

    return boxes, img_width, img_height

def main():
    """
    Main function for auto annotation.
    """
    input_dir = "/Users/lesun/BSL-floorplan-analysis/data/newData/images"
    output_dir = "/Users/lesun/BSL-floorplan-analysis/data/newData/labels"

    if not input_dir or not os.path.exists(input_dir):
        print("Invalid directory path. Please provide a valid path.")
        return
    if not output_dir or not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load the pretrained model
    model = YOLO("/Users/lesun/floor-plan-object-detection/models/20250709_1920/best.pt")

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
        boxes, img_width, img_height = annotate_image(image_path, model, confidence=0.25)
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
