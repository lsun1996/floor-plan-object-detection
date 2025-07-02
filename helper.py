import PIL
import pandas as pd
import numpy as np
import cv2

def count_detected_objects(model, filtered_boxes):
    """
    Count detected objects and return a dictionary of counts.
    """
    object_counts = {}
    for box in filtered_boxes:
        # Extract class label of detected object
        label = model.names[int(box.cls)]
        # Update count in dictionary
        object_counts[label] = object_counts.get(label, 0) + 1
    return object_counts

def generate_csv(object_counts):
    """
    Generate CSV data from detected object counts.
    """
    csv_data = pd.DataFrame(list(object_counts.items()), columns=['Label', 'Count'])
    csv_file = csv_data.to_csv(index=False)
    return csv_file

def download_yolo_labels(filename, boxes):
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
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
    yolo_file = "\n".join(yolo_labels)
    return yolo_file

def draw_custom_labels(image, boxes, model):
    """
    Draw custom labels on the image based on detected boxes.
    """
    if hasattr(image, 'mode') and image.mode != 'RGB':
        image = image.convert('RGB')
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in boxes:
        label = model.names[int(box.cls)]
        x0, y0, x1, y1 = map(int, box.xyxy[0])

        if label.lower() == 'door':
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 2
        cv2.putText(img, label, (x0, y0 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img