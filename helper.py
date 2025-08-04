from PIL import Image
import numpy as np
import cv2
from io import BytesIO

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
                    'x': left, # the x coordinate of the tile in the original image
                    'y': top, # the y coordinate of the tile in the original image
                })
    except Exception as e:
        print(f"Error tiling image: {e}")
        return 0, 0, []

    return tiles

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

def combine_labels(tiled_labels, img_width, img_height):
    """
    Combine labels from multiple tiles into a single array.
    
    Args:
        labels: list of label strings
    """
    combined_labels = []
    for label in tiled_labels:
        tile_boxes = label['boxes']
        x_offset, y_offset = label['x'], label['y']
        for box in tile_boxes:
            # adjust and normalize all coordinates to [0, 1] range relative to the image size
            x0, y0, x1, y1 = [float(coord) for coord in box.xyxy[0]]
            x0 = (x0 + x_offset) / img_width
            y0 = (y0 + y_offset) / img_height
            x1 = (x1 + x_offset) / img_width
            y1 = (y1 + y_offset) / img_height
            combined_labels.append((int(box.cls), (x0, y0, x1, y1)))

    return combined_labels

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

def download_yolo_labels(filename, boxes):
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

        # Convert label to abbreviated form
        abbreviated_label = 'd' if label.lower() == 'door' else 'w'
        
        # Set color based on label
        color = (0, 0, 255) if label.lower() == 'door' else (0, 255, 0)
        
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 2
        cv2.putText(img, abbreviated_label, (x0, y0 - 10), font, font_scale, color, font_thickness, cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def crop_detected_objects(image, boxes, padding_percent=0.25):
    """
    Crop detected objects from the image using their bounding boxes.
    
    Args:
        image: PIL Image object
        boxes: YOLO detection boxes
    
    Returns:
        List of dictionaries containing:
        - 'crop': cropped PIL Image
        - 'label': object label
        - 'confidence': detection confidence
        - 'bbox': original bounding box coordinates
    """
    crops = []
    for box in boxes:
        # Get coordinates and convert to integers
        x0, y0, x1, y1 = map(int, box.xyxy[0])
        
        # Add some padding around the box
        width = x1 - x0
        height = y1 - y0
        pad_x = int(width * padding_percent)
        pad_y = int(height * padding_percent)
        # padding = 5
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(image.width, x1 + pad_x)
        y1 = min(image.height, y1 + pad_y)

        # Crop the image
        crop = image.crop((x0, y0, x1, y1))
        
        crops.append({
            'crop': crop,
            'label': int(box.cls),
            'confidence': float(box.conf),
            'bbox': (x0, y0, x1, y1)
        })
    
    return crops

def convert_image_to_bytes(image):
    """
    Convert a PIL Image to bytes.
    """
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()