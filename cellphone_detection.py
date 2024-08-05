import numpy as np
import cv2
from pathlib import Path
import torch
from yolov5 import load  # YOLOv5 라이브러리에서 모델을 로드합니다.
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Constants
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Red

# Define class labels
CLASS_LABELS = {0: "cell phone"}  # Update this dictionary based on your model's class indices

def visualize(image, detections, threshold=0.2) -> np.ndarray:
    """Draws bounding boxes on the input image and returns it."""
    for det in detections:
        if det['confidence'] >= threshold:
            bbox = det['bbox']
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # Draw label and score
            class_id = det['class']
            label = CLASS_LABELS.get(class_id, f'Class {class_id}')
            label_text = f'{label} ({det["confidence"]:.2f})'
            text_location = (MARGIN + int(bbox[0]), MARGIN + ROW_SIZE + int(bbox[1]))
            cv2.putText(image, label_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image

def run_yolov5_detection(weights, source, conf_thres=0.2):
    # Load YOLOv5 model
    model = load(weights)

    # Load image
    img = cv2.imread(source)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(img_rgb)

    # Extract detections
    detections = []
    for det in results.xyxy[0]:  # Get predictions
        if det[4] >= conf_thres:  # confidence threshold
            detections.append({
                'class': int(det[5]),
                'confidence': float(det[4]),
                'bbox': [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
            })

    return img, detections

# YOLOv5 model and image directory
weights = 'C:/Users/eintelligence/PycharmProjects/cellphone_detection/yolov5/best.pt'
image_directory = 'C:/Users/eintelligence/PycharmProjects/cellphone_detection/phone_dataset/test/images'

# Process each image
image_files = list(Path(image_directory).glob('*.jpg'))  # Adjust the pattern as needed
for image_file in image_files:
    image_path = str(image_file)
    img, detections = run_yolov5_detection(weights, image_path)

    # Process the detection results
    annotated_image = visualize(img, detections, threshold=0.2)
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
