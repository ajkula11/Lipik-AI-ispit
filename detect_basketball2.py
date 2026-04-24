from ultralytics import YOLO
import cv2
import os
from pathlib import Path

# Global variable to cache the model (singleton pattern)
_model = None


def get_model():
    """Load and return the YOLO model (loads once, reuses afterwards)"""
    global _model
    if _model is None:
        model_path = './model.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Using custom trained model: {model_path}")
        _model = YOLO(model_path)

    return _model


def detect_basketball(frame, roi=None, conf=0.1):
    """
    Run basketball detection on a frame.

    Args:
        frame: Either a path to an image (str/Path) or an OpenCV image (numpy array)
        roi: Optional Region of Interest as (x1, y1, x2, y2) tuple.
             If provided, detection runs only within this region.

    Returns:
        tuple: (annotated_image, detections)
            - annotated_image: Image with bounding boxes and labels drawn
            - detections: List of dicts with keys: 'class_name', 'confidence', 'bbox' (x1,y1,x2,y2)
    """
    model = get_model()

    # Load image if path is provided
    if isinstance(frame, (str, Path)):
        image = cv2.imread(str(frame))
        if image is None:
            raise ValueError(f"Could not read image from path: {frame}")
    else:
        # Assume it's already an OpenCV image (numpy array)
        image = frame.copy()

    # Handle ROI if provided
    roi_offset_x, roi_offset_y = 0, 0
    if roi is not None:
        x1, y1, x2, y2 = roi
        roi_offset_x, roi_offset_y = x1, y1
        # Crop image to ROI
        roi_image = image[y1:y2, x1:x2]
        inference_image = roi_image
    else:
        inference_image = image

    # cv2.imshow('Basketball Detection', inference_image)
    # cv2.waitKey(0)

    # Run detection
    results = model(inference_image, stream=True, verbose=False)

    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            if confidence > conf:

                # Check if detected object is a sports ball (COCO) or basketball (custom model)
                # if (class_name == 'sports ball' or class_name == 'basketball'):

                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Adjust coordinates if ROI was used
                x1 += roi_offset_x
                y1 += roi_offset_y
                x2 += roi_offset_x
                y2 += roi_offset_y

                # Draw bounding box on original image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store detection info
                detections.append({
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })

    return image, detections
