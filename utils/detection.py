import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO



# Load the YOLO model

def load_model(model_path="models/detection.pt"):
    """
    Load YOLOv8 model from local path.
    """
    model = YOLO(model_path)
    return model



# Load the image from URL

def load_image_from_url(url: str):
    """
    Download an image from a URL and return a PIL Image.
    Returns None if download fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print("Error loading image from URL:", e)
        return None



# Run predictions and draw the bounding boxes

def predict(model, pil_image, conf_thresh=0.3):
    """
    Run YOLOv8 detection on a PIL image and return:
      - list of detections (class, confidence, bbox)
      - annotated output image as PIL
    """
    import cv2
    import numpy as np
    from PIL import Image

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model.predict(img_rgb, verbose=False)

    detections = []
    names = results[0].names

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for score, cls, bbox in zip(scores, classes, boxes):
        if score < conf_thresh:
            continue  # skip detections below threshold

        x1, y1, x2, y2 = bbox
        class_name = names[int(cls)]
        label = f"{class_name} {score:.2f}"

        detections.append({
            "class": class_name,
            "confidence": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_bgr, (x1, y1 - h - 6), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    img_out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(img_out_rgb)

    return detections, annotated_image
