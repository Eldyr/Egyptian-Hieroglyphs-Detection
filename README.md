# Object Detection App

A **Streamlit web application** for detecting hieroglyphs in images using the **YOLOv8 model**.

---

## Features

- Upload images (`.jpg`, `.jpeg`, `.png`) or use an image URL.
- Real-time object detection using **YOLOv8**.
- Adjustable **confidence threshold** using the slider.
- Display detection results with bounding boxes and labels.
- Save and download the annotated images.

---

## Demo

Before 
![photo-1729336225054-735b13ab51d4](https://github.com/user-attachments/assets/6aaff021-0c0a-4384-b5d8-7e0595a67ab1)

After

<img width="3000" height="2250" alt="detection_result" src="https://github.com/user-attachments/assets/9aa56222-604e-4b9b-aa01-d161114b41b5" />


## Installation

1. Clone the repository:

```bash
git clone https://github.com/Eldyr/Egyptian-Hieroglyphs-Detection.git
cd Egyptian-Hieroglyphs-Detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app

```bash
streamlit run app.py
```

## How to Use

Select input type: Upload Image or Image URL.

If uploading, choose an image file. If using URL, paste the image link.

Adjust the confidence slider to filter detections.

View the annotated image with bounding boxes and confidence scores.

Download the annotated image using the Download button.

## Project Structure

object-detection-app/
│
├─ app.py # Main Streamlit app
├─ utils/
│ ├─ detection.py # YOLO model loading & prediction functions
├─ models/
│ └─ detection.pt # YOLOv8 model
├─ outputs/ # Saved annotated images
├─ requirements.txt # Python dependencies
└─ README.md

## Live Demo

https://egyptian-hieroglyphs-detection-thesis.streamlit.app/


