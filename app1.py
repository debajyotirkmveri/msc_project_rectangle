import os
pip install opencv-python==4.5.5.62
import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
#@st.cache(allow_output_mutation=True)
#def load_model():
    #return YOLO('/home/sysadm/Downloads/building/runs/detect/train/weights/best.pt')

#model = load_model()
# Load the YOLO model
model = YOLO('/home/sysadm/Downloads/building_rectangle/runs/detect/weights/best.pt')
# Decoding according to the .yaml file class names order
decoding_of_predictions = {0: 'undamagedcommercialbuilding', 
                           1: 'undamagedresidentialbuilding',
                           2: 'damagedresidentialbuilding',
                           3: 'damagedcommercialbuilding'}

# Define colors for different classes
class_colors = {'undamagedcommercialbuilding': (0, 255, 0),  # Green
                'undamagedresidentialbuilding': (255, 0, 0),  # Red
                'damagedresidentialbuilding': (0, 0, 255),  # Blue
                'damagedcommercialbuilding': (128, 128, 128)}  # Grey

def main():
    st.title("Object Detection with YOLOv8")
    st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
    st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🎯</p>", unsafe_allow_html=True)  
    st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            detect_objects(image)

def detect_objects(image):
    # Resize the image to 512x512
    image_resized = image.resize((512, 512))

    # Convert the PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model.predict(img_cv, save=True, iou=0.5, save_txt=True, conf=0.25,line_thickness = 1)

    for r in results:
        conf_list = r.boxes.conf.numpy().tolist()
        clss_list = r.boxes.cls.numpy().tolist()
        original_list = clss_list
        updated_list = []
        for element in original_list:
            updated_list.append(decoding_of_predictions[int(element)])

        bounding_boxes = r.boxes.xyxy.numpy()
        confidences = conf_list
        class_names = updated_list

        # Draw bounding boxes on the image
        for bbox, conf, cls in zip(bounding_boxes, confidences, class_names):
            x1, y1, x2, y2 = bbox.astype(int)
            box_color = class_colors[cls]  # Get color based on class
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)
            
            # Determine text color based on box color
            text_color = (0, 0, 0) if sum(box_color) > 382.5 else (255, 255, 255)
            
            cv2.putText(img_cv, f'{cls} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Convert the OpenCV image back to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    st.image(img_pil, caption='Output Image', use_column_width=True)

if __name__ == "__main__":
    main()

