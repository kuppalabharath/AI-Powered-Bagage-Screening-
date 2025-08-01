import streamlit as st
from ultralytics import YOLO
import cv2
import os
import pyttsx3

st.title("AI-Powered Baggage Scanner")

model_path = "runs/detect/train10/weights/best.onnx"
model = YOLO(model_path, task="detect")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = cv2.imread(image_path)
    results = model.predict(image_path, imgsz=320, conf=0.05)

    classes = ["portable_charger_1", "portable_charger_2", "water", "laptop", "mobile_phone", "tablet", "cosmetic", "nonmetallic_lighter"]
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128), (0, 255, 255), (255, 0, 255), (255, 165, 0), (255, 192, 203)]

    detected_items = []
    for result in results:
        boxes = result.boxes.xywh.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            center_x, center_y, w, h = boxes[i]
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            w = int(w)
            h = int(h)
            class_id = class_ids[i]
            confidence = confidences[i]
            color = colors[class_id]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]} {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_x = x + (w - label_width) // 2
            label_y = y - 10 if y - 10 > 10 else y + 10
            cv2.rectangle(image, (label_x, label_y - label_height), (label_x + label_width, label_y), (255, 255, 255), -1)
            cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            detected_items.append(classes[class_id])

    annotated_image_path = "annotated_image.jpg"
    cv2.imwrite(annotated_image_path, image)

    st.image(annotated_image_path, caption="Annotated Image", use_column_width=True)

    st.subheader("Detected Items:")
    if detected_items:
        for item in set(detected_items):
            st.write(f"- {item}")
    else:
        st.write("No items detected.")

    engine = pyttsx3.init()
    if detected_items:
        items_to_announce = ", ".join(set(detected_items))
        engine.say(f"Items detected: {items_to_announce}")
    else:
        engine.say("No items detected")
    engine.runAndWait()

    os.remove(image_path)
    os.remove(annotated_image_path)
