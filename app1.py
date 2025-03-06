import streamlit as st
import cv2
import os
import tempfile
import numpy as np
import json
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Tạo thư mục output nếu chưa có
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
plate_model = YOLO("models/license_plate_detector.pt")
vehicle_model = YOLO("models/yolov8n.pt")
ocr = PaddleOCR(use_angle_cls=False, lang_list=['en'], det=False, show_log=False)

def process_image(image):
    """Xử lý ảnh và hiển thị kết quả."""
    image = cv2.imread(image) if isinstance(image, str) else cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    
    # Nhận diện biển số
    plate_results = plate_model(image)
    detections = []
    
    for box in plate_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        plate_img = image[y1:y2, x1:x2]
        ocr_result = ocr.ocr(plate_img, cls=False)
        plate_text = " ".join(res[1][0] for res in ocr_result[0]) if ocr_result and len(ocr_result[0]) > 0 else ""
        
        if plate_text:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            detections.append({"plate_text": plate_text, "coordinates": [x1, y1, x2, y2]})
    
    image_path = os.path.join(OUTPUT_DIR, "processed_image.jpg")
    cv2.imwrite(image_path, image)
    json_path = os.path.join(OUTPUT_DIR, "image_detections.json")
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=4)
    
    return image[:, :, ::-1], image_path, json_path  # Chuyển về RGB

def process_video(video_path):
    """Xử lý video và lưu file kết quả."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Không thể mở video!", None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_video_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))
    
    detections = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        vehicle_results = vehicle_model.track(frame, persist=True)
        plate_results = plate_model.track(frame, persist=True)

        if plate_results[0].boxes.id is not None:
            for box in plate_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                plate_img = frame[y1:y2, x1:x2]
                ocr_result = ocr.ocr(plate_img, cls=False)
                plate_text = " ".join(res[1][0] for res in ocr_result[0]) if ocr_result and len(ocr_result[0]) > 0 else ""
                
                if plate_text:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    detections.append({"plate_text": plate_text, "coordinates": [x1, y1, x2, y2]})
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    json_path = os.path.join(OUTPUT_DIR, "video_detections.json")
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=4)
    
    return "Xử lý video hoàn tất!", output_video_path, json_path

st.title("📷 Nhận diện phương tiện & biển số xe 🚗")
input_type = st.radio("Chọn loại dữ liệu", ["Ảnh", "Video"])
uploaded_file = st.file_uploader("Tải lên ảnh hoặc video")

if uploaded_file:
    if input_type == "Ảnh":
        st.write("Đang xử lý ảnh...")
        image_result, image_path, json_path = process_image(uploaded_file)
        st.image(image_result, caption="Ảnh đã xử lý", use_column_width=True)
        st.download_button("📥 Tải ảnh xuống", data=open(image_path, "rb").read(), file_name="processed_image.jpg", mime="image/jpeg")
        st.download_button("📥 Tải JSON xuống", data=open(json_path, "rb").read(), file_name="image_detections.json", mime="application/json")
    
    elif input_type == "Video":
        st.write("Đang xử lý video...")
        status, video_path, json_path = process_video(uploaded_file.name)
        st.success(status)
        st.video(video_path)
        st.download_button("📥 Tải video xuống", data=open(video_path, "rb").read(), file_name="processed_video.mp4", mime="video/mp4")
        st.download_button("📥 Tải JSON xuống", data=open(json_path, "rb").read(), file_name="video_detections.json", mime="application/json")
