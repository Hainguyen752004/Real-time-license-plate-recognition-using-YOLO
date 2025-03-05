import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict

def process_video(video_path, plate_model_path, vehicle_model_path, display_placeholder, stop_processing):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không thể mở video!")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))
    
    plate_model = YOLO(plate_model_path)
    vehicle_model = YOLO(vehicle_model_path)
    ocr = PaddleOCR(use_angle_cls=False, lang_list=['en'], det=False, show_log=False)
    
    frame_count = 0
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        if stop_processing():
            break
        
        success, frame = cap.read()
        if not success:
            break
        
        vehicle_results = vehicle_model.track(frame, persist=True)
        plate_results = plate_model.track(frame, persist=True)
        
        if vehicle_results[0].boxes.id is not None:
            for box, cls, confidence in zip(
                vehicle_results[0].boxes.xyxy.cpu().numpy(),
                vehicle_results[0].boxes.cls.cpu().numpy().astype(int),
                vehicle_results[0].boxes.conf.cpu().numpy()
            ):
                if confidence < 0.8:
                    continue
                x1, y1, x2, y2 = map(int, box)
                class_name = vehicle_model.names[cls]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        if plate_results[0].boxes.id is not None:
            for box in plate_results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                plate_img = frame[y1:y2, x1:x2]
                ocr_result = ocr.ocr(plate_img, cls=False)
                plate_text = " ".join(res[1][0] for res in ocr_result[0]) if ocr_result and len(ocr_result[0]) > 0 else ""
                if plate_text:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    return output_path

st.title("Nhận diện phương tiện và biển số xe với YOLO và PaddleOCR")

uploaded_video = st.file_uploader("Chọn video để xử lý", type=["mp4", "avi", "mov"])
plate_model_path = "models/license_plate_detector.pt"
vehicle_model_path = "models/yolov8n.pt"

stop_flag = False

def stop_processing():
    return stop_flag

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name
    
    st.video(video_path)
    
    if st.button("Bắt đầu xử lý video"):
        st.write("Đang xử lý video...")
        display_placeholder = st.empty()
        output_video_path = process_video(video_path, plate_model_path, vehicle_model_path, display_placeholder, stop_processing)
        
        if output_video_path:
            st.success("Xử lý hoàn tất!")
            st.video(output_video_path)
    
    if st.button("Dừng xử lý"):
        stop_flag = True
