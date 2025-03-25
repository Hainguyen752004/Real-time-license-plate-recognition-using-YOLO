import os
import cv2
import json
import torch
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict
from PIL import Image
import numpy as np

def process_video_realtime(video_path, plate_model_path, vehicle_model_path, output_video_path, output_json_path):
    # Kiểm tra và sử dụng GPU nếu có
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng thiết bị: {device}")
    
    # Khởi tạo models
    plate_model = YOLO(plate_model_path)
    vehicle_model = YOLO(vehicle_model_path)
    ocr = PaddleOCR(use_angle_cls=False, lang="en", rec_algorithm="CRNN", show_log=False, use_gpu=(device == 'cuda'))
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    # Lấy thông số video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Khởi tạo video writer để lưu video đầu ra
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))
    
    # Dictionary lưu thông tin tracking
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0

    # Hiển thị video trong thời gian thực với Streamlit
    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Phát hiện phương tiện
        vehicle_results = vehicle_model.track(frame, persist=True)
        # Phát hiện biển số xe
        plate_results = plate_model.track(frame, persist=True)

        # Xử lý kết quả vehicle detection
        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)
            classes = vehicle_results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = vehicle_results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                if confidence < 0.7:
                    continue

                class_name = vehicle_model.names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue

                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / fps,
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                        'Plate Number': ""  # Thêm trường biển số
                    }
                else:
                    objects[class_name]['tracks'][track_id].update({
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    })

                # Vẽ vehicle detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = class_name.capitalize()  
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Xử lý kết quả plate detection và nhận diện biển số với OCR
        if plate_results[0].boxes.id is not None:
            plate_boxes = plate_results[0].boxes.xyxy.cpu().numpy()

            for box in plate_boxes:
                x1, y1, x2, y2 = map(int, box)
                plate_img = frame[y1:y2, x1:x2]  # Cắt vùng chứa biển số xe

                # OCR để nhận diện ký tự từ biển số xe
                ocr_result = ocr.ocr(plate_img, cls=False)
                plate_text = ""
                if ocr_result and len(ocr_result[0]) > 0:
                    for res in ocr_result[0]:
                        plate_text += res[1][0] + " "
                    plate_text = plate_text.strip()

                if plate_text:
                    for class_name, data in objects.items():
                        for track_id, info in data['tracks'].items():
                            box_vehicle = info['bounding_box']
                            if x1 >= box_vehicle[0] and y1 >= box_vehicle[1] and x2 <= box_vehicle[2] and y2 <= box_vehicle[3]:
                                info['Plate Number'] = plate_text

                    # Vẽ biển số và kết quả OCR lên frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Plate: {plate_text}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Chuyển đổi frame thành ảnh để hiển thị trong Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # Ghi video output
        out.write(frame)
        frame_count += 1

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Đã lưu video đã xử lý tại: {output_video_path}")

    # Lưu kết quả vào JSON
    json_results = []
    for class_name, data in objects.items():
        for track_id, info in data['tracks'].items():
            bounding_box = info.get('bounding_box', [0, 0, 0, 0])

            time_appeared = f"{int(info['Time_of_appearance'] // 60):02d}:{int(info['Time_of_appearance'] % 60):02d}"
            time_disappeared = f"{int(info['Time_of_disappearance'] // 60):02d}:{int(info['Time_of_disappearance'] % 60):02d}"

            json_results.append({
                "Object ID": info.get('Object ID', f"Plate {track_id}"),
                "Class": class_name,
                "Time appeared": time_appeared,
                "Time disappeared": time_disappeared,
                "Bounding box": bounding_box,  
                "Plate Number": info.get('Plate Number', ""),  
            })

    # Lưu kết quả JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu kết quả tracking tại: {output_json_path}")
st.markdown("""
    <style>
        /* Toàn bộ giao diện */
        body {
            background-color: #f4f4f9;
            color: #333;
            font-family: 'Arial', sans-serif;
        }

        /* Tên tiêu đề chính */
        .css-1q8dd3e {
            text-align: center;
            color: #4CAF50;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 20px;
        }

        /* Cải tiến phần cài đặt đầu vào */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 1.1em;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #45a049;
        }

        /* Style cho khung upload video */
        .stFileUploader {
            margin-top: 20px;
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }

        /* Cải tiến phần hiển thị ảnh */
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Tiêu đề phần "Processing completed" */
        .css-1dp5e43 {
            color: #ff5722;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
        }

    </style>
""", unsafe_allow_html=True)

# Streamlit UI
def main():
    st.title("Vehicle and License Plate Detection")
    
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        video_path = f"./temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        plate_model_path = st.text_input("Enter plate model path", "models/custom11_plate.pt")
        vehicle_model_path = st.text_input("Enter vehicle model path", "models/yolov8n.pt")
        
        output_dir = st.text_input("Enter output directory", "./output")
        
        base_name = os.path.splitext(uploaded_video.name)[0]
        output_video = os.path.join(output_dir, f"{base_name}_Out_plate.mp4")
        output_json = os.path.join(output_dir, f"{base_name}_Out_plate.json")
        
        if st.button("Start Processing"):
            process_video_realtime(video_path, plate_model_path, vehicle_model_path, output_video, output_json)
            st.success(f"Processing completed! Video and JSON saved to {output_dir}")

if __name__ == "__main__":
    main()
