import gradio as gr
import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load models
plate_model = YOLO("models/license_plate_detector.pt")
vehicle_model = YOLO("models/yolov8n.pt")
ocr = PaddleOCR(use_angle_cls=False, lang_list=['en'], det=False, show_log=False)

def process_image(image):
    """Xử lý ảnh và hiển thị kết quả."""
    image = cv2.imread(image) if isinstance(image, str) else cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    
    # Nhận diện biển số
    plate_results = plate_model(image)
    for box in plate_results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        plate_img = image[y1:y2, x1:x2]
        ocr_result = ocr.ocr(plate_img, cls=False)
        plate_text = " ".join(res[1][0] for res in ocr_result[0]) if ocr_result and len(ocr_result[0]) > 0 else ""
        if plate_text:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return image[:, :, ::-1]  # Chuyển về RGB

def process_video(video_path):
    """Xử lý video và trả về video đã xử lý."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Không thể mở video!", None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps, (frame_width, frame_height))

    while cap.isOpened():
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

    cap.release()
    out.release()
    return "Xử lý video hoàn tất!", output_path

def handle_input(input_type, file):
    """Xử lý input dựa trên loại file (ảnh hoặc video)."""
    if input_type == "Ảnh":
        return process_image(file), None
    elif input_type == "Video":
        return None, process_video(file)
    return "Chọn ảnh hoặc video!", None

# Giao diện Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 📷 Nhận diện phương tiện & biển số xe 🚗")

    input_type = gr.Radio(["Ảnh", "Video"], label="Chọn loại dữ liệu")
    file_input = gr.File(label="Tải lên ảnh hoặc video")
    process_btn = gr.Button("Bắt đầu xử lý")
    output_img = gr.Image(label="Kết quả ảnh", visible=False)
    output_text = gr.Textbox(label="Trạng thái")
    output_video = gr.Video(label="Kết quả video", visible=False)

    def process_all(input_type, file):
        img_result, vid_result = handle_input(input_type, file)
        return (
            img_result if img_result is not None else gr.update(visible=False),
            "Xử lý hoàn tất!" if vid_result else "Chọn ảnh hoặc video!",
            vid_result if vid_result else gr.update(visible=False)
        )

    process_btn.click(process_all, inputs=[input_type, file_input], outputs=[output_img, output_text, output_video])

demo.launch()
