if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('license_plate_detector.pt')
    model.train(data='data.yaml', epochs=200, batch=16, imgsz=640, lr0=0.0001,device=0)
