from ultralytics import YOLO

# yolo model creation
model = YOLO("/Users/chiragnsundar/Documents/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/yolo-weights/yolov8l.pt")
model.train(data="coco128.yaml", imgsz=320, batch=4, epochs=20, workers=0)