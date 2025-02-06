import cv2
import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone
from app.utils import (
    predict_number_plate, 
    send_violation_email, 
    is_valid_indian_number_plate, 
    check_daily_violation, 
    save_violation_image
)
from app.db import log_violation

# Constants and Global Variables
device = torch.device("cpu")
classNames = ["with helmet", "without helmet", "rider", "number plate"]
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("app/models/yolov8_best.pt")

def process_frame(img):
    """Processes a single frame for helmet detection and number plate extraction."""
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(new_img, stream=True, device="mps")
    li = dict()
    rider_box = []

    for r in results:
        boxes = r.boxes
        xy = boxes.xyxy
        confidences = boxes.conf
        classes = boxes.cls
        new_boxes = torch.cat((xy.to(device), confidences.unsqueeze(1).to(device), classes.unsqueeze(1).to(device)), 1)
        new_boxes = new_boxes[new_boxes[:, -1].sort()[1]]

        try:
            indices = torch.where(new_boxes[:, -1] == 2)  # Rider detection
            rows = new_boxes[indices]
            for box in rows:
                x1, y1, x2, y2 = map(int, box[:4])
                rider_box.append((x1, y1, x2, y2))
        except:
            pass

        for i, box in enumerate(new_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            conf = round(float(box[4]) * 100) / 100
            cls = int(box[5])

            if classNames[cls] in ["without helmet", "rider", "number plate"] and conf >= 0.5:
                if classNames[cls] == "rider":
                    rider_box.append((x1, y1, x2, y2))

                for j, rider in enumerate(rider_box):
                    if x1 + 10 >= rider[0] and y1 + 10 >= rider[1] and x2 <= rider[2] and y2 <= rider[3]:
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
                        cvzone.putTextRect(img, f"{classNames[cls].upper()}", (x1 + 10, y1 - 10), scale=1.5,
                                           offset=10, thickness=2, colorT=(39, 40, 41), colorR=(248, 222, 34))
                        li.setdefault(f"rider{j}", []).append(classNames[cls])

                        if classNames[cls] == "number plate":
                            crop = img[y1:y1 + h, x1:x1 + w]
                            if len(set(li[f"rider{j}"])) == 3:
                                try:
                                    vehicle_number, conf = predict_number_plate(crop, ocr)
                                    if vehicle_number and conf:
                                        cvzone.putTextRect(img, f"{vehicle_number} {round(conf * 100, 2)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                        
                                        without_helmet_detected = any(
                                            'without helmet' in rider_classes for rider_classes in li.values())
                                        
                                        if (without_helmet_detected and 
                                            is_valid_indian_number_plate(vehicle_number) and 
                                            not check_daily_violation(vehicle_number)):
                                            
                                            # Save violation image
                                            violation_image = save_violation_image(img, vehicle_number)
                                            
                                            # Log violation
                                            log_violation(vehicle_number, "Without Helmet", violation_image)
                                            
                                            # Send email
                                            send_violation_email(vehicle_number, violation_image)
                                except Exception as e:
                                    print(e)
    return img

def process_video(video_path):
    """Detects helmet violations from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame)
        cv2.imshow("Helmet Detection - Video", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()     