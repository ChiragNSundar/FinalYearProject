import cv2
import torch
from datetime import datetime
import csv
import re
from paddleocr import PaddleOCR
from ultralytics import YOLO
import cvzone
import math
from image_to_text import predict_number_plate

# Constants and Global Variables
device = torch.device("cpu")  # Change to "cuda" for GPU support
classNames = ["with helmet", "without helmet", "rider", "number plate"]
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Initialize OCR

# YOLO Model
model = YOLO("/Users/chiragnsundar/Documents/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/runs/detect/train7/weights/best.pt")  # Replace with the actual path to the YOLO model

# Helper Functions
def is_valid_indian_number_plate(number_plate):
    """Validates Indian number plate format."""
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, number_plate) is not None


def extract_and_store_number_plate(vehicle_number, conf, without_helmet_detected, csv_file_path='number_plates.csv'):
    """Extracts and stores valid number plate details."""
    if vehicle_number and conf and without_helmet_detected and is_valid_indian_number_plate(vehicle_number):
        existing_entries = set()
        try:
            with open(csv_file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                existing_entries = {row[0] for row in reader}
        except FileNotFoundError:
            pass

        if vehicle_number not in existing_entries:
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # Add header if file is empty
                    writer.writerow(['Vehicle Number', 'Confidence', 'Timestamp', 'Helmet Violation'])
                writer.writerow([vehicle_number, round(conf * 100, 2),
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "Yes"])
            print(f"New entry added: {vehicle_number}")
            return True
        else:
            print(f"Duplicate entry not added: {vehicle_number}")
    return False


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
                                    vechicle_number, conf = predict_number_plate(crop, ocr)
                                    if vechicle_number and conf:
                                        cvzone.putTextRect(img, f"{vechicle_number} {round(conf * 100, 2)}%",
                                                           (x1, y1 - 50), scale=1.5, offset=10,
                                                           thickness=2, colorT=(39, 40, 41), colorR=(105, 255, 255))
                                        without_helmet_detected = any(
                                            'without helmet' in rider_classes for rider_classes in li.values())
                                        extract_and_store_number_plate(vechicle_number, conf, without_helmet_detected)
                                except Exception as e:
                                    print(e)
    return img


def detect_realtime_camera():
    """Detects helmet violations in real-time using webcam."""
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = process_frame(frame)
        cv2.imshow("Helmet Detection - Real-Time", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_from_video(video_path):
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


if __name__ == "__main__":
    choice = input("Enter '1' for real-time camera detection or '2' to process a video file: ")
    if choice == '1':
        detect_realtime_camera()
    elif choice == '2':
        video_path = "/Users/chiragnsundar/Documents/Real-Time-Detection-of-Helmet-Violations-and-Capturing-Bike-Numbers-from-Number-Plates-main/videos/22.mp4"
        detect_from_video(video_path)
    else:
        print("Invalid input. Exiting program.")