import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email import encoders
import datetime
import cv2
from app.db import read_violations_by_vehicle
from email.mime.base import MIMEBase

def is_valid_indian_number_plate(vehicle_number):
    """Validates Indian number plate format."""
    pattern = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'
    return re.match(pattern, vehicle_number) is not None

def check_daily_violation(vehicle_number):
    """Check if violation for this vehicle has already been logged today."""
    try:
        violations = read_violations_by_vehicle(vehicle_number)
        today = datetime.date.today()
        daily_violations = [v for v in violations if datetime.datetime.strptime(v['timestamp'], "%Y-%m-%d %H:%M:%S").date() == today]
        return len(daily_violations) > 0
    except Exception as e:
        print(f"Error checking daily violations: {e}")
        return False

def save_violation_image(frame, vehicle_number):
    """Save violation image with timestamp."""
    violation_dir = "violation_images"
    os.makedirs(violation_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{violation_dir}/{vehicle_number}_{timestamp}.jpg"
    
    cv2.imwrite(filename, frame)
    return filename

def predict_number_plate(img, ocr):
    """Predict and clean vehicle number plate with enhanced filtering."""
    try:
        result = ocr.ocr(img, cls=True)
        result = result[0]
        texts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        cleaned_text = ''.join(re.findall(r'[A-Z0-9]', texts[0].upper()))
        
        if (scores[0]*100) >= 90 and len(cleaned_text) == 10:
            return cleaned_text, scores[0]
        
        return None, None
    
    except Exception as e:
        print(f"Number plate prediction error: {e}")
        return None, None

def send_violation_email(vehicle_number, violation_image):
    """Send email notification for helmet violation."""
    sender_email = "sender-email"
    sender_password = ""  # Use App Password for Gmail
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = "receiver-email"
    msg['Subject'] = "Helmet Violation Detected"
    
    body = f"""
    Helmet Violation Detected!
    
    Vehicle Number: {vehicle_number}
    Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Violation Type: No Helmet
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with open(violation_image, 'rb') as file:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(file.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(violation_image)}"')
        msg.attach(attachment)
    except Exception as e:
        print(f"Error attaching image: {e}")
        return

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Violation email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")