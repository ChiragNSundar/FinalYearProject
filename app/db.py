import os
import csv
import json
from datetime import datetime, date

VIOLATIONS_FILE = "violations.json"

def ensure_violations_file_exists():
    if not os.path.exists(VIOLATIONS_FILE):
        with open(VIOLATIONS_FILE, 'w') as file:
            json.dump([], file)

def log_violation(vehicle_number, violation_type, violation_image_path):
    ensure_violations_file_exists()
    
    # Read existing violations
    with open(VIOLATIONS_FILE, 'r') as file:
        try:
            violations = json.load(file)
        except json.JSONDecodeError:
            violations = []
    
    # Check for existing violation today
    today = datetime.now().date()
    existing_violation = next((v for v in violations 
                                if v['vehicle_number'] == vehicle_number 
                                and datetime.strptime(v['timestamp'], "%Y-%m-%d %H:%M:%S").date() == today), 
                               None)
    
    # If no violation exists today, add new violation
    if not existing_violation:
        violation = {
            'vehicle_number': vehicle_number,
            'violation_type': violation_type,
            'violation_image': violation_image_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        violations.append(violation)
        
        # Write back to file
        with open(VIOLATIONS_FILE, 'w') as file:
            json.dump(violations, file, indent=2)
        
        return violation
    
    return None

def read_violations_by_vehicle(vehicle_number):
    ensure_violations_file_exists()
    
    with open(VIOLATIONS_FILE, 'r') as file:
        try:
            violations = json.load(file)
        except json.JSONDecodeError:
            violations = []
    
    return [v for v in violations if v['vehicle_number'] == vehicle_number]