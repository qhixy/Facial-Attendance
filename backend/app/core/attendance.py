import os
import json
from datetime import datetime

ATTENDANCE_FILE = "attendance.json"

def load_attendance_data():
    if not os.path.exists(ATTENDANCE_FILE):
        return {}
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_attendance_data(data):
    with open(ATTENDANCE_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def get_today_string():
    return datetime.now().strftime("%Y-%m-%d")
