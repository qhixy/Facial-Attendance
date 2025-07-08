# app/api/v1/attendance.py

from fastapi import APIRouter, UploadFile, File, Depends, Query
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import cv2
from datetime import datetime

from app.core.detect import FaceDetectionRecognition
from app.core.attendance import load_attendance_data, save_attendance_data, get_today_string
from app.models.attendance import ClockInResponse, ClockOutResponse, AttendanceStatusResponse, AttendanceStatusRecord

def get_recognizer():
    return FaceDetectionRecognition("embeddings.pkl", model_dir="models")

router = APIRouter()

@router.post("/clock-in", response_model=ClockInResponse)
async def clock_in(
    image: UploadFile = File(...),
    recognizer: FaceDetectionRecognition = Depends(get_recognizer)
):
    file_bytes = await image.read()
    np_bytes = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    _, faces_info = recognizer.process_frame(frame)
    if not faces_info:
        return JSONResponse(status_code=400, content={"error": "No face detected"})
    face_info = faces_info[0]
    if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
        return JSONResponse(status_code=400, content={"error": "Face not recognized"})
    if face_info["score"] < recognizer.recognition_threshold:
        return JSONResponse(status_code=400, content={"error": "Face verification failed - possible spoof"})

    attendance_data = load_attendance_data()
    today = get_today_string()
    uid_face = str(face_info["uid_face"])
    if uid_face not in attendance_data:
        attendance_data[uid_face] = {}
    if today in attendance_data[uid_face]:
        if attendance_data[uid_face][today].get("clock_in_time"):
            return JSONResponse(status_code=400, content={
                "error": "Already clocked in today",
                "clock_in_time": attendance_data[uid_face][today]["clock_in_time"]
            })

    current_time = datetime.now()
    attendance_data[uid_face][today] = {
        "name": face_info["name"],
        "clock_in_time": current_time.strftime("%H:%M:%S"),
        "clock_in_datetime": current_time.isoformat(),
        "clock_out_time": None,
        "clock_out_datetime": None,
        "duration": None,
        "status": "clocked_in"
    }
    save_attendance_data(attendance_data)
    return ClockInResponse(
        status="success",
        message="Clock in successful",
        name=face_info["name"],
        uid_face=face_info["uid_face"],
        clock_in_time=current_time.strftime("%H:%M:%S"),
        date=today
    )

@router.post("/clock-out", response_model=ClockOutResponse)
async def clock_out(
    image: UploadFile = File(...),
    recognizer: FaceDetectionRecognition = Depends(get_recognizer)
):
    file_bytes = await image.read()
    np_bytes = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    _, faces_info = recognizer.process_frame(frame)
    if not faces_info:
        return JSONResponse(status_code=400, content={"error": "No face detected"})
    face_info = faces_info[0]
    if face_info["name"] == "Unknown" or face_info["uid_face"] is None:
        return JSONResponse(status_code=400, content={"error": "Face not recognized"})
    if face_info["score"] < recognizer.recognition_threshold:
        return JSONResponse(status_code=400, content={"error": "Face verification failed - possible spoof"})

    attendance_data = load_attendance_data()
    today = get_today_string()
    uid_face = str(face_info["uid_face"])
    if uid_face not in attendance_data or today not in attendance_data[uid_face]:
        return JSONResponse(status_code=400, content={"error": "No clock in record found for today"})
    today_record = attendance_data[uid_face][today]
    if today_record.get("clock_out_time"):
        return JSONResponse(status_code=400, content={
            "error": "Already clocked out today",
            "clock_out_time": today_record["clock_out_time"],
            "duration": today_record["duration"]
        })
    if not today_record.get("clock_in_time"):
        return JSONResponse(status_code=400, content={"error": "Must clock in first"})

    current_time = datetime.now()
    clock_in_datetime = datetime.fromisoformat(today_record["clock_in_datetime"])
    duration_seconds = (current_time - clock_in_datetime).total_seconds()
    duration_hours = duration_seconds / 3600
    duration_formatted = f"{int(duration_hours)}h {int((duration_seconds % 3600) / 60)}m"
    attendance_data[uid_face][today].update({
        "clock_out_time": current_time.strftime("%H:%M:%S"),
        "clock_out_datetime": current_time.isoformat(),
        "duration": duration_formatted,
        "duration_hours": round(duration_hours, 2),
        "status": "completed"
    })
    save_attendance_data(attendance_data)
    return ClockOutResponse(
        status="success",
        message="Clock out successful",
        name=face_info["name"],
        uid_face=face_info["uid_face"],
        clock_in_time=today_record["clock_in_time"],
        clock_out_time=current_time.strftime("%H:%M:%S"),
        duration=duration_formatted,
        duration_hours=round(duration_hours, 2),
        date=today
    )

@router.get("/attendance-status", response_model=AttendanceStatusResponse)
def attendance_status(
    uid_face: Optional[int] = Query(None),
    date_param: Optional[str] = Query(None),
):
    from app.core.attendance import load_attendance_data, get_today_string
    attendance_data = load_attendance_data()
    if not date_param:
        date_param = get_today_string()
    if uid_face:
        uid_face = str(uid_face)
        if uid_face not in attendance_data or date_param not in attendance_data[uid_face]:
            return JSONResponse(content={
                "uid_face": uid_face,
                "date": date_param,
                "status": "not_clocked_in",
                "message": "No attendance record found"
            })
        record = attendance_data[uid_face][date_param]
        attendance = [AttendanceStatusRecord(
            uid_face=int(uid_face),
            name=record["name"],
            date=date_param,
            status=record["status"],
            clock_in_time=record.get("clock_in_time"),
            clock_out_time=record.get("clock_out_time"),
            duration=record.get("duration"),
            duration_hours=record.get("duration_hours")
        )]
        return AttendanceStatusResponse(
            date=date_param,
            total_records=1,
            attendance=attendance
        )
    else:
        all_status = []
        for uid, user_data in attendance_data.items():
            if date_param in user_data:
                record = user_data[date_param]
                all_status.append(AttendanceStatusRecord(
                    uid_face=int(uid),
                    name=record["name"],
                    date=date_param,
                    status=record["status"],
                    clock_in_time=record.get("clock_in_time"),
                    clock_out_time=record.get("clock_out_time"),
                    duration=record.get("duration"),
                    duration_hours=record.get("duration_hours")
                ))
        return AttendanceStatusResponse(
            date=date_param,
            total_records=len(all_status),
            attendance=all_status
        )
