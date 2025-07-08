# app/api/v1/face.py

from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import cv2
import os
from datetime import datetime

from app.core.detect import FaceDetectionRecognition
from app.models.face import UserListResponse, VerifyFaceResponse, RegisterFaceResponse

# Dependency singleton (global instance)
def get_recognizer():
    # Pastikan path sesuai
    return FaceDetectionRecognition("embeddings.pkl", model_dir="models")

router = APIRouter()

@router.get("/get-user-face", response_model=UserListResponse)
def get_user_face(recognizer: FaceDetectionRecognition = Depends(get_recognizer)):
    users = [
        {"uid_face": face_data["uid_face"], "name": face_data["name"]}
        for uid, face_data in recognizer.embeddings_db.items()
    ]
    return {"users": users}

@router.post("/verify-face", response_model=VerifyFaceResponse)
async def verify_face(
    image: UploadFile = File(...),
    recognizer: FaceDetectionRecognition = Depends(get_recognizer)
):
    file_bytes = await image.read()
    np_bytes = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    _, faces_info = recognizer.process_frame(frame)
    if not faces_info:
        return VerifyFaceResponse(status="no face detected")
    face_info = faces_info[0]
    return VerifyFaceResponse(
        status="ok",
        name=face_info["name"],
        confidence=face_info["score"],
        uid_face=face_info["uid_face"],
        spoof=face_info["score"] < recognizer.recognition_threshold
    )

@router.post("/register-face", response_model=RegisterFaceResponse)
async def register_face(
    video: UploadFile = File(...),
    name: str = Form(...),
    uid_face: Optional[int] = Form(None),
    recognizer: FaceDetectionRecognition = Depends(get_recognizer)
):
    # Generate UID jika belum ada
    if not uid_face:
        uid_face = recognizer.generate_uid_face()
        while int(uid_face) in recognizer.embeddings_db:
            uid_face = recognizer.generate_uid_face()
    else:
        uid_face = int(uid_face)
        if uid_face in recognizer.embeddings_db:
            return JSONResponse(status_code=400, content={"error": "UID already exists"})

    # Save video temp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = name.replace(" ", "_")
    save_dir = os.path.join("video", safe_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.mp4")
    with open(save_path, "wb") as f:
        f.write(await video.read())

    # Get best face from video
    best_frame, best_face, quality_score = recognizer.get_best_face_from_video(save_path)
    if best_frame is None:
        return JSONResponse(status_code=400, content={"error": "No face detected in video"})

    embedding = recognizer.get_face_embedding(best_frame, best_face)
    recognizer.embeddings_db[uid_face] = {
        "name": name,
        "embeddings": embedding,
        "uid_face": uid_face
    }
    # Save embeddings
    import pickle
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(recognizer.embeddings_db, f)

    return RegisterFaceResponse(
        status="registered",
        name=name,
        uid_face=uid_face,
        quality_score=float(quality_score)
    )

@router.post("/register-face-average", response_model=RegisterFaceResponse)
async def register_face_average(
    video: UploadFile = File(...),
    name: str = Form(...),
    uid_face: Optional[int] = Form(None),
    recognizer: FaceDetectionRecognition = Depends(get_recognizer)
):
    if not uid_face:
        uid_face = recognizer.generate_uid_face()
        while int(uid_face) in recognizer.embeddings_db:
            uid_face = recognizer.generate_uid_face()
    else:
        uid_face = int(uid_face)
        if uid_face in recognizer.embeddings_db:
            return JSONResponse(status_code=400, content={"error": "UID already exists"})

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    safe_name = name.replace(" ", "_")
    save_dir = os.path.join("video", safe_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.mp4")
    with open(save_path, "wb") as f:
        f.write(await video.read())

    # Extract embeddings from video
    import cv2
    cap = cv2.VideoCapture(save_path)
    embeddings = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = recognizer.detect_faces(frame)
        if faces is not None and len(faces) > 0:
            face = faces[0]
            embedding = recognizer.get_face_embedding(frame, face)
            embeddings.append(embedding)
        frame_count += 1
    cap.release()
    if len(embeddings) == 0:
        return JSONResponse(status_code=400, content={"error": "No face detected in video"})

    mean_embedding = np.mean(embeddings, axis=0)
    recognizer.embeddings_db[uid_face] = {
        "name": name,
        "embeddings": mean_embedding,
        "uid_face": uid_face
    }
    import pickle
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(recognizer.embeddings_db, f)

    return RegisterFaceResponse(
        status="registered",
        name=name,
        uid_face=uid_face,
        frames_used=len(embeddings),
        total_frames=frame_count
    )
