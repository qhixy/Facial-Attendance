# app/main.py

from fastapi import FastAPI
from app.api.v1 import face, attendance

app = FastAPI(title="Facial Attendance FastAPI")

app.include_router(face.router, prefix="/api/v1/face", tags=["Face"])
app.include_router(attendance.router, prefix="/api/v1/attendance", tags=["Attendance"])
