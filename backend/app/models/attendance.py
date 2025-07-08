from pydantic import BaseModel, Field
from typing import Optional, List

class ClockInResponse(BaseModel):
    status: str
    message: str
    name: str
    uid_face: int
    clock_in_time: str
    date: str

class ClockOutResponse(BaseModel):
    status: str
    message: str
    name: str
    uid_face: int
    clock_in_time: str
    clock_out_time: str
    duration: str
    duration_hours: float
    date: str

class AttendanceStatusRecord(BaseModel):
    uid_face: int
    name: str
    date: str
    status: str
    clock_in_time: Optional[str]
    clock_out_time: Optional[str]
    duration: Optional[str]
    duration_hours: Optional[float]

class AttendanceStatusResponse(BaseModel):
    date: str
    total_records: int
    attendance: List[AttendanceStatusRecord]
