
from pydantic import BaseModel, Field
from typing import Optional, List

class UserFace(BaseModel):
    uid_face: int
    name: str

class UserListResponse(BaseModel):
    users: List[UserFace]

class VerifyFaceRequest(BaseModel):
    # NOTE: Untuk upload file, FastAPI akan menggunakan UploadFile, tidak perlu didefinisikan di schema.
    pass

class VerifyFaceResponse(BaseModel):
    status: str
    name: Optional[str] = None
    confidence: Optional[float] = None
    uid_face: Optional[int] = None
    spoof: Optional[bool] = None

class RegisterFaceResponse(BaseModel):
    status: str
    name: str
    uid_face: int
    quality_score: Optional[float] = None
    frames_used: Optional[int] = None
    total_frames: Optional[int] = None
