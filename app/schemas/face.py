from pydantic import BaseModel
from typing import Optional, List


class VerifyResponseData(BaseModel):
    faceDetected: bool
    faceCount: int
    qualityOk: bool
    qualityIssues: List[str] = []
    isLive: Optional[bool] = None
    livenessConfidence: Optional[float] = None
    similarity: Optional[float] = None
    verified: Optional[bool] = None


class EnrollResponseData(BaseModel):
    faceDetected: bool
    faceCount: int
    qualityOk: bool
    qualityIssues: List[str] = []
    isLive: Optional[bool] = None
    livenessConfidence: Optional[float] = None
    embedding: Optional[List[float]] = None


class VerifyResponseMeta(BaseModel):
    model: str
    detector: str
    processingTimeMs: float


class VerifyResponse(BaseModel):
    success: bool
    data: VerifyResponseData
    meta: VerifyResponseMeta
    error: Optional[str] = None


class EnrollResponse(BaseModel):
    success: bool
    data: EnrollResponseData
    meta: VerifyResponseMeta
    error: Optional[str] = None
