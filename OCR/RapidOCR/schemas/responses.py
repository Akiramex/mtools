from pydantic import BaseModel


class OcrBox(BaseModel):
    polygon: list[list[int]]
    text: str
    score: float


class OcrData(BaseModel):
    text: str
    boxes: list[OcrBox] | None = None
    elapsed_ms: int = 0


class OcrResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: OcrData | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
