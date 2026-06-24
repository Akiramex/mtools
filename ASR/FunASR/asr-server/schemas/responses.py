from datetime import timedelta

from pydantic import BaseModel


def to_srt_time(milliseconds: int) -> str:
    t = timedelta(milliseconds=milliseconds)
    return f"{t.seconds // 3600:02d}:{(t.seconds // 60) % 60:02d}:{t.seconds % 60:02d}.{t.microseconds // 1000:03d}"


class SentenceInfo(BaseModel):
    index: int
    text: str
    start: int
    end: int
    speaker: int
    speaker_name: str | None = None


class AsrData(BaseModel):
    text: str
    sentence_list: list[SentenceInfo]


class AsrResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: AsrData | None = None


class SerData(BaseModel):
    emotion: str
    scores: dict[str, float]


class SerResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: SerData | None = None


class SpeakerInfo(BaseModel):
    name: str


class SpeakerListResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: list[SpeakerInfo]


class SpeakerRegisterResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: dict


class HealthResponse(BaseModel):
    status: str
    asr_model_loaded: bool
    sv_model_loaded: bool
    ser_model_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    code: int = -1
    msg: str
    data: dict | None = None


class TaskSubmitResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: dict


class TaskResultData(BaseModel):
    task_id: str
    status: str
    created_at: str
    finished_at: str | None = None
    error: str | None = None
    result: AsrData | None = None


class TaskResultResponse(BaseModel):
    code: int = 0
    msg: str = ""
    data: TaskResultData | None = None
