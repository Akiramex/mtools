from pydantic import BaseModel


class SentenceInfo(BaseModel):
    index: int
    text: str
    start: int
    end: int
    speaker: int


class AsrData(BaseModel):
    text: str
    sentence_list: list[SentenceInfo]


class AsrResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: AsrData | None = None


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


class TaskSubmitHuiyanResponse(BaseModel):
    status: str = "00000"
    message: str = ""
    data: dict | None = None


class TaskResultHuiyanResponse(BaseModel):
    status: str = "00000"
    message: str = ""
    data: dict | list | None = None
    statistics: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
