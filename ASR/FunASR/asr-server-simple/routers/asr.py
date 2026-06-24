import logging
from datetime import datetime

from config import AppConfig
from fastapi import APIRouter, File, HTTPException, UploadFile
from models.asr import AsrModel
from schemas.responses import (
    AsrData,
    AsrResponse,
    SentenceInfo,
    TaskResultData,
    TaskResultHuiyanResponse,
    TaskResultResponse,
    TaskSubmitHuiyanResponse,
    TaskSubmitResponse,
)
from utils.audio import cleanup_temp, save_upload_temp, validate_upload
from utils.task_manager import TaskInfo, TaskManager

logger = logging.getLogger(__name__)
router = APIRouter()

_asr_model: AsrModel | None = None
_config: AppConfig | None = None
_task_manager: TaskManager | None = None


def set_models(asr_model: AsrModel, config: AppConfig, task_manager: TaskManager):
    global _asr_model, _config, _task_manager
    _asr_model = asr_model
    _config = config
    _task_manager = task_manager


def _ms_to_timestamp(ms: int) -> str:
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _parse_result(result: dict) -> AsrData:
    text = result.get("text", "")
    sentence_info = result.get("sentence_info", [])
    sentence_list = []
    for i, s in enumerate(sentence_info):
        sentence_list.append(
            SentenceInfo(
                index=i + 1,
                text=s.get("text", ""),
                start=s.get("start", 0),
                end=s.get("end", 0),
                speaker=s.get("spk", 0),
            )
        )
    return AsrData(text=text, sentence_list=sentence_list)


@router.post("/file", response_model=AsrResponse)
async def asr_file(file: UploadFile = File(..., description="Audio file")):
    if _asr_model is None:
        raise HTTPException(status_code=500, detail="ASR model not initialized")

    temp_path = ""
    try:
        content = await file.read()
        validate_upload(file.filename, len(content), _config.upload)
        temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

        result = await _asr_model.transcribe(temp_path)
        if not result:
            return AsrResponse(data=AsrData(text="", sentence_list=[]))

        return AsrResponse(data=_parse_result(result))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ASR inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR inference failed: {e}")
    finally:
        cleanup_temp(temp_path)


async def _process_task(task: TaskInfo):
    try:
        result = await _asr_model.transcribe(task.audio_path)
        if not result:
            task.result = AsrData(text="", sentence_list=[]).model_dump()
            return

        asr_data = _parse_result(result)
        task.result = asr_data.model_dump()
    finally:
        cleanup_temp(task.audio_path)


@router.post("/task", response_model=TaskSubmitResponse)
async def submit_asr_task(file: UploadFile = File(..., description="Audio file")):
    if _asr_model is None or _task_manager is None:
        raise HTTPException(status_code=500, detail="ASR service not initialized")

    content = await file.read()
    validate_upload(file.filename, len(content), _config.upload)
    temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

    task_id = await _task_manager.submit(temp_path)
    return TaskSubmitResponse(data={"task_id": task_id})


@router.get("/task/{task_id}", response_model=TaskResultResponse)
async def get_asr_task(task_id: str):
    if _task_manager is None:
        raise HTTPException(status_code=500, detail="ASR service not initialized")

    task = _task_manager.get_task(task_id)
    if task is None:
        return TaskResultResponse(code=-1, msg="task not found", data=None)

    status_map = {
        "pending": "pending",
        "processing": "processing",
        "completed": "success",
        "failed": "failed",
    }
    msg = status_map.get(task.status, task.status)

    result_data = TaskResultData(
        task_id=task.task_id,
        status=task.status,
        created_at=datetime.fromtimestamp(task.created_at).isoformat(),
        finished_at=datetime.fromtimestamp(task.finished_at).isoformat()
        if task.finished_at
        else None,
        error=task.error,
    )

    if task.status == "completed" and task.result:
        result_data.result = AsrData(**task.result)
    elif task.status == "failed":
        return TaskResultResponse(code=-1, msg=msg, data=result_data)

    return TaskResultResponse(msg=msg, data=result_data)


@router.post("/request", response_model=TaskSubmitHuiyanResponse)
async def submit_asr_task_huiyan(
    file: UploadFile = File(..., description="Audio file"),
):
    if _asr_model is None or _task_manager is None:
        return TaskSubmitHuiyanResponse(
            status="20326", message="ASR task_manager 服务未初始化成功，请联系管理员"
        )

    content = await file.read()
    validate_upload(file.filename, len(content), _config.upload)
    temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

    task_id = await _task_manager.submit(temp_path)
    return TaskSubmitHuiyanResponse(
        status="00000", data={"task_id": task_id, "duration": 0}
    )


@router.get("/getResult", response_model=TaskResultHuiyanResponse)
async def get_asr_task_huiyan(task_id: str):
    if _task_manager is None:
        return TaskResultHuiyanResponse(
            status="20326", message="ASR task_manager 服务未初始化成功，请联系管理员"
        )

    task = _task_manager.get_task(task_id)
    if task is None:
        return TaskResultResponse(code=-1, msg="task not found", data=None)

    if task.status == "completed" and task.result:
        sentence_list = task.result.get("sentence_list", [])
        data = []
        for i, s in enumerate(sentence_list):
            data.append({
                "begin": _ms_to_timestamp(s.get("start", 0)),
                "cluster_id": s.get("speaker", 0),
                "confidence": 1,
                "end": _ms_to_timestamp(s.get("end", 0)),
                "lang_type": "",
                "paragraph": i + 1,
                "seg_num": i + 1,
                "transcript": s.get("text", ""),
                "volume": 100,
            })
        return TaskResultHuiyanResponse(
            status="00000",
            message="success",
            data=data,
            statistics={},
        )
    elif task.status == "pending":
        return TaskResultHuiyanResponse(
            status="20320",
            message="",
            data={
                "desc": "等待中",
                "file_name": "",
                "insert_time": "",
                "progress": "1",
                "duration": "",
            },
        )
    elif task.status == "processing":
        return TaskResultHuiyanResponse(
            status="20320",
            message="",
            data={
                "desc": "等待中",
                "file_name": "",
                "insert_time": "",
                "progress": "100",
                "duration": "",
            },
        )
    else:
        return TaskResultHuiyanResponse(
            status="20326",
            message="",
            data={
                "desc": "失败",
                "file_name": "",
                "insert_time": "",
                "progress": "",
                "duration": "",
            },
        )
