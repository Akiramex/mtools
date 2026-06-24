import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import AppConfig
from models.asr import AsrModel
from models.sv import SvModel
from schemas.responses import (
    AsrData,
    AsrResponse,
    SentenceInfo,
    TaskResultData,
    TaskResultResponse,
    TaskSubmitResponse,
)
from utils.audio import cleanup_temp, extract_segment, save_upload_temp, validate_upload
from utils.task_manager import TaskInfo, TaskManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/asr")

_asr_model: AsrModel | None = None
_sv_model: SvModel | None = None
_config: AppConfig | None = None
_task_manager: TaskManager | None = None


def set_models(asr_model: AsrModel, sv_model: SvModel | None, config: AppConfig, task_manager: TaskManager):
    global _asr_model, _sv_model, _config, _task_manager
    _asr_model = asr_model
    _sv_model = sv_model
    _config = config
    _task_manager = task_manager


def _parse_result(result: dict, spk_name_map: dict | None = None) -> AsrData:
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
                speaker_name=(spk_name_map or {}).get(s.get("spk", 0)),
            )
        )
    return AsrData(text=text, sentence_list=sentence_list)


@router.post("/file", response_model=AsrResponse)
async def asr_file(
    file: UploadFile = File(..., description="Audio file"),
    identify_speakers: str = Form("false"),
):
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

        spk_name_map = await _identify_speakers(temp_path, result, identify_speakers)
        return AsrResponse(data=_parse_result(result, spk_name_map))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ASR inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR inference failed: {e}")
    finally:
        cleanup_temp(temp_path)


async def _identify_speakers(audio_path: str, result: dict, identify_speakers: str) -> dict[int, str]:
    spk_name_map: dict[int, str] = {}
    do_identify = identify_speakers.lower() in ("true", "1", "yes")
    if not do_identify or not _sv_model:
        return spk_name_map

    sentence_info = result.get("sentence_info", [])
    if not sentence_info:
        return spk_name_map

    spk_segments: dict[int, tuple[int, int]] = {}
    for s in sentence_info:
        spk_id = s.get("spk", 0)
        start = s.get("start", 0)
        end = s.get("end", 0)
        if spk_id not in spk_segments:
            spk_segments[spk_id] = (start, end)
        else:
            prev_start, prev_end = spk_segments[spk_id]
            spk_segments[spk_id] = (min(prev_start, start), max(prev_end, end))

    for spk_id, (start_ms, end_ms) in spk_segments.items():
        seg_path = ""
        try:
            seg_path = extract_segment(audio_path, start_ms, end_ms, _config.server.temp_dir)
            name, _ = await _sv_model.identify_speaker(seg_path)
            spk_name_map[spk_id] = name
        except Exception as e:
            logger.warning("Speaker identification failed for spk=%d: %s", spk_id, e)
        finally:
            cleanup_temp(seg_path)

    return spk_name_map


# --- Async task queue endpoints ---

async def _process_task(task: TaskInfo):
    result = await _asr_model.transcribe(task.audio_path)
    if not result:
        task.result = {"text": "", "sentence_info": []}
        return

    spk_name_map = {}
    if task.identify_speakers and _sv_model:
        sentence_info = result.get("sentence_info", [])
        if sentence_info:
            spk_segments: dict[int, tuple[int, int]] = {}
            for s in sentence_info:
                spk_id = s.get("spk", 0)
                start = s.get("start", 0)
                end = s.get("end", 0)
                if spk_id not in spk_segments:
                    spk_segments[spk_id] = (start, end)
                else:
                    prev_start, prev_end = spk_segments[spk_id]
                    spk_segments[spk_id] = (min(prev_start, start), max(prev_end, end))

            for spk_id, (start_ms, end_ms) in spk_segments.items():
                seg_path = ""
                try:
                    seg_path = extract_segment(task.audio_path, start_ms, end_ms, _config.server.temp_dir)
                    name, _ = await _sv_model.identify_speaker(seg_path)
                    spk_name_map[spk_id] = name
                except Exception as e:
                    logger.warning("Speaker identification failed for spk=%d: %s", spk_id, e)
                finally:
                    cleanup_temp(seg_path)

    asr_data = _parse_result(result, spk_name_map)
    task.result = asr_data.model_dump()
    cleanup_temp(task.audio_path)


@router.post("/task", response_model=TaskSubmitResponse)
async def submit_asr_task(
    file: UploadFile = File(..., description="Audio file"),
    identify_speakers: str = Form("false"),
):
    if _asr_model is None or _task_manager is None:
        raise HTTPException(status_code=500, detail="ASR service not initialized")

    content = await file.read()
    validate_upload(file.filename, len(content), _config.upload)
    temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

    do_identify = identify_speakers.lower() in ("true", "1", "yes")
    task_id = await _task_manager.submit(temp_path, do_identify)
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
        finished_at=datetime.fromtimestamp(task.finished_at).isoformat() if task.finished_at else None,
        error=task.error,
    )

    if task.status == "completed" and task.result:
        result_data.result = AsrData(**task.result)
    elif task.status == "failed":
        return TaskResultResponse(code=-1, msg=msg, data=result_data)

    return TaskResultResponse(msg=msg, data=result_data)
