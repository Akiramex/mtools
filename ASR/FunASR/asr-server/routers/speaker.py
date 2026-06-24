import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import AppConfig
from models.sv import SvModel
from schemas.responses import (
    ErrorResponse,
    SpeakerInfo,
    SpeakerListResponse,
    SpeakerRegisterResponse,
)
from utils.audio import cleanup_temp, save_upload_temp

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/speaker")

_sv_model: SvModel | None = None
_config: AppConfig | None = None


def set_model(sv_model: SvModel, config: AppConfig):
    global _sv_model, _config
    _sv_model = sv_model
    _config = config


@router.post("/register", response_model=SpeakerRegisterResponse)
async def register_speaker(
    file: UploadFile = File(..., description="Speaker audio file (WAV recommended)"),
    name: str = Form(..., description="Speaker name"),
):
    if _sv_model is None:
        raise HTTPException(status_code=500, detail="SV model not initialized")

    if not name.strip():
        raise HTTPException(status_code=400, detail="Speaker name cannot be empty")

    temp_path = ""
    try:
        content = await file.read()
        validate_upload(file.filename, len(content), _config.upload)
        temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

        await _sv_model.register_speaker(temp_path, name.strip())
        return SpeakerRegisterResponse(data={"name": name.strip(), "message": "registered"})

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Speaker registration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    finally:
        cleanup_temp(temp_path)


@router.get("/list", response_model=SpeakerListResponse)
async def list_speakers():
    if _sv_model is None:
        raise HTTPException(status_code=500, detail="SV model not initialized")

    names = _sv_model.list_speakers()
    return SpeakerListResponse(data=[SpeakerInfo(name=n) for n in names])


@router.delete("/{name}")
async def delete_speaker(name: str):
    if _sv_model is None:
        raise HTTPException(status_code=500, detail="SV model not initialized")

    if _sv_model.delete_speaker(name):
        return {"code": 0, "msg": "success", "data": {"name": name}}
    raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")


def validate_upload(filename: str, content_length: int, upload_config):
    from utils.audio import validate_upload as _validate

    _validate(filename, content_length, upload_config)
