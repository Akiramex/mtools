import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import AppConfig
from models.ser import SerModel
from schemas.responses import SerData, SerResponse
from utils.audio import cleanup_temp, save_upload_temp, validate_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ser")

_ser_model: SerModel | None = None
_config: AppConfig | None = None


def set_model(ser_model: SerModel | None, config: AppConfig):
    global _ser_model, _config
    _ser_model = ser_model
    _config = config


@router.post("/file", response_model=SerResponse)
async def ser_file(file: UploadFile = File(..., description="Audio file")):
    if _ser_model is None:
        raise HTTPException(status_code=500, detail="SER model not initialized or disabled")

    temp_path = ""
    try:
        content = await file.read()
        validate_upload(file.filename, len(content), _config.upload)
        temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

        result = await _ser_model.detect_emotion(temp_path)
        return SerResponse(data=SerData(emotion=result["emotion"], scores=result["scores"]))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("SER inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"SER inference failed: {e}")
    finally:
        cleanup_temp(temp_path)
