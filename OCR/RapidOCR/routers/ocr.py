import logging

from config import AppConfig
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from models.ocr import OcrModel
from schemas.responses import OcrData, OcrResponse
from utils.image import cleanup_temp, save_upload_temp, validate_upload

logger = logging.getLogger(__name__)
router = APIRouter()

_ocr_model: OcrModel | None = None
_config: AppConfig | None = None


def set_model(ocr_model: OcrModel, config: AppConfig):
    global _ocr_model, _config
    _ocr_model = ocr_model
    _config = config


@router.post("/ocr", response_model=OcrResponse)
async def ocr_file(
    file: UploadFile = File(..., description="图片文件"),
    detail: str = Form("true", description="true 返回框坐标；false 仅返回纯文本"),
):
    if _ocr_model is None:
        raise HTTPException(status_code=500, detail="OCR 模型未初始化")
    if _config is None:
        raise HTTPException(status_code=500, detail="配置未初始化")

    temp_path = ""
    try:
        content = await file.read()
        validate_upload(file.filename, len(content), _config.upload)
        temp_path = save_upload_temp(content, file.filename, _config.server.temp_dir)

        result = await _ocr_model.recognize(temp_path)

        text = " ".join(result.txts)
        if not result.txts:
            return OcrResponse(
                code=-1,
                msg="未识别到文字",
                data=OcrData(text="", elapsed_ms=result.elapsed_ms),
            )

        want_detail = detail.strip().lower() not in ("false", "0", "no")
        boxes = None
        if want_detail:
            boxes = [
                {"polygon": poly, "text": txt, "score": sc}
                for poly, txt, sc in zip(result.boxes, result.txts, result.scores)
            ]

        return OcrResponse(
            data=OcrData(text=text, boxes=boxes, elapsed_ms=result.elapsed_ms)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OCR 推理失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"OCR 推理失败: {e}")
    finally:
        cleanup_temp(temp_path)
