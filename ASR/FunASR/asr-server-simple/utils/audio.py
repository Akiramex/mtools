import logging
import os
import uuid

from fastapi import HTTPException

from config import UploadConfig

logger = logging.getLogger(__name__)


def validate_upload(filename: str, content_length: int | None, config: UploadConfig) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {config.allowed_extensions}",
        )

    if content_length and content_length > config.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {config.max_file_size_mb}MB",
        )


def save_upload_temp(content: bytes, filename: str, temp_dir: str) -> str:
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(filename)[1]
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    with open(temp_path, "wb") as f:
        f.write(content)
    logger.debug("Saved temp file: %s (%d bytes)", temp_path, len(content))
    return temp_path


def cleanup_temp(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception as e:
            logger.warning("Failed to cleanup temp file %s: %s", path, e)
