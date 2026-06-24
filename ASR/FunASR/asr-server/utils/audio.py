import logging
import os
import uuid
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf

from config import UploadConfig

logger = logging.getLogger(__name__)


def validate_upload(filename: str, content_length: int | None, config: UploadConfig) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.allowed_extensions:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {config.allowed_extensions}",
        )

    if content_length and content_length > config.max_file_size_mb * 1024 * 1024:
        from fastapi import HTTPException

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


def extract_segment(audio_path: str, start_ms: int, end_ms: int, temp_dir: str) -> str:
    """从音频文件中截取 [start_ms, end_ms) 毫秒的片段，保存为临时 WAV 文件。"""
    data, sr = sf.read(audio_path)
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)
    segment = data[start_sample:end_sample]

    if segment.ndim == 2:
        segment = segment[:, 0]
    if segment.dtype != np.int16:
        segment = (np.clip(segment, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        segment = segment.astype(np.int16)

    os.makedirs(temp_dir, exist_ok=True)
    out_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
    wavfile.write(out_path, sr, segment)
    return out_path
