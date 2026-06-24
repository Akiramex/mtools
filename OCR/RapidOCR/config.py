import logging
import os
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "info"
    temp_dir: str = "temp_files"
    log_file: str = "logs/app.log"


class OcrModelsConfig(BaseModel):
    det_model_path: str | None = None
    cls_model_path: str | None = None
    rec_model_path: str | None = None
    rec_keys_path: str | None = None
    intra_op_num_threads: int = 4


class ModelsConfig(BaseModel):
    ocr: OcrModelsConfig = OcrModelsConfig()


class ConcurrencyConfig(BaseModel):
    thread_pool_size: int = 4
    max_concurrent_ocr: int = 2


class UploadConfig(BaseModel):
    max_file_size_mb: int = 20
    allowed_extensions: list[str] = [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    ]


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: ModelsConfig = ModelsConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    upload: UploadConfig = UploadConfig()


_ENV_MAP = {
    "OCR_HOST": ("server", "host"),
    "OCR_PORT": ("server", "port"),
    "OCR_LOG_LEVEL": ("server", "log_level"),
    "OCR_LOG_FILE": ("server", "log_file"),
    "OCR_THREAD_POOL_SIZE": ("concurrency", "thread_pool_size"),
    "OCR_MAX_CONCURRENT_OCR": ("concurrency", "max_concurrent_ocr"),
    "OCR_MAX_FILE_SIZE_MB": ("upload", "max_file_size_mb"),
    "OCR_INTRA_OP_NUM_THREADS": ("models", "ocr", "intra_op_num_threads"),
}


def _apply_env_overrides(raw: dict) -> dict:
    for env_key, path in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        node = raw
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = val
    return raw


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _apply_env_overrides(raw)
    config = AppConfig(**raw)
    logger.info(
        "Config loaded. port=%d max_concurrent_ocr=%d",
        config.server.port,
        config.concurrency.max_concurrent_ocr,
    )
    return config
