import logging
from pathlib import Path

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 9090
    log_level: str = "info"
    temp_dir: str = "temp_files"
    log_file: str = ""


class AsrModelsConfig(BaseModel):
    model: str
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None


class ModelsConfig(BaseModel):
    asr: AsrModelsConfig


class HardwareConfig(BaseModel):
    device: str = "cpu"
    ncpu: int = 4


class ConcurrencyConfig(BaseModel):
    thread_pool_size: int = 4
    max_concurrent_asr: int = 4


class TaskQueueConfig(BaseModel):
    max_workers: int = 2
    result_ttl_sec: int = 3600
    cleanup_interval_sec: int = 60


class UploadConfig(BaseModel):
    max_file_size_mb: int = 100
    allowed_extensions: list[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: ModelsConfig
    hardware: HardwareConfig = HardwareConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    task_queue: TaskQueueConfig = TaskQueueConfig()
    upload: UploadConfig = UploadConfig()


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = AppConfig(**raw)
    logger.info("Config loaded. Device: %s", config.hardware.device)
    return config
