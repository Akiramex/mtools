import logging
from pathlib import Path

import torch
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


class SvModelsConfig(BaseModel):
    model: str


class SerModelsConfig(BaseModel):
    model: str
    enabled: bool = True


class ModelsConfig(BaseModel):
    asr: AsrModelsConfig
    sv: SvModelsConfig
    ser: SerModelsConfig


class HardwareConfig(BaseModel):
    device: str = "auto"  # auto/ cpu / cuda:0
    ncpu: int = 4
    ngpu: int = 1


class ConcurrencyConfig(BaseModel):
    thread_pool_size: int = 4
    max_concurrent_asr: int = 2
    max_concurrent_sv: int = 2
    max_concurrent_ser: int = 2
    use_process_pool: bool = False


class SpeakerDbConfig(BaseModel):
    path: str = "speaker_db.json"
    reload_interval_sec: int = 5
    similarity_threshold: float = 0.5


class TaskQueueConfig(BaseModel):
    max_workers: int = 2
    result_ttl_sec: int = 3600
    cleanup_interval_sec: int = 60


class UploadConfig(BaseModel):
    max_file_size_mb: int = 100
    allowed_extensions: list[str] = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]


class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: ModelsConfig
    hardware: HardwareConfig = HardwareConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    speaker_db: SpeakerDbConfig = SpeakerDbConfig()
    task_queue: TaskQueueConfig = TaskQueueConfig()
    upload: UploadConfig = UploadConfig()


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = AppConfig(**raw)

    if config.hardware.device == "auto":
        if config.hardware.ngpu > 0 and torch.cuda.is_available():
            config.hardware.device = "cuda"
        else:
            config.hardware.device = "cpu"

    logger.info("Config loaded. Device resolved to: %s", config.hardware.device)
    return config
