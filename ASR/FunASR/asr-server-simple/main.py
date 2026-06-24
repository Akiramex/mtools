import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import load_config
from schemas.responses import HealthResponse

logger = logging.getLogger(__name__)


def setup_logging(log_level: str, log_file: str = ""):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


_config = None
_executor: ThreadPoolExecutor | None = None
_asr_model = None
_task_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _executor, _asr_model, _task_manager

    config_path = Path(__file__).parent / "config.yaml"
    _config = load_config(str(config_path))
    setup_logging(_config.server.log_level, _config.server.log_file)

    _executor = ThreadPoolExecutor(max_workers=_config.concurrency.thread_pool_size)

    from funasr import AutoModel
    from models.asr import AsrModel, ModelRunner

    asr_kwargs = {
        "model": _config.models.asr.model,
        "device": _config.hardware.device,
        "ncpu": _config.hardware.ncpu,
        "disable_update": True,
        "disable_pbar": True,
        "disable_log": True,
    }
    if _config.models.asr.vad_model:
        asr_kwargs["vad_model"] = _config.models.asr.vad_model
    if _config.models.asr.punc_model:
        asr_kwargs["punc_model"] = _config.models.asr.punc_model
    if _config.models.asr.spk_model:
        asr_kwargs["spk_model"] = _config.models.asr.spk_model

    logger.info("Loading ASR model...")
    asr_auto = AutoModel(**asr_kwargs)
    asr_runner = ModelRunner(
        model=asr_auto,
        executor=_executor,
        semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_asr),
    )
    _asr_model = AsrModel(runner=asr_runner)
    logger.info("ASR model loaded.")

    from routers import asr as asr_router
    from utils.task_manager import TaskManager

    _task_manager = TaskManager(_config.task_queue)
    _task_manager.start(asr_router._process_task)

    asr_router.set_models(_asr_model, _config, _task_manager)

    logger.info("Server ready on %s:%d", _config.server.host, _config.server.port)
    yield

    logger.info("Shutting down...")
    if _task_manager:
        await _task_manager.stop()
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Done.")


app = FastAPI(title="FunASR ASR Server (Simple)", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if _config:
        max_bytes = _config.upload.max_file_size_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "File too large"})
    return await call_next(request)


from routers import asr as asr_router_mod

app.include_router(asr_router_mod.router, tags=["ASR"])


@app.get("/")
async def root():
    return {"service": "FunASR ASR Server (Simple)", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_asr_model is not None,
        timestamp=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    if _config is None:
        _config = load_config(str(Path(__file__).parent / "config.yaml"))
    uvicorn.run(
        "main:app",
        host=_config.server.host,
        port=_config.server.port,
        log_level=_config.server.log_level,
    )
