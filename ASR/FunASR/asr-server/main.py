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
_sv_model = None
_ser_model = None
_task_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _executor, _asr_model, _sv_model, _ser_model, _task_manager

    config_path = Path(__file__).parent / "config.yaml"
    _config = load_config(str(config_path))
    setup_logging(_config.server.log_level, _config.server.log_file)

    _executor = ThreadPoolExecutor(max_workers=_config.concurrency.thread_pool_size)

    # Load ASR model (composite: ASR + VAD + PUNC + SPK diarization)
    from funasr import AutoModel
    from models.asr import AsrModel
    from models.base import ModelRunner

    asr_kwargs = {
        "model": _config.models.asr.model,
        "device": _config.hardware.device,
        "ncpu": _config.hardware.ncpu,
        "ngpu": _config.hardware.ngpu,
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

    if _config.concurrency.use_process_pool:
        import functools as ft
        from models.base import create_model_from_config

        logger.info("Loading ASR model (process pool, workers=%d)...", _config.concurrency.max_concurrent_asr)
        asr_runner = ModelRunner(
            semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_asr),
            model_factory=ft.partial(create_model_from_config, **asr_kwargs),
            max_workers=_config.concurrency.max_concurrent_asr,
        )
        _asr_model = AsrModel(runner=asr_runner)
        logger.info("ASR model process pool ready.")
    else:
        logger.info("Loading ASR model...")
        asr_auto = AutoModel(**asr_kwargs)
        asr_runner = ModelRunner(
            model=asr_auto,
            semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_asr),
            executor=_executor,
        )
        _asr_model = AsrModel(runner=asr_runner)
        logger.info("ASR model loaded.")

    # Load SV model (standalone for registration/identification)
    from models.sv import SvModel

    logger.info("Loading SV model...")
    sv_auto = AutoModel(
        model=_config.models.sv.model,
        device=_config.hardware.device,
        ncpu=_config.hardware.ncpu,
        ngpu=_config.hardware.ngpu,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
    )
    sv_runner = ModelRunner(
        model=sv_auto,
        semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_sv),
        executor=_executor,
    )
    db_path = str(Path(__file__).parent / _config.speaker_db.path)
    _sv_model = SvModel(runner=sv_runner, config=_config.speaker_db, db_path=db_path)
    logger.info("SV model loaded.")

    # Load SER model (optional)
    from models.ser import SerModel

    if _config.models.ser.enabled:
        logger.info("Loading SER model...")
        ser_auto = AutoModel(
            model=_config.models.ser.model,
            device=_config.hardware.device,
            ncpu=_config.hardware.ncpu,
            ngpu=_config.hardware.ngpu,
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
        ser_runner = ModelRunner(
            model=ser_auto,
            semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_ser),
            executor=_executor,
        )
        _ser_model = SerModel(runner=ser_runner)
        logger.info("SER model loaded.")
    else:
        logger.info("SER model disabled.")

    # Inject models into routers
    from routers import asr as asr_router
    from routers import ser as ser_router
    from routers import speaker as speaker_router
    from utils.task_manager import TaskManager

    _task_manager = TaskManager(_config.task_queue)
    _task_manager.start(asr_router._process_task)

    asr_router.set_models(_asr_model, _sv_model, _config, _task_manager)
    speaker_router.set_model(_sv_model, _config)
    ser_router.set_model(_ser_model, _config)

    logger.info("All models initialized. Server ready on %s:%d", _config.server.host, _config.server.port)
    yield

    logger.info("Shutting down...")
    if _task_manager:
        await _task_manager.stop()
    if _asr_model and hasattr(_asr_model, "runner"):
        _asr_model.runner.shutdown()
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)
    logger.info("Done.")


app = FastAPI(title="FunASR ASR Server", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if _config:
        max_bytes = _config.upload.max_file_size_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "File too large"})
    return await call_next(request)


from routers import asr as asr_router_mod
from routers import ser as ser_router_mod
from routers import speaker as speaker_router_mod

app.include_router(asr_router_mod.router, tags=["ASR"])
app.include_router(speaker_router_mod.router, tags=["Speaker"])
app.include_router(ser_router_mod.router, tags=["SER"])


@app.get("/")
async def root():
    return {"service": "FunASR ASR Server", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "asr_model_loaded": _asr_model is not None,
        "sv_model_loaded": _sv_model is not None,
        "ser_model_loaded": _ser_model is not None,
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    if _config is None:
        _config = load_config(str(Path(__file__).parent / "config.yaml"))
    uvicorn.run(
        "main:app",
        host=_config.server.host,
        port=_config.server.port,
        log_level=_config.server.log_level,
    )
