import asyncio
import logging
import os
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
_ocr_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _executor, _ocr_model

    config_path = Path(__file__).parent / "config.yaml"
    _config = load_config(str(config_path))
    setup_logging(_config.server.log_level, _config.server.log_file)

    # 必须在导入 onnxruntime/rapidocr 之前设置，使线程数上限生效
    os.environ["OMP_NUM_THREADS"] = str(_config.models.ocr.intra_op_num_threads)

    _executor = ThreadPoolExecutor(max_workers=_config.concurrency.thread_pool_size)

    from models.ocr import ModelRunner, OcrModel, build_engine

    logger.info("正在加载 RapidOCR 模型...")
    engine = build_engine(_config)
    runner = ModelRunner(
        model=engine,
        executor=_executor,
        semaphore=asyncio.Semaphore(_config.concurrency.max_concurrent_ocr),
    )
    _ocr_model = OcrModel(runner=runner)
    logger.info("RapidOCR 模型加载完成。")

    from routers import ocr as ocr_router

    ocr_router.set_model(_ocr_model, _config)

    logger.info("服务就绪: %s:%d", _config.server.host, _config.server.port)
    yield

    logger.info("正在关闭...")
    if _ocr_model is not None and _ocr_model.runner is not None:
        _ocr_model.runner.shutdown()
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
    logger.info("已关闭。")


app = FastAPI(title="RapidOCR Server", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if _config is not None:
        max_bytes = _config.upload.max_file_size_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "File too large"})
    return await call_next(request)


from routers import ocr as ocr_router_mod  # noqa: E402

app.include_router(ocr_router_mod.router, tags=["OCR"])


@app.get("/")
async def root():
    return {
        "service": "RapidOCR Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/": "服务信息",
            "/health": "健康检查",
            "/ocr": "图片文字识别 (POST)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_ocr_model is not None,
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
