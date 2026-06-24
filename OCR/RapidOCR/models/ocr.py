import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config import AppConfig
from utils.serialize import to_python

logger = logging.getLogger(__name__)


def build_engine(config: AppConfig) -> Any:
    """按 config.models.ocr 拼 RapidOCR 的 params 并构造引擎。

    - model_path 系列留空 → 用 rapidocr 包默认模型（自动下载到安装目录的 models/）
    - 填了路径 → 走离线本地模型
    """
    from rapidocr import RapidOCR

    ocr = config.models.ocr
    params: dict[str, Any] = {
        "EngineConfig.onnxruntime.intra_op_num_threads": ocr.intra_op_num_threads,
    }
    if ocr.det_model_path:
        params["Det.model_path"] = ocr.det_model_path
    if ocr.cls_model_path:
        params["Cls.model_path"] = ocr.cls_model_path
    if ocr.rec_model_path:
        params["Rec.model_path"] = ocr.rec_model_path
    if ocr.rec_keys_path:
        params["Rec.rec_keys_path"] = ocr.rec_keys_path
    if ocr.max_side_len:
        params["Global.max_side_len"] = ocr.max_side_len
    return RapidOCR(params=params)


@dataclass
class OcrResult:
    boxes: list = field(default_factory=list)
    txts: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    elapsed_ms: int = 0


class ModelRunner:
    """RapidOCR 引擎的信号量+线程池封装，避免 CPU 密集推理阻塞事件循环。"""

    def __init__(self, model, executor, semaphore):
        self.model = model
        self.executor = executor
        self.sem = semaphore

    async def __call__(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        async with self.sem:
            return await loop.run_in_executor(
                self.executor, functools.partial(self.model, *args, **kwargs)
            )

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


class OcrModel:
    def __init__(self, runner: ModelRunner):
        self.runner = runner

    async def recognize(self, image_path: str) -> OcrResult:
        start = time.perf_counter()
        output = await self.runner(image_path)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        raw_boxes = to_python(getattr(output, "boxes", None))
        txts = list(getattr(output, "txts", None) or [])
        scores = [float(s) for s in (getattr(output, "scores", None) or [])]
        elapse_list = [int(round(s * 1000)) for s in (getattr(output, "elapse_list", None) or [])]

        boxes = []
        if raw_boxes:
            for poly in raw_boxes:
                boxes.append([[int(round(p)) for p in point] for point in poly])

        logger.info(
            "OCR %d boxes, total %dms, per-stage(det/cls/rec) ms=%s",
            len(txts), elapsed_ms, elapse_list,
        )
        return OcrResult(boxes=boxes, txts=txts, scores=scores, elapsed_ms=elapsed_ms)
