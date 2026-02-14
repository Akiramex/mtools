import logging
import os
from pathlib import Path

import torch

from funasr import AutoModel


def setup_logger():
    """日志配置函数，主模块启动时调用一次"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log", encoding="utf-8"),  # 输出到文件
            logging.StreamHandler(),  # 同时输出到控制台
        ],
    )


setup_logger()
logger = logging.getLogger(__name__)


ser_model = "iic/emotion2vec_base_finetuned"
model: AutoModel | None = None
wav_path = Path(__file__).resolve().parents[1] / "data" / "asr_example_zh.wav"


def get_ser_model():
    global model
    if model is None:
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = AutoModel(
                model=ser_model,
                hub="ms",
                device=device,
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as e:
            logger.error(f"Failed to load SER model: {e}")
    return model


def generate_with_emotion(audio: str) -> list[dict] | None:
    """ASR + per-sentence SER (local audio file path only)."""
    if not os.path.exists(audio):
        raise FileNotFoundError(f"Audio file not found: {audio}")
    ser = get_ser_model()
    if ser is not None:
        result = ser.generate(
            audio,
            granularity="utterance",
            extract_embedding=False,
        )
        return result
    return None


print(generate_with_emotion(str(wav_path)))
