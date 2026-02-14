import logging
import os
from datetime import timedelta
from multiprocessing import Manager
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

_model: AutoModel | None = None


asr_model = (
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)

vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

punc_model = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

spk_model = "iic/speech_campplus_sv_zh-cn_16k-common"

wav_path = Path(__file__).resolve().parents[1] / "data" / "asr_example_zh.wav"


def to_date(milliseconds):
    """将时间戳转换为SRT格式的时间"""
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"


def get_model():
    global _model
    if _model is None:
        try:
            _model = AutoModel(
                model=asr_model,
                vad_model=vad_model,
                punc_model=punc_model,
                spk_model=spk_model,
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
                ngpu=1 if torch.cuda.is_available() else 0,
                ncpu=os.cpu_count(),
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return _model


def generate(audio: str) -> dict | None:
    model = get_model()
    if model:
        result = model.generate(input=audio, batch_size_s=300)
        if result:
            text_all = result[0]["text"]
            if len(text_all) > 0:
                sentence_info = result[0]["sentence_info"]
                sentence_list = []
                i = 1
                for sentence in sentence_info:
                    start = to_date(sentence["start"])
                    end = to_date(sentence["end"])
                    text = sentence["text"]
                    spk = sentence["spk"]
                    sentence_list.append(
                        {
                            "sentence_index": i,
                            "text": text,
                            "start": start,
                            "end": end,
                            "speaker": spk,
                        }
                    )
                i += 1
                return {"text": text_all, "segments": sentence_list}
            else:
                return None
        else:
            return None
    else:
        return None
