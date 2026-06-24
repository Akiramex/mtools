#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MeloTTS API Service
基于 FastAPI 的语音合成服务

使用方法:
    python app.py                    # 启动服务 (默认端口 8000)
    python app.py --port 8080        # 指定端口
    python app.py --reload            # 开发模式 (热重载)

Docker 部署:
    docker build -t melotts-api .
    docker run -d -p 8000:8000 -p 8888:8888 melotts-api
"""

import asyncio
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# 尝试导入 melo，如果失败则提供清晰的错误提示
try:
    from melo.api import TTS
except ImportError:
    print("Error: melo package not found. Please install MeloTTS first:")
    print("    git clone https://github.com/myshell-ai/MeloTTS.git")
    print("    cd MeloTTS")
    print("    pip install -e .")
    print("    python -m unidic download")
    raise

# ============== 配置 ==============
CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "9091")),
    "workers": int(os.getenv("WORKERS", "1")),
    "max_workers": int(os.getenv("MAX_WORKERS", "4")),
    "default_device": os.getenv("DEVICE", "cpu"),
    "default_speed": 1.0,
    "cache_models": os.getenv("CACHE_MODELS", "true").lower() == "true",
}

# 支持的语言和说话人配置
LANGUAGE_CONFIG = {
    "EN": {
        "name": "English",
        "speakers": ["EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"],
    },
    "ES": {"name": "Spanish", "speakers": ["ES"]},
    "FR": {"name": "French", "speakers": ["FR"]},
    "ZH": {"name": "Chinese", "speakers": ["ZH"]},
    "JP": {"name": "Japanese", "speakers": ["JP"]},
    "KR": {"name": "Korean", "speakers": ["KR"]},
}

# ============== 全局状态 ==============
# 模型缓存: {f"{lang}_{device}": TTS}
model_cache: Dict[str, TTS] = {}
executor = ThreadPoolExecutor(max_workers=CONFIG["max_workers"])


# ============== Pydantic 模型 ==============
class TTSRequest(BaseModel):
    """TTS 请求模型"""

    text: str = Field(..., min_length=1, max_length=2000, description="要转换的文本")
    language: str = Field(default="EN", description="语言代码: EN, ES, FR, ZH, JP, KR")
    speaker: Optional[str] = Field(
        default=None, description="说话人ID，如 EN-US, ZH 等"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速倍率")
    device: str = Field(default="cpu", description="设备: cpu, cuda, cuda:0, mps")
    format: str = Field(default="wav", description="输出格式: wav, mp3")


class TTSResponse(BaseModel):
    """TTS 响应模型"""

    success: bool
    message: str
    audio_url: Optional[str] = None
    duration: Optional[float] = None


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    version: str
    cached_models: List[str]
    supported_languages: List[str]


# ============== 辅助函数 ==============
def get_model(language: str, device: str = "cpu") -> TTS:
    """获取或加载 TTS 模型"""
    cache_key = f"{language}_{device}"

    if cache_key in model_cache:
        return model_cache[cache_key]

    print(f"Loading model: {language} on {device}...")
    model = TTS(language=language, device=device)

    if CONFIG["cache_models"]:
        model_cache[cache_key] = model

    return model


def get_default_speaker(language: str) -> str:
    """获取默认说话人"""
    if language in LANGUAGE_CONFIG:
        return LANGUAGE_CONFIG[language]["speakers"][0]
    return "EN-Default"


def synthesize(
    text: str, language: str, speaker: Optional[str], speed: float, device: str
) -> tuple:
    """
    同步合成语音

    Returns:
        tuple: (audio_bytes, duration)
    """
    # 获取模型
    model = get_model(language, device)
    speaker_ids = model.hps.data.spk2id

    # 确定说话人
    if speaker:
        if speaker not in speaker_ids:
            raise ValueError(
                f"Speaker '{speaker}' not found. Available: {list(speaker_ids.keys())}"
            )
        speaker_id = speaker_ids[speaker]
    else:
        default_speaker = get_default_speaker(language)
        if default_speaker not in speaker_ids:
            raise ValueError(
                f"Default speaker '{default_speaker}' not found for language '{language}'"
            )
        speaker_id = speaker_ids[default_speaker]

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        output_path = f.name

    try:
        # 合成语音
        model.tts_to_file(text, speaker_id, output_path, speed=speed)

        # 读取音频数据
        with open(output_path, "rb") as f:
            audio_data = f.read()

        # 获取音频时长（近似）
        duration = len(audio_data) / (44100 * 2)  # 假设 44.1kHz, 16-bit

        return audio_data, duration

    finally:
        # 清理临时文件
        if os.path.exists(output_path):
            os.unlink(output_path)


async def synthesize_async(
    text: str, language: str, speaker: Optional[str], speed: float, device: str
) -> tuple:
    """异步合成语音（在线程池中执行）"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, synthesize, text, language, speaker, speed, device
    )


# ============== FastAPI 应用 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 50)
    print("MeloTTS API Service Starting...")
    print(f"Config: {CONFIG}")
    print(f"Supported languages: {list(LANGUAGE_CONFIG.keys())}")
    print("=" * 50)

    yield

    # 关闭时清理
    executor.shutdown(wait=True)
    print("MeloTTS API Service Shutdown.")


app = FastAPI(
    title="MeloTTS API",
    description="High-quality multi-lingual text-to-speech API based on MeloTTS",
    version="1.0.0",
    lifespan=lifespan,
)

# ============== API 路由 ==============


@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "name": "MeloTTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "tts": "/tts",
            "stream": "/tts/stream",
            "health": "/health",
            "languages": "/languages",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        cached_models=list(model_cache.keys()),
        supported_languages=list(LANGUAGE_CONFIG.keys()),
    )


@app.get("/languages", tags=["Info"])
async def get_languages():
    """获取支持的语言列表"""
    return LANGUAGE_CONFIG


@app.post("/tts", tags=["TTS"])
async def text_to_speech(request: TTSRequest):
    """
    文本转语音

    - **text**: 要转换的文本
    - **language**: 语言代码 (EN, ES, FR, ZH, JP, KR)
    - **speaker**: 说话人 (可选)
    - **speed**: 语速 (0.5-2.0)
    - **device**: 设备 (cpu, cuda, mps)
    """
    try:
        # 验证语言
        if request.language not in LANGUAGE_CONFIG:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language}. "
                f"Supported: {list(LANGUAGE_CONFIG.keys())}",
            )

        # 验证说话人
        if request.speaker:
            available_speakers = LANGUAGE_CONFIG[request.language]["speakers"]
            if request.speaker not in available_speakers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker '{request.speaker}' not available for {request.language}. "
                    f"Available: {available_speakers}",
                )

        # 异步合成
        audio_data, duration = await synthesize_async(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            speed=request.speed,
            device=request.device,
        )

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=speech.wav",
                "X-Duration": str(duration),
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/tts/json", response_model=TTSResponse, tags=["TTS"])
async def text_to_speech_json(request: TTSRequest):
    """
    文本转语音 (JSON 响应)

    返回包含 base64 编码音频的 JSON
    """
    import base64

    try:
        audio_data, duration = await synthesize_async(
            text=request.text,
            language=request.language,
            speaker=request.speaker,
            speed=request.speed,
            device=request.device,
        )

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        return TTSResponse(
            success=True,
            message="Synthesis completed",
            audio_url=f"data:audio/wav;base64,{audio_b64}",
            duration=duration,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/tts/stream", tags=["TTS"])
async def tts_stream(
    text: str = Query(..., description="Text to synthesize"),
    language: str = Query("EN", description="Language code"),
    speaker: Optional[str] = Query(None, description="Speaker ID"),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speech speed"),
):
    """流式 TTS (GET 请求)"""
    try:
        audio_data, duration = await synthesize_async(
            text=text,
            language=language,
            speaker=speaker,
            speed=speed,
            device=CONFIG["default_device"],
        )

        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== 主入口 ==============
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MeloTTS API Service")
    parser.add_argument("--host", default=CONFIG["host"], help="Host to bind")
    parser.add_argument("--port", type=int, default=CONFIG["port"], help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--workers", type=int, default=CONFIG["workers"], help="Number of workers"
    )
    args = parser.parse_args()

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )
