import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.param_functions import File
from fastapi.responses import JSONResponse
from funasr import AutoModel

funAsrConfig = {
    "asr_model": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "vad_model": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_model": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "spk_model": "iic/speech_campplus_sv_zh-cn_16k-common",
    "ser_model": "iic/emotion2vec_base_finetuned",
    "open_ser": True,
}

__asr_model: AutoModel | None = None
__ser_model: AutoModel | None = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler(Path(__file__).parent / "app.log", encoding="utf-8"),
            logging.StreamHandler(),  # 输出到控制台
        ],
        force=True,  # 强制重新配置，避免被其他库覆盖
    )


logger = logging.getLogger(__name__)


def init_model():
    global __asr_model
    global __ser_model

    try:
        __asr_model = AutoModel(
            model=funAsrConfig.get("asr_model"),
            vad_model=funAsrConfig.get("vad_model"),
            punc_model=funAsrConfig.get("punc_model"),
            spk_model=funAsrConfig.get("spk_model"),
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )
    except Exception as e:
        logger.error(f"初始化 ASR model 错误: {e}")
        raise e

    try:
        if funAsrConfig.get("open_ser"):
            __ser_model = AutoModel(
                model=funAsrConfig.get("ser_model"),
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
    except Exception as e:
        logger.error(f"初始化 SER model 错误: {e}")
        raise e

    logger.info("模型初始化成功")


def get_asr_model():
    return __asr_model


def get_ser_model():
    return __ser_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_model()
    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """
    根路径，返回服务信息
    """
    return {
        "service": "FunASR语音识别服务",
        "version": "1.0.0",
        "endpoints": {
            "/": "服务信息",
            "/health": "健康检查",
            "/asr/file": "ASR音频文件识别 (POST)",
            "/ser/file": "SER音频文件识别 (POST)",
        },
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    asr_model = get_asr_model()
    ser_model = get_ser_model()

    return {
        "status": "healthy",
        "asr_model_loaded": asr_model is not None,
        "ser_model_loaded": ser_model is not None and funAsrConfig.get("open_ser"),
        "timestamp": datetime.now().isoformat(),
    }


def response_format(code: int, msg: str, data: dict = None):
    return {"code": code, "msg": msg, "data": data or {}}


def to_date(milliseconds):
    """将时间戳转换为SRT格式的时间"""
    time_obj = timedelta(milliseconds=milliseconds)
    return f"{time_obj.seconds // 3600:02d}:{(time_obj.seconds // 60) % 60:02d}:{time_obj.seconds % 60:02d}.{time_obj.microseconds // 1000:03d}"


@app.post("/ser/file")
async def ser_file(file: UploadFile = File(..., description="音频文件")):
    """
    接收音频文件，进行ASR识别并返回结果
    """
    temp_file_path = None
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(
            (".wav", ".mp3", ".m4a", ".flac", ".ogg")
        ):
            raise HTTPException(
                status_code=400,
                detail="不支持的文件格式，请上传wav、mp3、m4a、flac或ogg格式的音频文件",
            )

        # 创建临时文件保存上传的音频
        with tempfile.NamedTemporaryFile(
            delete=False, suffix="ser_" + os.path.splitext(file.filename)[1]
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        logger.info(
            f"文件已保存到临时路径: {temp_file_path}, 文件大小: {len(content)} bytes"
        )

        # 获取SER模型
        ser_model = get_ser_model()
        if ser_model is None:
            raise HTTPException(status_code=500, detail="SER模型未初始化")

        # 进行SER推理
        logger.info(f"开始SER推理，文件: {file.filename}")
        result = ser_model.generate(
            input=temp_file_path,
            granularity="utterance",
            extract_embedding=False,
        )

        # 解析结果
        if result and len(result) > 0:
            # FunASR返回的结果结构通常是列表，包含文本和其他信息
            ser_result = result[0] if isinstance(result, list) else result

            # 提取文本
            if isinstance(ser_result, dict):
                response_data = response_format(
                    code=0,
                    msg="success",
                    data=ser_result,
                )
            else:
                # 如果结果是字符串或其他类型
                response_data = response_format(
                    code=-1,
                    msg="asr 失败",
                )
            return JSONResponse(content=response_data)
        else:
            logger.warning(f"ASR推理返回空结果，文件: {file.filename}")
            return JSONResponse(
                content={
                    "text": "",
                    "filename": file.filename,
                    "file_size": len(content),
                    "message": "未识别到有效语音",
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR推理过程中发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR推理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"已清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")


@app.post("/asr/file")
async def asr_file(file: UploadFile = File(..., description="音频文件")):
    """
    接收音频文件，进行ASR识别并返回结果
    """
    temp_file_path = None
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(
            (".wav", ".mp3", ".m4a", ".flac", ".ogg")
        ):
            raise HTTPException(
                status_code=400,
                detail="不支持的文件格式，请上传wav、mp3、m4a、flac或ogg格式的音频文件",
            )

        # 创建临时文件保存上传的音频
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name

        logger.info(
            f"文件已保存到临时路径: {temp_file_path}, 文件大小: {len(content)} bytes"
        )

        # 获取ASR模型
        asr_model = get_asr_model()
        if asr_model is None:
            raise HTTPException(status_code=500, detail="ASR模型未初始化")

        # 进行ASR推理
        logger.info(f"开始ASR推理，文件: {file.filename}")
        result = asr_model.generate(input=temp_file_path)

        # 解析结果
        if result and len(result) > 0:
            # FunASR返回的结果结构通常是列表，包含文本和其他信息
            asr_result = result[0] if isinstance(result, list) else result

            # 提取文本
            if isinstance(asr_result, dict):
                all_content = asr_result.get("text", "")
                sentence_info = asr_result.get("sentence_info", [])
                sentence_list = []
                i = 1
                for sentence in sentence_info:
                    start = to_date(sentence["start"])
                    end = to_date(sentence["end"])
                    text = sentence["text"]
                    spk = sentence["spk"]
                    sentence_list.append(
                        {
                            "index": i,
                            "text": text,
                            "start": start,
                            "endTime": end,
                            "spk": spk,
                        }
                    )
                    i += 1
                response_data = response_format(
                    code=0,
                    msg="success",
                    data={
                        "text": all_content,
                        "sentence_list": sentence_list,
                    },
                )
            else:
                # 如果结果是字符串或其他类型
                response_data = response_format(
                    code=-1,
                    msg="asr 失败",
                )
            return JSONResponse(content=response_data)
        else:
            logger.warning(f"ASR推理返回空结果，文件: {file.filename}")
            return JSONResponse(
                content={
                    "text": "",
                    "filename": file.filename,
                    "file_size": len(content),
                    "message": "未识别到有效语音",
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR推理过程中发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ASR推理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"已清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9090,
        # log_config=None,  # 取消注释会将 uvicorn的日志配置禁用，使用我们自己的配置
        log_level="info",
    )
