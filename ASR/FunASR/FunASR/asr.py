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
                            "emotion": "",
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


def generate_str(audio: str) -> str | None:
    """生成格式为 [start,end]spk_x text 的字符串

    返回格式示例:
    [00:00:01.730,00:00:03.990]spk_0 欢迎使用百度网盘同步空间，
    [00:00:04.330,00:00:07.510]spk_0 一起来体验文件多端同步效率神器吧。
    """
    model = get_model()
    if model:
        result = model.generate(input=audio, batch_size_s=300)
        if result:
            text_all = result[0]["text"]
            if len(text_all) > 0:
                sentence_info = result[0]["sentence_info"]
                lines = []
                for sentence in sentence_info:
                    start = to_date(sentence["start"])
                    end = to_date(sentence["end"])
                    text = sentence["text"]
                    spk = sentence["spk"]
                    # 格式: [start,end]spk_x text
                    line = f"[{start},{end}]spk_{spk} {text}"
                    lines.append(line)
                return "\n".join(lines)
            else:
                return None
        else:
            return None
    else:
        return None


def process_directory(directory_path: str) -> None:
    """扫描目录下所有的wav文件，对每个wav进行ASR，生成funasr.txt在该wav的目录下

    Args:
        directory_path: 要扫描的目录路径
    """
    directory = Path(directory_path)

    if not directory.exists():
        logger.error(f"目录不存在: {directory_path}")
        return

    if not directory.is_dir():
        logger.error(f"路径不是目录: {directory_path}")
        return

    # 递归查找所有wav文件
    wav_files = list(directory.rglob("*.wav"))

    if not wav_files:
        logger.info(f"在目录 {directory_path} 中没有找到wav文件")
        return

    logger.info(f"找到 {len(wav_files)} 个wav文件")

    for i, wav_file in enumerate(wav_files, 1):
        try:
            logger.info(f"处理文件 {i}/{len(wav_files)}: {wav_file}")

            # 进行ASR识别
            result_str = generate_str(str(wav_file))

            if result_str:
                # 在wav文件所在目录下创建funasr.txt
                output_file = wav_file.parent / "funasr.txt"

                # 写入结果
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(result_str)

                # 统计信息
                line_count = len(result_str.split("\n"))
                logger.info(f"  成功生成: {output_file} (共{line_count}行)")
            else:
                logger.warning(f"  无法识别文件: {wav_file}")

        except Exception as e:
            logger.error(f"  处理文件失败 {wav_file}: {e}")


if __name__ == "__main__":
    # 示例用法
    import sys

    if len(sys.argv) > 1:
        # 使用命令行参数指定的目录
        target_directory = sys.argv[1]
    else:
        # 使用默认目录（当前脚本所在目录的父目录）
        target_directory = Path(__file__).parent.parent

    print(f"开始扫描目录: {target_directory}")
    process_directory(str(target_directory))
    print("处理完成")
