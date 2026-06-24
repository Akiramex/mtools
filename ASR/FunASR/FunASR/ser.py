import logging
import os
import subprocess
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
wav_path_copy = Path(__file__).resolve().parents[1] / "data" / "asr_example_zh1.wav"
scp_file = Path(__file__).resolve().parents[1] / "data" / "wav.scp"


def get_ser_model():
    global model
    if model is None:
        try:
            model = AutoModel(
                model=ser_model,
                disable_update=True,
                disable_pbar=True,
                disable_log=True,
            )
        except Exception as e:
            logger.error(f"Failed to load SER model: {e}")
    return model


def generate_with_emotion(audio: str | list[str]) -> str | None:
    """ASR + per-sentence SER (local audio file path only)."""
    ser = get_ser_model()
    if ser is not None:
        result = ser.generate(
            input=audio,
            granularity="utterance",
            extract_embedding=False,
        )

        item = result[0]
        labels = item["labels"]
        scores = item["scores"]

        # 找到最高分数的索引
        max_score = max(scores)
        max_index = scores.index(max_score)
        max_label = labels[max_index]

        return max_label
    return None


def get_audio_duration_ms(wav_path: str | Path) -> float:
    """获取音频时长（毫秒）"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip()) * 1000


def split_wav_by_ms(
    wav_path: str | Path,
    intervals: list[tuple[int, int]],
    output_dir: str | Path | None = None,
    prefix: str = "slice",
) -> list[Path]:
    """
    将 WAV 文件按毫秒区间分割成多个文件（使用 ffmpeg）。

    Args:
        wav_path: 输入 WAV 文件路径
        intervals: 毫秒区间列表，如 [(0, 1000), (1000, 2000)] 表示 0-1秒, 1-2秒
        output_dir: 输出目录，默认为输入文件同目录下的 'slices' 文件夹
        prefix: 输出文件名前缀

    Returns:
        分割后的 WAV 文件路径列表
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # 获取总时长
    duration_ms = get_audio_duration_ms(wav_path)

    # 设置输出目录
    if output_dir is None:
        output_dir = wav_path.parent / "slices"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []

    for i, (start_ms, end_ms) in enumerate(intervals):
        # 边界检查
        start_ms = max(0, min(start_ms, duration_ms))
        end_ms = max(0, min(end_ms, duration_ms))

        # 确保 start <= end
        if start_ms > end_ms:
            start_ms, end_ms = end_ms, start_ms

        # 跳过无效区间
        if start_ms == end_ms:
            logger.warning(f"Skipped empty interval: {start_ms}ms - {end_ms}ms")
            continue

        # 生成输出文件名
        output_path = output_dir / f"{prefix}_{start_ms}ms_{end_ms}ms.wav"

        # 使用 ffmpeg 切割
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖已存在的文件
            "-i",
            str(wav_path),
            "-ss",
            str(start_ms / 1000),  # 转为秒
            "-t",
            str((end_ms - start_ms) / 1000),  # 时长
            "-ar",
            "16000",  # 重采样到 16kHz
            "-ac",
            "1",  # 单声道
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        output_files.append(output_path)
        logger.info(f"Saved: {output_path} ({end_ms - start_ms}ms)")

    return output_files


def split_wav_equally(
    wav_path: str | Path,
    chunk_size_ms: int = 1000,
    overlap_ms: int = 0,
    output_dir: str | Path | None = None,
    prefix: str = "chunk",
) -> list[Path]:
    """
    将 WAV 文件按固定时长均匀分割（使用 ffmpeg）。

    Args:
        wav_path: 输入 WAV 文件路径
        chunk_size_ms: 每个片段的时长（毫秒）
        overlap_ms: 片段之间的重叠时长（毫秒）
        output_dir: 输出目录
        prefix: 输出文件名前缀

    Returns:
        分割后的 WAV 文件路径列表
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    duration_ms = get_audio_duration_ms(wav_path)

    # 生成区间
    intervals = []
    current = 0
    while current < duration_ms:
        start = current
        end = min(current + chunk_size_ms, duration_ms)
        intervals.append((int(start), int(end)))
        current += chunk_size_ms - overlap_ms

    return split_wav_by_ms(wav_path, intervals, output_dir, prefix)


# 示例用法
if __name__ == "__main__":
    print(generate_with_emotion(str(wav_path)))

    # 方式 1: 按固定时长分割（每 1 秒一个片段）
    # slices = split_wav_equally(wav_path, chunk_size_ms=1000)
    # print(f"Created {len(slices)} slices (1s each)")

    # 方式 2: 自定义区间
    # intervals = [(0, 500), (500, 1500), (1500, 2500)]  # 0-0.5s, 0.5-1.5s, 1.5-2.5s
    # slices = split_wav_by_ms(wav_path, intervals)
    # print(f"Created {len(slices)} custom slices")
