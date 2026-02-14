from pathlib import Path

from funasr import AutoModel

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
wav_path = Path(__file__).resolve().parents[1] / "data" / "asr_example_zh.wav"

if not wav_path.exists():
    raise FileNotFoundError(f"WAV file not found: {wav_path}")

model = AutoModel(
    model=model_dir,
    device="cpu",
)
res = model.generate(input=[str(wav_path)], cache={}, batch_size_s=0)
text = res[0]["text"]

print(text)
