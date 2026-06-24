# CosyVoice API 快速参考

## 安装

```bash
# 克隆
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# 环境
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice
pip install -r requirements.txt

# 可选: Sox
# Ubuntu: sudo apt-get install sox libsox-dev
# CentOS: sudo yum install sox sox-devel
```

## 模型下载

```python
# ModelScope (国内推荐)
from modelscope import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', 
                  local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', 
                  local_dir='pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', 
                  local_dir='pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', 
                  local_dir='pretrained_models/CosyVoice-300M-Instruct')

# HuggingFace (海外)
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

## 基础用法

```python
from cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

# 初始化
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# SFT 模式 - 固定说话人
for i in cosyvoice.inference_sft('你好，欢迎使用CosyVoice。', '中文女'):
    torchaudio.save(f'sft_{i}.wav', i['tts_speech'], 22050)

# 零样本克隆
prompt_speech = load_wav('reference.wav', 16000)
for i in cosyvoice.inference_zero_shot(
    '这是克隆后的声音。',
    '参考音频的文本内容',
    prompt_speech
):
    torchaudio.save(f'zero_shot_{i}.wav', i['tts_speech'], 22050)

# 跨语言合成
for i in cosyvoice.inference_cross_lingual(
    'This is English text.',
    '参考音频的中文文本',
    prompt_speech
):
    torchaudio.save(f'cross_lingual_{i}.wav', i['tts_speech'], 22050)

# 指令控制
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
for i in cosyvoice.inference_instruct(
    '你好世界。',
    '用快乐的语气朗读'
):
    torchaudio.save(f'instruct_{i}.wav', i['tts_speech'], 22050)
```

## vLLM 加速推理

```bash
# 安装 vLLM
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm

# vLLM 0.9.0
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4

# vLLM 0.11.x (推荐)
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4
```

```python
# vllm_example.py
# 详见项目中的 vllm_example.py
```

## WebUI

```bash
python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
# 访问 http://localhost:50000
```

## 服务部署

### FastAPI

```bash
cd runtime/python/fastapi
python server.py --port 50000 --model_dir ../../pretrained_models/CosyVoice-300M
python client.py --port 50000 --mode zero_shot
```

### gRPC

```bash
cd runtime/python/grpc
python server.py --port 50000 --max_conc 4 --model_dir ../../pretrained_models/CosyVoice-300M
python client.py --port 50000 --mode zero_shot
```

### Docker

```bash
cd runtime/python
docker build -t cosyvoice:v1.0 .

# FastAPI
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/runtime/python/fastapi && \
  python3 server.py --port 50000 --model_dir iic/CosyVoice-300M"

# gRPC
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 \
  /bin/bash -c "cd /opt/CosyVoice/runtime/python/grpc && \
  python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M"
```

### TensorRT-LLM (4x 加速)

```bash
cd runtime/triton_trtllm
docker compose up -d
```

## 推理模式说明

| 模式 | 模型 | 用途 | 输入 |
|------|------|------|------|
| `sft` | CosyVoice-300M-SFT | 单说话人 | 文本 |
| `zero_shot` | CosyVoice-300M | 声音克隆 | 文本 + 参考音频 |
| `cross_lingual` | CosyVoice-300M | 跨语言 | 文本 + 参考音频 |
| `instruct` | CosyVoice-300M-Instruct | 指令控制 | 文本 + 指令 |

## 流式推理 (CosyVoice 2.0+)

```python
# 启用流式
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')

# 流式输出
for chunk in cosyvoice.inference_sft_streaming('长文本...', '中文女'):
    # 实时播放或处理
    pass
```

## 音素控制 (Fun-CosyVoice 3.0)

```python
# 中文拼音
text = "你好(ni3 hao3)世界"

# 英文音素 (CMU)
text = "hello HH EH L OW world"
```

## 常见问题

### Q: 如何选择模型?

- **通用推荐**: Fun-CosyVoice3-0.5B
- **实时对话**: CosyVoice2-0.5B
- **声音克隆**: CosyVoice-300M
- **指令控制**: CosyVoice-300M-Instruct

### Q: GPU 内存需求?

- 推理: 8GB+ (300M), 16GB+ (0.5B)
- 训练: 24GB+

### Q: 支持的语言?

中文、英文、日语、韩语、德语、西班牙语、法语、意大利语、俄语，以及18+种中文方言

### Q: 如何提高克隆质量?

1. 使用 3-10 秒高质量参考音频
2. 参考音频与目标文本语言一致时效果最佳
3. 避免背景噪音和音乐
