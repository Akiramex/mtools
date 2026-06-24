# FunASR ASR Server

基于 [FunASR](https://github.com/modelscope/FunASR) 的语音服务，提供语音识别（ASR）、声纹识别（SV）、语音情感识别（SER）三大能力，通过 FastAPI HTTP API 对外提供服务。

## 功能概览

| 功能 | 说明 |
|------|------|
| **ASR 语音识别** | 上传音频文件，返回带时间戳的逐句转写结果，支持说话人分离（Speaker Diarization） |
| **SV 声纹管理** | 注册、删除、查询说话人声纹，识别音频中的说话人 |
| **SER 情感识别** | 分析音频中的情感（如 angry / happy / sad / neutral 等） |
| **异步任务队列** | 支持提交长音频异步转写，轮询获取结果 |

## 快速开始

### 环境要求

- Python 3.11+
- CUDA（可选，GPU 推理）
- 内存 >= 8 GB（模型加载需要较大内存）

### 安装依赖

```bash
cd asr-server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> GPU 用户需安装对应 CUDA 版本的 PyTorch，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

### 修改配置

编辑 `config.yaml`，默认配置开箱即用。关键配置项：

```yaml
server:
  host: "0.0.0.0"
  port: 9090

models:
  asr:
    model: "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    vad_model: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    punc_model: "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    spk_model: "iic/speech_campplus_sv_zh-cn_16k-common"
  sv:
    model: "iic/speech_campplus_sv_zh-cn_16k-common"
  ser:
    model: "iic/emotion2vec_base_finetuned"
    enabled: true           # 设为 false 可禁用 SER

hardware:
  device: "auto"            # auto / cuda / cpu
  ngpu: 1
```

### 启动服务

```bash
python main.py
```

启动后访问：
- API 文档（Swagger）：`http://localhost:9090/docs`
- 健康检查：`http://localhost:9090/health`

## API 接口

### 1. ASR 语音识别

**同步转写** — 上传音频，立即返回结果。

```
POST /asr/file
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 音频文件（.wav / .mp3 / .m4a / .flac / .ogg） |
| `identify_speakers` | string | 否 | `"true"` 开启说话人识别，默认 `"false"` |

**响应示例：**

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "text": "你好世界",
    "sentence_list": [
      {
        "index": 1,
        "text": "你好世界",
        "start": 0,
        "end": 1500,
        "speaker": 0,
        "speaker_name": "张三"
      }
    ]
  }
}
```

---

**异步转写** — 提交长音频任务，轮询获取结果。

```
POST /asr/task
```

参数与 `/asr/file` 相同，返回任务 ID：

```json
{"code": 0, "msg": "success", "data": {"task_id": "a1b2c3d4e5f6g7h8"}}
```

```
GET /asr/task/{task_id}
```

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "task_id": "a1b2c3d4e5f6g7h8",
    "status": "completed",
    "created_at": "2026-04-21T10:00:00",
    "finished_at": "2026-04-21T10:00:15",
    "error": null,
    "result": {
      "text": "...",
      "sentence_list": [...]
    }
  }
}
```

`status` 取值：`pending` → `processing` → `completed` / `failed`

---

### 2. SV 声纹管理

**注册说话人**

```
POST /speaker/register
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 说话人音频（建议 WAV，>= 5 秒） |
| `name` | string | 是 | 说话人名称 |

**查询已注册说话人**

```
GET /speaker/list
```

**删除说话人**

```
DELETE /speaker/{name}
```

---

### 3. SER 情感识别

```
POST /ser/file
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 音频文件 |

**响应示例：**

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "emotion": "happy",
    "scores": {"angry": 0.05, "happy": 0.80, "sad": 0.05, "neutral": 0.10}
  }
}
```

---

### 4. 通用接口

**服务信息**

```
GET /
```

**健康检查**

```
GET /health
```

## 配置说明

### 完整配置项（config.yaml）

```yaml
server:
  host: "0.0.0.0"          # 监听地址
  port: 9090               # 监听端口
  log_level: "info"        # 日志级别
  temp_dir: "temp_files"   # 临时文件目录
  log_file: "logs/app.log" # 日志文件路径，留空则仅输出到控制台

models:
  asr:
    model: "..."           # ASR 主模型
    vad_model: "..."       # 语音活动检测模型
    punc_model: "..."      # 标点预测模型
    spk_model: "..."       # 说话人分离模型
  sv:
    model: "..."           # 声纹识别模型
  ser:
    model: "..."           # 情感识别模型
    enabled: true          # 是否启用 SER

hardware:
  device: "auto"           # auto 自动检测 / cuda / cpu
  ncpu: 4                  # CPU 线程数
  ngpu: 1                  # GPU 数量

concurrency:
  thread_pool_size: 4      # 线程池大小
  max_concurrent_asr: 2    # ASR 最大并发
  max_concurrent_sv: 2     # SV 最大并发
  max_concurrent_ser: 2    # SER 最大并发

speaker_db:
  path: "speaker_db.json"              # 声纹数据库路径
  reload_interval_sec: 5               # 数据库热加载间隔
  similarity_threshold: 0.5            # 声纹相似度阈值

task_queue:
  result_ttl_sec: 3600                 # 异步任务结果保留时间
  cleanup_interval_sec: 60             # 过期任务清理间隔

upload:
  max_file_size_mb: 100                # 上传文件大小限制（MB）
  allowed_extensions:                   # 允许的文件格式
    - ".wav"
    - ".mp3"
    - ".m4a"
    - ".flac"
    - ".ogg"
```

## 使用示例

### cURL

```bash
# 语音识别（含说话人识别）
curl -X POST http://localhost:9090/asr/file \
  -F "file=@audio.wav" \
  -F "identify_speakers=true"

# 注册说话人
curl -X POST http://localhost:9090/speaker/register \
  -F "file=@speaker_voice.wav" \
  -F "name=张三"

# 查询说话人列表
curl http://localhost:9090/speaker/list

# 情感识别
curl -X POST http://localhost:9090/ser/file \
  -F "file=@audio.wav"

# 异步转写
TASK_ID=$(curl -s -X POST http://localhost:9090/asr/task \
  -F "file=@long_audio.wav" | python -c "import sys,json;print(json.load(sys.stdin)['data']['task_id'])")

# 查询任务结果
curl http://localhost:9090/asr/task/$TASK_ID
```

### Python（requests）

```python
import requests

BASE = "http://localhost:9090"

# 语音识别
with open("audio.wav", "rb") as f:
    resp = requests.post(f"{BASE}/asr/file", files={"file": f}, data={"identify_speakers": "true"})
print(resp.json())

# 注册说话人
with open("speaker_voice.wav", "rb") as f:
    resp = requests.post(f"{BASE}/speaker/register", files={"file": f}, data={"name": "张三"})
print(resp.json())

# 情感识别
with open("audio.wav", "rb") as f:
    resp = requests.post(f"{BASE}/ser/file", files={"file": f})
print(resp.json())
```

## 项目结构

```
asr-server/
├── main.py              # 入口，FastAPI 应用与生命周期管理
├── config.py            # 配置加载与校验
├── config.yaml          # 配置文件
├── requirements.txt     # Python 依赖
├── models/
│   ├── base.py          # 模型推理基类（线程池 + 信号量）
│   ├── asr.py           # ASR 模型封装
│   ├── sv.py            # 声纹识别模型封装
│   └── ser.py           # 情感识别模型封装
├── routers/
│   ├── asr.py           # ASR 路由（同步 + 异步任务）
│   ├── speaker.py       # 声纹管理路由
│   └── ser.py           # 情感识别路由
├── schemas/
│   └── responses.py     # 响应模型定义
├── utils/
│   ├── audio.py         # 音频文件处理
│   ├── task_manager.py  # 异步任务队列
│   └── serialize.py     # 序列化工具
├── speaker_db.json      # 声纹数据库
└── temp_files/          # 临时文件目录
```

## 常见问题

**Q: 首次启动很慢？**
A: 首次启动会自动从 ModelScope 下载模型文件（约数 GB），下载完成后会缓存到本地。

**Q: 如何在纯 CPU 环境运行？**
A: 将 `config.yaml` 中 `hardware.device` 设为 `"cpu"`，`ngpu` 设为 `0`。

**Q: 上传文件大小限制？**
A: 默认 100 MB，可在 `config.yaml` 的 `upload.max_file_size_mb` 中调整。
