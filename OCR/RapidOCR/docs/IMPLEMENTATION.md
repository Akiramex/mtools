# 实现文档（IMPLEMENTATION）

> 结构与约定参考同仓库 `ASR/FunASR/asr-server-simple`，按 YAGNI 裁剪。
> 设计依据见 [DESIGN.md](./DESIGN.md)，部署见 [DEPLOYMENT.md](./DEPLOYMENT.md)。

## 1. 目标目录结构

```
OCR/RapidOCR/
├── README.md              # 根说明 + 文档导航（已存在）
├── docs/                  # 本套文档（已存在）
│   ├── DESIGN.md
│   ├── IMPLEMENTATION.md
│   └── DEPLOYMENT.md
├── main.py                # 入口：FastAPI app、lifespan、上传中间件、/ 与 /health
├── config.py              # Pydantic 配置模型 + load_config + 环境变量覆盖
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖
├── Dockerfile             # 构建镜像（FROM python3.12:base）
├── .dockerignore
├── models/
│   ├── __init__.py
│   └── ocr.py             # OcrModel + ModelRunner（线程池 + 信号量）
├── routers/
│   ├── __init__.py
│   └── ocr.py             # POST /ocr 路由
├── schemas/
│   ├── __init__.py
│   └── responses.py       # HealthResponse / OcrBox / OcrData 等
└── utils/
    ├── __init__.py
    └── image.py           # 扩展名校验、临时文件落盘/清理
```

> 相对 `asr-server-simple` 的裁剪：去掉 `task_queue` 配置段、`utils/task_manager.py`、`speaker_db`、SV/SER 模型与路由、进程池分支；`models/asr.py` → `models/ocr.py`，`routers/asr.py` → `routers/ocr.py`，`utils/audio.py` → `utils/image.py`。

## 2. 各模块职责

### `main.py`
- `setup_logging(log_level, log_file)`：stdout + 可选文件，`force=True`，与 `asr-server-simple` 一致。
- `lifespan(app)`：
  1. `load_config(config.yaml)`（含环境变量覆盖）；
  2. 建日志；
  3. `ThreadPoolExecutor(max_workers=concurrency.thread_pool_size)`；
  4. 构造 `RapidOCR` 引擎，包进 `ModelRunner(model=engine, executor=_executor, semaphore=Semaphore(concurrency.max_concurrent_ocr))`，再包进 `OcrModel(runner=...)`；
  5. `routers.ocr.set_model(_ocr_model, _config)` 注入；
  6. `yield`；退出时关线程池。
- 上传大小中间件：`content-length` > `upload.max_file_size_mb` 时返回 413。
- `GET /`：服务信息 + 端点清单。
- `GET /health`（`response_model=HealthResponse`）：`model_loaded` 等。
- `uvicorn.run("main:app", host, port, log_level)`。

### `config.py`
- Pydantic 模型（见下「配置」）。
- `load_config(path)`：读 YAML → `AppConfig(**raw)`。
- **环境变量覆盖**：`load_config` 内对关键字段读取 `os.environ`（如 `OCR_HOST`、`OCR_PORT`、`OCR_LOG_LEVEL`、`OCR_MAX_FILE_SIZE_MB`、`OCR_THREAD_POOL_SIZE`、`OCR_MAX_CONCURRENT_OCR`），存在则覆盖 YAML 值。

### `models/ocr.py`
- `ModelRunner`（薄封装，照搬 `asr-server-simple` 思路）：
  ```python
  class ModelRunner:
      def __init__(self, model, executor, semaphore):
          self.model = model; self.executor = executor; self.semaphore = semaphore

      async def run(self, image_path: str):
          async with self.semaphore:
              loop = asyncio.get_running_loop()
              return await loop.run_in_executor(self.executor, self._infer, image_path)

      def _infer(self, image_path):
          return self.model(image_path)   # RapidOCR 引擎调用

      def shutdown(self): self.executor.shutdown(wait=False, cancel_futures=True)
  ```
- `OcrModel`：持有 `runner`，暴露 `async def recognize(path) -> OcrResult`，负责把引擎输出归一化成 `(boxes, txts, scores)`。
- `build_engine(config)`：按 `config.models.ocr` 拼 `params` 构造 `RapidOCR` 引擎（`intra_op_num_threads`、各 `*_model_path` / `rec_keys_path`），留空字段用包默认。`rapidocr` 延迟导入，便于先设 `OMP_NUM_THREADS`。

### `routers/ocr.py`
- 模块级 `set_model(ocr_model, config)` 注入依赖（与 `asr-server-simple` 同款手动注入）。
- `POST /ocr`：
  1. 取 `file: UploadFile`、`detail: str = "true"`；
  2. 校验扩展名（`config.upload.allowed_extensions`）→ 否则 400；
  3. `utils.image.save_upload(file)` → 临时路径；
  4. `result = await ocr_model.recognize(path)`；
  5. 组装 `text`（`" ".join(result.txts)` 或保留引擎顺序）、`boxes`（`detail` 为真时）、`elapsed_ms`；
  6. 成功 `{"code":0,"msg":"success","data":...}`；空结果 `{"code":-1,"msg":"未识别到文字",...}`；
  7. `finally` 删临时文件。
- `except HTTPException: raise`；其余 `Exception` 记堆栈后 500。

### `schemas/responses.py`
- `HealthResponse`：`status:str`、`model_loaded:bool`、`timestamp:str`。
- `OcrBox`：`polygon:list[list[int]]`、`text:str`、`score:float`。
- `OcrData`：`text:str`、`boxes:list[OcrBox]|None`、`elapsed_ms:int`。
- 响应信封统一用 `dict` 返回（与 `asr-server-simple` 一致），不强约束顶层模型以免限制错误形态。

### `utils/image.py`
- `allowed(filename, exts) -> bool`、`save_upload(upload) -> path`（`tempfile.NamedTemporaryFile(delete=False, suffix=ext)`）、`cleanup(path)`。

## 3. 配置

### `config.py`（Pydantic 模型）

```python
class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8002
    log_level: str = "info"
    temp_dir: str = "temp_files"
    log_file: str = "logs/app.log"

class OcrModelsConfig(BaseModel):
    det_model_path: str | None = None   # None = 用包默认模型（自动下载到安装目录的 models/）
    cls_model_path: str | None = None
    rec_model_path: str | None = None
    rec_keys_path: str | None = None    # 识别模型字典文件
    intra_op_num_threads: int = 4

class ModelsConfig(BaseModel):
    ocr: OcrModelsConfig

class ConcurrencyConfig(BaseModel):
    thread_pool_size: int = 4
    max_concurrent_ocr: int = 2

class UploadConfig(BaseModel):
    max_file_size_mb: int = 20
    allowed_extensions: list[str] = [".png",".jpg",".jpeg",".bmp",".webp",".tif",".tiff"]

class AppConfig(BaseModel):
    server: ServerConfig = ServerConfig()
    models: ModelsConfig
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    upload: UploadConfig = UploadConfig()
```

> 注意：**没有** `task_queue` 段（与 `asr-server-simple` 的关键区别）。

### `config.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8002
  log_level: "info"
  temp_dir: "temp_files"
  log_file: "logs/app.log"

models:
  ocr:
    det_model_path: null     # null → 用包默认模型；填路径走离线本地模型
    cls_model_path: null
    rec_model_path: null
    rec_keys_path: null      # 识别模型字典文件（用本地 rec 模型时一起填）
    intra_op_num_threads: 4

concurrency:
  thread_pool_size: 4
  max_concurrent_ocr: 2

upload:
  max_file_size_mb: 20
  allowed_extensions:
    - ".png"
    - ".jpg"
    - ".jpeg"
    - ".bmp"
    - ".webp"
    - ".tif"
    - ".tiff"
```

### 环境变量覆盖映射

| 环境变量 | 覆盖字段 |
|----------|----------|
| `OCR_HOST` | `server.host` |
| `OCR_PORT` | `server.port` |
| `OCR_LOG_LEVEL` | `server.log_level` |
| `OCR_LOG_FILE` | `server.log_file` |
| `OCR_THREAD_POOL_SIZE` | `concurrency.thread_pool_size` |
| `OCR_THREAD_POOL_SIZE` | `concurrency.thread_pool_size` |
| `OCR_MAX_CONCURRENT_OCR` | `concurrency.max_concurrent_ocr` |
| `OCR_MAX_FILE_SIZE_MB` | `upload.max_file_size_mb` |

## 4. RapidOCR v3 集成要点

- 安装：`pip install rapidocr onnxruntime`（统一包 v3 默认走 onnxruntime）。
- 引擎构造在 `models/ocr.py` 的 `build_engine(config)` 里，按配置拼 `params` 传给 `RapidOCR(params=...)`。常用键（前缀 `Det.`/`Cls.`/`Rec.` 同理）：
  - `EngineConfig.onnxruntime.intra_op_num_threads`：ONNX 推理线程数（**这才是真正生效的 key**，不是 `Global.intra_op_num_threads`）。
  - `Det.model_path` / `Cls.model_path` / `Rec.model_path`：本地 ONNX 模型文件路径（离线）。
  - `Rec.rec_keys_path`：识别模型字典文件（**onnxruntime 不需要**——字典已内嵌在 ONNX 里；仅 paddle/mnn 等引擎才需要）。
  - `Det.model_dir` / `Rec.model_dir`：仅 **Paddle** 多文件格式用目录；其余引擎用 `model_path`（`Cls.model_dir` 暂无效）。
  - `Det.lang_type` / `Det.model_type` / `Det.ocr_version` / `Det.engine_type`：按身份选模型（自动下载对应模型）。
- 调用：`result = engine(image_path)`（接受路径 / ndarray / bytes / PIL.Image）。
- 输出归一化：`result.boxes`（`[N,4,2]`）、`result.txts`（`tuple[str]`）、`result.scores`（`tuple[float]`）。三者按下标对齐；空结果时为空。
- **模型版本锁定**：`requirements.txt` 锁定 `rapidocr` 版本（默认模型随上游会变，如 3.9.0 默认已是 PP-OCRv6）；离线场景务必用 `*_model_path` 显式指定。

## 5. 并发实现细节

- `max_concurrent_ocr=2`、`intra_op_num_threads=4` 为默认；二者乘积不应远超物理核数，否则 ONNX 线程互相抢核导致延迟抖动。
- 调参建议：
  - 低并发、低延迟优先：`max_concurrent_ocr=1`，`intra_op_num_threads` 设为物理核数。
  - 高吞吐优先：`max_concurrent_ocr≈物理核数/2`，`intra_op_num_threads=2`。
- 信号量 + 线程池均通过配置可调，无需改代码。

## 6. requirements.txt

```
fastapi>=0.110
uvicorn>=0.29
rapidocr>=3.9.0       # v3 统一包；>=3.9 支持 PP-OCRv6 模型
onnxruntime>=1.17     # 需有 cp312 wheel
pydantic>=2.0
pyyaml>=6.0
python-multipart>=0.0.6
pillow>=10.0
```

> 基镜像为 Python 3.12，`onnxruntime` / `rapidocr` 对 cp312 支持成熟；如改回 3.13 需 `onnxruntime>=1.20`（cp313 wheel）。

## 7. 编码约定（对齐仓库现状）
- Python 3.12、类型注解、`X | None` 写法。
- 日志格式与 `asr-server-simple` 一致：`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`。
- 响应信封 `{code,msg,data}`；错误走 HTTPException。
- 中文用户提示文案（与 ASR 服务一致）。
