# 设计文档（DESIGN）

> 目标读者：需要理解服务架构、API 契约与设计权衡的开发者。
> 实现细节见 [IMPLEMENTATION.md](./IMPLEMENTATION.md)，部署见 [DEPLOYMENT.md](./DEPLOYMENT.md)。

## 1. 目标与非目标

### 目标
- 对外提供**纯同步**的图片文字识别 HTTP 接口（det + rec + cls）。
- 返回结构化结果（文本框多边形 + 文字 + 置信度），可降级为纯文本。
- Docker 单镜像部署，**离线可运行、版本确定**（模型构建时烘进镜像）。
- 结构与同仓库 `asr-server-simple` 风格一致，便于团队维护。

### 非目标（YAGNI，后续可扩展）
- PDF / 多页 / 多帧图渲染。
- 批量端点、异步任务队列（图片 OCR 单次耗时短，同步足够）。
- 版面分析、表格识别、公式识别、可搜索 PDF。
- GPU 推理（基镜像为 CPU）。

## 2. 架构概览

```
                    ┌──────────────────────────────────────────┐
  HTTP 客户端  ───►  │  Uvicorn (ASGI)  ── FastAPI app          │
                    │                          │                │
                    │   上传大小中间件 (413)    │                │
                    │   lifespan: 装载配置/日志/线程池/RapidOCR │
                    │                          ▼                │
                    │   routers/ocr.py:  POST /ocr             │
                    │     1. 校验扩展名/大小                    │
                    │     2. 存临时文件                         │
                    │     3. ModelRunner.run(推理)  ── 信号量   │
                    │     4. 组装 {code,msg,data}               │
                    │     5. 删除临时文件                        │
                    │                          │                │
                    │   models/ocr.py:  RapidOCR 引擎封装       │
                    │     └─ ThreadPoolExecutor (CPU 密集卸载)  │
                    └──────────────────────────────────────────┘
                                       │
                            烘进镜像的 PP-OCRv4 ONNX 模型（只读）
```

### 请求处理数据流
1. 客户端 `POST /ocr`（multipart，单张图片 + 可选 `detail` 表单字段）。
2. 上传大小中间件按 `content-length` 预判，超限直接返回 413。
3. 路由层校验扩展名 → 写入 `temp_files/`（容器内）。
4. `ModelRunner` 经 `asyncio.Semaphore` 取得并发槽位，把推理任务投递到 `ThreadPoolExecutor`，避免阻塞事件循环（ONNX 推理为 CPU 密集）。
5. RapidOCR 引擎产出 `boxes[N,4,2] / txts[N] / scores[N]`。
6. 路由层组装响应信封，按 `detail` 决定是否带 `boxes`。
7. `finally` 清理临时文件。

## 3. 接口契约

### 3.1 `POST /ocr` — 图片文字识别

**请求**（multipart/form-data）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片，`.png/.jpg/.jpeg/.bmp/.webp/.tif/.tiff`，≤ `upload.max_file_size_mb`（默认 20MB） |
| `detail` | string | 否 | `"true"`（默认）返回 `boxes`；`"false"` 仅返回拼接纯文本 |

**响应**（`detail=true`，默认）

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "text": "识别到的全部文本，按阅读顺序拼接",
    "boxes": [
      {
        "polygon": [[12, 34], [120, 34], [120, 60], [12, 60]],
        "text": "示例文字",
        "score": 0.987
      }
    ],
    "elapsed_ms": 234
  }
}
```

**响应**（`detail=false`）

```json
{
  "code": 0,
  "msg": "success",
  "data": { "text": "识别到的全部文本…", "elapsed_ms": 234 }
}
```

> 文本框 `polygon` 为 4 个角点的整数坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`，左上原点。`text` 拼接顺序为 RapidOCR 给出的阅读顺序（默认从上到下、从左到右）。

### 3.2 通用接口

| 方法 | 路径 | 响应 |
|------|------|------|
| GET | `/` | `{"service","version","status":"running","endpoints":{...}}` |
| GET | `/health` | `{"status":"healthy","model_loaded":true,"timestamp":"<iso8601>"}` |

## 4. 响应信封规范

统一沿用 `asr-server-simple` 的 `{code, msg, data}` 结构：

| `code` | 含义 | 对应 HTTP |
|--------|------|-----------|
| `0` | 成功 | 200 |
| `-1` | 业务失败（如未识别到文字） | 200 |

> 参数/格式错误、超限、推理异常**不**走该信封，而是标准 HTTP 错误（见下），便于客户端按状态码快速分流。

## 5. 错误映射

| 场景 | HTTP | body |
|------|------|------|
| 扩展名不支持 | 400 | `{"detail":"不支持的文件格式…"}` |
| 文件过大（content-length 超限） | 413 | `{"detail":"File too large"}` |
| 模型未初始化 | 500 | `{"detail":"OCR 模型未初始化"}` |
| 推理异常 | 500 | `{"detail":"OCR 推理失败: <msg>"}` |

所有异常经 `except HTTPException: raise` 透传，其余 `Exception` 记录完整堆栈后转 500。临时文件在 `finally` 中清理。

## 6. 并发模型

- **为什么需要并发控制**：ONNXRuntime 推理是 CPU 密集，直接在事件循环里跑会阻塞所有请求。
- **方案**（复用 `asr-server-simple` 的 `ModelRunner`）：
  - `ThreadPoolExecutor(max_workers=thread_pool_size)`：把同步推理调用 `loop.run_in_executor` 卸载到线程池。
  - `asyncio.Semaphore(max_concurrent_ocr)`：限制**同时在跑**的推理数，防止 CPU 过度超订。
- **ONNX 线程数**：设置 `intra_op_num_threads` 为物理核数（经 RapidOCR 参数 `EngineConfig.onnxruntime.intra_op_num_threads` 注入，`main.py` 启动时也设进 `OMP_NUM_THREADS`）。当 `max_concurrent_ocr>1` 时，单次推理的线程数应小于总核数以避免超订——配置示例见 [IMPLEMENTATION.md](./IMPLEMENTATION.md#配置)。
- **优雅关闭**：`lifespan` 退出时 `executor.shutdown(wait=False, cancel_futures=True)`。

## 7. 设计决策与权衡（内联说明，不单开 ADR）

| 决策 | 理由 |
|------|------|
| **CPU + ONNXRuntime** | 基镜像 `python3.12:base` 为 CPU；OCR 经 ONNXRuntime CPU 已足够快，且无需 CUDA，镜像更小更通用。 |
| **纯同步、无任务队列** | 图片 OCR 单次通常数百毫秒，同步即可；`asr-server-simple` 里的任务队列是面向长音频的，OCR 场景属过度设计。如未来需处理超大图/高并发背压，**新增** `/ocr/task` 异步端点是叠加式改动，不影响现有契约，故无需提前引入。 |
| **模型构建时烘进镜像** | 换取离线可运行与版本确定性；det+rec+cls 的 ONNX 体积可接受（数十 MB量级）。代价是镜像变大、换模型需重建——此决策较难逆转，详见 [DEPLOYMENT.md#模型烘进](./DEPLOYMENT.md#模型烘进)。 |
| **以 `asr-server-simple` 为结构参考** | 与同仓库既有服务风格统一、团队熟悉；在其基础上按 YAGNI 裁掉任务队列/SV/SER/进程池。 |

## 8. 安全边界
- 容器以 **root** 运行（base 默认即 root，省去 bind-mount 写权限的麻烦，代价是安全加固降级）。如需收紧：建非 root 用户 + `chown` 挂载目录。
- 上传大小与扩展名双重校验（中间件 + 路由）。
- 临时文件请求结束即删；不落盘持久化用户数据。
- 仅监听必要端口（8002）；不内置任何鉴权（如需，建议在反向代理/网关层处理）。
