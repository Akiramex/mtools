# 部署文档（DEPLOYMENT）

> 构建方式：**手动 commit**——开 `python3.12:base` 容器 → 跑 `setup.sh` 装好 → `docker commit` 导出**自包含镜像**（源码 + 依赖 + 模型全在镜像内）。
> 运行：`docker run` 直接跑，无需挂源码/模型。
> 仓库内 `Dockerfile` 为可复现的备选方案（见第 8 节）。
> 设计依据见 [DESIGN.md](./DESIGN.md)，结构见 [IMPLEMENTATION.md](./IMPLEMENTATION.md)。

## 1. 工作流总览

1. 起 `python3.12:base` 容器，把宿主源码**复制**进 `/app`（`-v` 挂载只为搬运，不进镜像）。
2. 容器内跑 `setup.sh`（以 root）：装系统库、装依赖、验证模型。
3. `docker commit --change ...` 导出 `rapidocr-server:latest`（带 `WORKDIR/EXPOSE/CMD`）。
4. `docker run` 运行导出的镜像。

> **关键**：源码必须**复制**进容器（`cp`），不能用 `-v` 挂载——挂载目录不会被 `docker commit` 收进镜像，导出的镜像里就不会有源码。

## 2. 详细步骤

### 2.1 起容器并放入源码

宿主（在 `OCR/RapidOCR` 目录下）：

```bash
docker run -it --name rapidocr-build -v "${PWD}:/hostsrc" python3.12:base bash
```

容器内（bash）：

```bash
mkdir -p /app && cp -a /hostsrc/. /app/ && cd /app
bash setup.sh
exit
```

> `${PWD}` 在 PowerShell / bash 都能展开为当前目录；Docker Desktop 会处理 Windows 路径。

### 2.2 导出镜像

宿主：

```bash
docker commit \
  --change 'WORKDIR /app' \
  --change 'EXPOSE 8002' \
  --change 'ENV OMP_NUM_THREADS=4' \
  --change 'CMD ["python", "main.py"]' \
  rapidocr-build rapidocr-server:latest

docker rm rapidocr-build
```

> `docker commit` 默认只固化文件系统，**不**带 base 镜像的运行时配置；`--change` 用来写入 `WORKDIR/CMD` 等，否则导出的镜像跑的是 `python3.12:base` 的默认 CMD。
> 镜像以 **root** 运行（base 默认即 root），不再单独建 appuser——省去 bind-mount 写权限的麻烦。

### 2.3 运行

```bash
docker run -d --name rapidocr \
  -p 8002:8002 \
  -v "${PWD}/logs:/app/logs" \
  --restart=unless-stopped \
  rapidocr-server:latest
```

| 参数 | 说明 |
|------|------|
| `-p 8002:8002` | 端口映射；避开同主机 ASR(9090)/TTS 常用端口 |
| `-v "${PWD}/logs:/app/logs"` | 日志持久化到宿主当前目录的 `logs/`（容器内 `/app/logs`） |
| `--restart=unless-stopped` | 异常退出自动拉起 |

> 镜像以 **root** 运行，无需挂源码或模型。只挂 `logs`（临时上传文件 `temp_files` 请求结束即删，不必持久化）。root 可直接写宿主挂载的 `logs` 目录，无权限问题。

## 3. setup.sh 做了什么

| 步骤 | 内容 |
|------|------|
| 装系统库 | `apt-get install libgl1 libglib2.0-0`（OpenCV 依赖，精简镜像默认没有，否则 `import cv2` 报 `libGL.so.1` 缺失） |
| 装依赖 | `pip install -r requirements.txt`（rapidocr / onnxruntime / fastapi 等） |
| 验证模型 | 跑 `build_engine(load_config("config.yaml"))`，按 `*_model_path` 加载 `loacl_models/` 下本地模型，确认路径/文件无误（**不联网下载**） |

> **本地模型**：`config.yaml` 把 `det_model_path`/`rec_model_path` 指向 `loacl_models/`（v6 det small、v6 rec small），`cls_model_path` 留 `null` 用 whl 内置默认 cls（v4 mobile，与默认输入尺寸匹配）。det/rec 随源码一起 `cp` 进 `/app`、随 `commit` 进镜像，运行时直接加载、不联网。
> **rec 字典（无需操心）**：onnxruntime 引擎的字符字典**已内嵌在 ONNX 模型里**（源码 `get_character_dict` 走 `session.have_key()` 分支，注释原话「onnx has character, other engine need dict_path」）。`rec_keys_path` 留空即可，**不需要 `ppocr_keys_v1.txt`**——该文件只有 paddle/mnn 等引擎才下载。
> **镜像瘦身**：`loacl_models/` 里用不上的模型也会被 `cp` 进镜像。只保留 `config.yaml` 实际引用的 3 个文件，可显著减小镜像（约 几百 MB → ~32 MB）。

## 4. 环境变量（运行期可调）

| 变量 | 默认 | 说明 |
|------|------|------|
| `OCR_HOST` | `0.0.0.0` | 监听地址 |
| `OCR_PORT` | `8002` | 监听端口（改后同步改 `-p`） |
| `OCR_LOG_LEVEL` | `info` | `debug/info/warning/error` |
| `OCR_LOG_FILE` | `logs/app.log` | 留空仅 stdout |
| `OCR_THREAD_POOL_SIZE` | `4` | 线程池大小 |
| `OCR_MAX_CONCURRENT_OCR` | `2` | 最大并发推理数 |
| `OCR_MAX_FILE_SIZE_MB` | `20` | 上传上限(MB) |
| `OCR_INTRA_OP_NUM_THREADS` | `4` | ONNX 推理线程数 |

## 5. 验证

```bash
curl http://localhost:8002/health
curl http://localhost:8002/
# 识别（结构化）
curl -X POST http://localhost:8002/ocr -F "file=@example.png"
# 识别（仅纯文本）
curl -X POST http://localhost:8002/ocr -F "file=@example.png" -F "detail=false"
```

查日志：`docker logs -f rapidocr`。

## 6. 故障排查

| 现象 | 排查 |
|------|------|
| `import cv2` 报 `libGL.so.1: cannot open shared object file` | OpenCV 缺系统库。`apt-get update && apt-get install -y libgl1 libglib2.0-0` 后重做（setup.sh 已含此步；已建好的容器里单独跑这条 apt 即可） |
| 启动报错找不到模型 / 路径错 | `config.yaml` 里 `*_model_path` 是相对 `/app` 的，确认文件随源码 `cp` 进了容器且路径拼写对；用 `docker run --rm rapidocr-server:latest ls /app/loacl_models` 核对 |
| 启动报错找不到模型 / 路径错 | `config.yaml` 里 `*_model_path` 是相对 `/app` 的，确认文件随源码 `cp` 进了容器且路径拼写对；用 `docker run --rm rapidocr-server:latest ls /app/loacl_models` 核对 |
| 识别乱码 / rec 无输出 | onnxruntime 下字典已内嵌，无需 keys 文件。先查 `*_model_path` 路径是否对、图片是否清晰；只有换成 paddle/mnn 引擎才需 `rec_keys_path` |
| 导出的镜像没起来 / 跑成 python REPL | commit 漏了 `--change 'CMD ["python", "main.py"]'`，导致沿用 base 默认 CMD |
| 镜像里没有源码 | 2.1 用了 `-v` 挂载而非 `cp` 复制（挂载不进 commit）；改成 `cp -a /hostsrc/. /app/` 后重做 |
| `docker commit` 找不到容器 | 容器名不是 `rapidocr-build`，用 `docker ps -a` 确认 |
| 8002 端口冲突 | 换 `-p` 映射端口 |
| 高并发下延迟抖动 | 调小 `OCR_MAX_CONCURRENT_OCR` / `OCR_INTRA_OP_NUM_THREADS`，避免 ONNX 线程超订 |

## 7. 更新源码 / 模型

改代码或换模型后，重跑第 2 节：起新容器复制新源码 → `setup.sh` → `docker commit`。建议带版本 tag 便于回滚：

```bash
docker commit --change '...' rapidocr-build rapidocr-server:v2
```

## 8. 备选：Dockerfile（可复现构建）

如需可复现、可纳入 CI 的构建，用仓库内现成的 `Dockerfile`，与手动 commit 等价（同样 root、同样烘模型、同样 CMD）：

```bash
docker build -t rapidocr-server:latest .
```

> 手动 commit 胜在快、可交互排错；Dockerfile 胜在可复现、可版本化。两者产出等价镜像。
