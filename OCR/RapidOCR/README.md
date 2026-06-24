# RapidOCR Server

基于 [RapidOCR](https://github.com/RapidAI/RapidOCR)（统一包 `rapidocr` v3，ONNXRuntime CPU 后端）的图片文字识别服务，通过 FastAPI HTTP API 对外提供 **det + rec + cls** 纯文字识别能力，使用 Docker 部署。

## 功能概览

| 能力 | 说明 |
|------|------|
| **文字识别** | 上传图片，返回每个文本框的多边形坐标、文字内容、置信度；可开关仅返回纯文本 |
| **方向分类 (cls)** | 自动识别并校正倒置/旋转文本 |
| **中英文支持** | 默认 PP-OCRv4 中英文套件，跟随 `rapidocr` 包默认配置 |

> 非目标（YAGNI）：PDF/多页、批量、异步任务队列、版面/表格/公式识别、GPU 推理。后续如需可平滑扩展。

## 文档导航

完整设计与部署文档位于 `docs/`：

| 文档 | 内容 |
|------|------|
| [docs/DESIGN.md](docs/DESIGN.md) | 架构、API 契约、请求/响应规范、并发模型、设计决策与权衡 |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | 目录结构、各模块职责、配置 schema、RapidOCR 集成、编码约定 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 手动 commit 导出自包含镜像、setup、运行、环境变量、健康检查、故障排查 |

## 快速上手

镜像为**自包含**（源码 + 依赖 + 模型都在镜像内）。完整步骤见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)，核心三步：

```bash
# 1. 起 python3.12:base 容器，复制源码进去，跑 setup.sh（装依赖/烘模型/建用户）
docker run -it --name rapidocr-build -v "${PWD}:/hostsrc" python3.12:base bash
#    容器内执行：
#      mkdir -p /app && cp -a /hostsrc/. /app/ && cd /app && bash setup.sh && exit

# 2. 导出镜像（带 WORKDIR/USER/CMD）
docker commit --change 'WORKDIR /app' --change 'EXPOSE 8002' --change 'ENV OMP_NUM_THREADS=4' --change 'CMD ["python", "main.py"]' rapidocr-build rapidocr-server:latest && docker rm rapidocr-build

# 3. 运行 + 识别（日志映射到宿主 ./logs）
docker run -d --name rapidocr -p 8002:8002 -v "${PWD}/logs:/app/logs" --restart=unless-stopped rapidocr-server:latest
curl -X POST http://localhost:8002/ocr -F "file=@example.png"
```

## 接口速览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查 |
| POST | `/ocr` | 图片文字识别（multipart 上传，单图同步） |

详见 [docs/DESIGN.md](docs/DESIGN.md#接口契约)。

## 技术栈

- Python 3.12（基镜像 `python3.12:base`，CPU）
- `rapidocr`（v3 统一包）+ `onnxruntime`（CPU）
- FastAPI + Uvicorn
- Pydantic v2 + PyYAML（配置校验）
- 结构参考同仓库的 `ASR/FunASR/asr-server-simple`，按 YAGNI 裁剪
