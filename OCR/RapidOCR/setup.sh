#!/usr/bin/env bash
# 在 python3.12:base 容器内执行（以 root）：装系统库、装 python 依赖、验证模型。
# 前提：源码已复制到 /app（由宿主 docker run + cp 完成）。
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"

echo "[1/3] 装系统库（OpenCV 需要的 libGL / libglib，精简镜像默认没有）..."
apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

echo "[2/3] 安装 Python 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/3] 验证本地模型可加载（按 config.yaml 的 *_model_path 构造引擎，不联网下载）..."
python -c 'from config import load_config; from models.ocr import build_engine; build_engine(load_config("config.yaml"))'

echo ""
echo "setup 完成。退出容器后用 docker commit 导出镜像（见 docs/DEPLOYMENT.md）。"
