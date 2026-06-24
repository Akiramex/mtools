# 性能优化记录（PERFORMANCE）

> 记录一次针对"手写大图 CPU 推理慢"的优化全过程。配套代码改动见下文「相关配置项」。

## 1. 背景
- 输入:手机拍摄的手写文字图（`handwrite.jpg`，大尺寸），onnxruntime CPU 推理。
- 现象:单张图 `elapsed_ms` ~1900ms，偏高（同类简单图正常应 300~800ms）。
- 目标:在 CPU 上把单图耗时压到可接受范围。

## 2. 诊断方法
在 `models/ocr.py` 的 `recognize` 里，把 RapidOCR 结果对象的 `elapse_list`（det/cls/rec 各阶段秒数）转成毫秒打到日志：

```
OCR 11 boxes, total Xms, per-stage(det/cls/rec) ms=[d, c, r]
```

诊断结论（实测）：
- **det 占 ~87%**（1670~1925ms），是绝对瓶颈；cls（~20ms）、rec（~200ms）正常。
- **不是冷启动**：第二次请求仍 ~1918ms（若仅首次慢属会话预热，需 lifespan 预热；本例不是）。

## 3. 优化过程（实测数据，11 个文本框）

| 步骤 | 改动 | total | det | cls | rec |
|------|------|------:|----:|----:|----:|
| 0 初始 | small 模型 + `max_side_len=2000`(默认) + 4 线程 | ~1920 | ~1670 | ~20 | ~210 |
| 1 | `max_side_len: 960` | ~800 | ~540 | ~24 | ~205 |
| 2 | `intra_op_num_threads: 16`（机器 48 核） | ~500 | ~250 | ~30 | ~180 |
| 3 | 换 `det_tiny` + `rec_tiny` | **~200** | ~110 | ~22 | ~45 |

**从 ~1920ms 降到 ~200ms，约 9.5×。** 连续 9 次请求耗时稳定（186~243ms），无退化。

## 4. 优化项详解

### 4.1 `max_side_len`（无损精度，最大收益）
- 默认 `Global.max_side_len = 2000`：det 把图最大边缩到 2000px 再跑。
- det 耗时 ≈ 与像素数成正比（面积）。`960/2000` 平方 ≈ 0.23，理论 ~4× 提速，实测 det 1670→540ms。
- **仅缩放，不损失模型精度**；只是小字可能因缩太小而漏检。
- 取值：手写大字 `960` 够用；密集/小字场景改 `1280`（仍 ~2.4× 提速）。

### 4.2 `intra_op_num_threads`（无损精度）
- 默认 4。本机 `nproc=48`（约 24 物理核），严重没吃满 → det 是 CPU 密集、高度可并行。
- 调到 **16** 后 det 540→250ms。
- **不是越多越好**：ONNX 这类 conv 模型到 ~物理核数附近拐点，再往上因线程开销收益递减甚至变慢。建议试 8 / 16 / 24 取 det 最小值。

### 4.3 tiny 模型（有损精度，换速度）
- `det_tiny`(1.8MB) / `rec_tiny`(4.5MB) 比 `small`(9.9/21MB) 快很多。
- 配合 4.1+4.2，det ~110ms、rec ~45ms，总 ~200ms。
- **务必核对识别质量**（尤其手写 rec）：拿样图对比 small vs tiny 的文字输出，看错字/漏字能否接受。det 若框不准（漏行/多框）可**单独**把 det 换回 small。

## 5. 推荐配置（按需求三选一）

| 目标 | det/rec | max_side_len | intra_op | 预期耗时 |
|------|---------|-------------|----------|---------|
| 准确优先 | small | 1280 | 物理核数 | ~600~900ms |
| 平衡 | small | 960 | 16(48核机) | ~500ms |
| 极速（精度可接受） | tiny | 960 | 16 | **~200ms** |

## 6. 相关配置项
均在 `config.yaml` 的 `models.ocr` 下（经 `models/ocr.py` 的 `build_engine` 传给 RapidOCR `params`）：

| config 字段 | 传入 RapidOCR 的 key | 说明 |
|-------------|---------------------|------|
| `max_side_len` | `Global.max_side_len` | det 输入图最大边像素（包默认 2000） |
| `intra_op_num_threads` | `EngineConfig.onnxruntime.intra_op_num_threads` | ONNX 推理线程数（env: `OCR_INTRA_OP_NUM_THREADS`） |
| `det_model_path` / `rec_model_path` | `Det.model_path` / `Rec.model_path` | 指向 `loacl_models/` 下 small 或 tiny 的 ONNX |

> 包内 `Global` 其它尺寸默认值：`max_side_len:2000`、`min_side_len:30`、`min_height:30`、`width_height_ratio:8`。

## 7. 进一步优化（可选）
- **关 cls（省 ~20ms）**：图不旋转时设 `Global.use_cls: false`（目前未接入 config，可在 `build_engine` 加）。
- **`enable_cpu_mem_arena: true`**：包默认 false，开启复用内存，小幅提升。
- **GPU（大招）**：机器有 GPU 时换 `onnxruntime-gpu` + CUDA + `use_cuda:true`，总耗时进 ~50~100ms。需另建镜像（装 CUDA 版 onnxruntime），改动较大。

## 8. 调参流程（供复现）
1. 改 `config.yaml`（源码挂载时直接改宿主文件）→ `docker restart rapidocr`，无需重建镜像。
2. 连发两次请求，看日志 `per-stage(det/cls/rec)`：第一次排除冷启动，第二次才是稳态。
3. 每次只动一个旋钮（max_side_len → 线程 → tiny），观察 det/rec 变化定拐点。
