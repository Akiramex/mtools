# CosyVoice 技术分析文档索引

本目录包含对 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 项目的完整技术分析。

## 📚 文档列表

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | **完整技术分析** - 项目概述、架构、特性、性能、部署 |
| [API_REFERENCE.md](./API_REFERENCE.md) | **API快速参考** - 安装、使用、部署代码示例 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | **架构深度解析** - 监督式Token、LLM、Flow Matching详解 |

## 🎯 快速导航

### 想了解项目概览?
→ 阅读 [README.md](./README.md) 第1-2章

### 想快速上手使用?
→ 阅读 [API_REFERENCE.md](./API_REFERENCE.md)

### 想深入理解技术原理?
→ 阅读 [ARCHITECTURE.md](./ARCHITECTURE.md)

### 想看性能对比?
→ 阅读 [README.md](./README.md) 第6章

### 想部署到生产?
→ 阅读 [README.md](./README.md) 第7章 + [API_REFERENCE.md](./API_REFERENCE.md) 部署章节

## 🔑 核心要点

### CosyVoice 是什么?
阿里巴巴开源的多语言大模型TTS系统，支持零样本声音克隆。

### 核心创新
1. **监督式语义Token** - 首创的语音离散化方法
2. **流式推理** - 150ms低延迟
3. **多语言覆盖** - 9种语言 + 18种中文方言

### 推荐使用
- **通用**: Fun-CosyVoice3-0.5B
- **实时**: CosyVoice2-0.5B
- **克隆**: CosyVoice-300M

## 📊 性能亮点

| 指标 | Fun-CosyVoice3-0.5B-RL |
|------|------------------------|
| 中文CER | **0.81%** |
| 英文WER | **1.68%** |
| 说话人相似度 | 77.4% |
| 流式延迟 | 150ms |

## 🔗 官方资源

- [GitHub](https://github.com/FunAudioLLM/CosyVoice)
- [论文1](https://arxiv.org/abs/2407.05407) | [论文2](https://arxiv.org/abs/2412.10117)
- [Demo](https://funaudiollm.github.io/cosyvoice3/)
- [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)

---

> 分析时间: 2025-03-05  
> 分析工具: OpenCode Sisyphus
