# Qwen3-TTS Apple Silicon

> 完全离线的高性能本地语音合成服务，专为 Mac M 系列芯片深度优化  
> 基于 [MLX](https://github.com/ml-explore/mlx) 框架 + Metal GPU 加速

---

## 📁 项目结构

```
Qwen3-TTS-MLX-Mac/
├── README.md              # 本文档
├── setup.sh               # 一键安装 + 下载模型
├── requirements.txt       # Python 依赖
├── demo.py                # 综合演示脚本 (CLI)
├── fastapi_server.py      # FastAPI + ProcessPool 生产级服务端
└── output/                # 生成的音频文件

~/Downloads/Qwen3-TTS-Models/   # 本地模型权重 (完全离线可用)
├── Base-8bit/                  # 基础朗读版 (~2.9GB)
├── VoiceDesign-8bit/           # 音色定制版 (~2.9GB)
└── CustomVoice-8bit/           # 多角色预设版 (~2.6GB)
```

---

## ⚡ 快速开始

```bash
# 一键安装 (创建环境 + 安装依赖 + 下载全部 3 个 8-bit 模型)
chmod +x setup.sh && ./setup.sh

# 激活环境
source .venv/bin/activate

# 运行演示
python demo.py basic --play          # 基础合成
python demo.py fast --play           # 4-bit 极速流式
python demo.py voicedesign --play    # 音色定制
python demo.py multilingual --play   # 中英日无缝切换
python demo.py customvoice --play    # 预设角色
python demo.py concurrent           # 4 路并发压测
python demo.py all --play            # 运行全部

# 启动 API 服务器
python fastapi_server.py
```

---

## 🧠 三种模型对比

| 模型 | 用途 | 关键参数 | 适用场景 |
|------|------|---------|---------|
| **Base** | 通用标准朗读 | `text` | 有声书、新闻播报、零样本克隆底模 |
| **VoiceDesign** | 文字描述→生成音色 | `text` + `instruct` | 角色扮演、个性化音色（如"沧桑老爷爷"） |
| **CustomVoice** | 预设说话人切换 | `text` + `voice` | 多角色对话、固定人设、多语言专声 |

### CustomVoice 预设说话人

| 名称 | 特点 |
|------|------|
| `vivian` | 英文女声 |
| `serena` | 英文女声 |
| `ono_anna` | 日文女声 |
| `sohee` | 韩文女声 |
| `uncle_fu` | 中文男声 |
| `ryan` | 英文男声 |
| `aiden` | 英文男声 |
| `eric` | 英文男声 |
| `dylan` | 英文男声 |

> ⚠️ **VoiceDesign** 必须提供 `instruct` 描述音色，不支持纯音频克隆  
> ⚠️ **CustomVoice** 必须使用上表中的预设 `voice` 名称

---

## 🚀 API 服务端

### 架构设计

```
┌──────────────────────────────────────────────┐
│           FastAPI (Uvicorn ASGI)              │
│          异步事件循环 · 永不阻塞               │
│                                              │
│  POST /generate ──► run_in_executor() ──┐    │
│  GET  /speakers                         │    │
│  GET  /models                           │    │
│  ┌──────────────────────────────────────▼──┐ │
│  │      ProcessPoolExecutor (N workers)    │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐      │ │
│  │  │Worker 1│ │Worker 2│ │  ...   │      │ │
│  │  │独立GPU │ │独立GPU │ │        │      │ │
│  │  └───┬────┘ └───┬────┘ └────────┘      │ │
│  └──────┴──────────┴──────────────────────┘ │
│              Apple Unified Memory            │
│           M-series Metal GPU Accel           │
└──────────────────────────────────────────────┘
```

### 4 个关键设计决策

1. **`spawn` 模式**：macOS 默认 `fork` 会复制 Metal GPU 句柄导致崩溃，必须用 `spawn`
2. **延迟导入 (Lazy Import)**：`mlx_audio` 仅在 worker 子进程内 import，保证 Metal 上下文隔离
3. **并发上限控制**：每 worker 峰值占用 ~5-6.5GB 统一内存，按机型配置 `MAX_WORKERS`
4. **异步委托**：`run_in_executor` 确保 GPU 运算不阻塞 HTTP 请求分发

### 启动

```bash
python fastapi_server.py
# 访问 Swagger UI: http://localhost:8000/docs
```

### API 接口

**POST /generate** — 合成语音（直接返回 WAV 文件）

```bash
# VoiceDesign 模式
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "model_type": "VoiceDesign", "instruct": "A warm female voice."}' \
  --output output.wav

# CustomVoice 模式
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "model_type": "CustomVoice", "voice": "ryan"}' \
  --output output.wav

# Base 模式
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "基础模型测试", "model_type": "Base"}' \
  --output output.wav
```

**GET /speakers** — 列出 CustomVoice 可用说话人  
**GET /models** — 列出已安装模型状态

---

## 📊 M1 实测性能

| 指标 | 单路 (4-bit) | 单路 (8-bit) | 4路并发 (4-bit) |
|------|-------------|-------------|----------------|
| RTF (实时率) | ~0.30x | ~0.79x | ~0.30x/路 |
| TTFB (首字延迟) | <100ms (流式) | ~2s | — |
| 峰值内存 | ~4.8GB | ~6.4GB | ~5.1GB/路 |
| 10秒音频耗时 | ~3s | ~8s | ~27s (4段总计) |

---

## 🔧 性能调优

| 场景 | 推荐配置 |
|------|---------|
| 追求极速+低延迟 | 4-bit 模型 + `stream=True` |
| 追求音质+稳定 | 8-bit 模型 + `stream=False` |
| 有声书批量生成 | Base-8bit + 多进程并发 |
| 实时对话 Agent | VoiceDesign-4bit + 流式 |
| 多角色场景 | CustomVoice-8bit + voice 切换 |

### 硬件建议

| 芯片 | MAX_WORKERS | 备注 |
|------|-------------|------|
| M1 16GB | 2 | 单路满速，双路不卡 |
| M1 Pro/Max 32GB | 4 | 并发利器 |
| M2/M3/M4 | 4+ | 带宽更高，RTF 更低 |

### 性能监控

```bash
# 终端 GPU 监控 (类似 nvtop)
pip install asitop && sudo asitop

# 或使用 Mac 内置: 活动监视器 → 窗口 → GPU 历史记录 (Cmd+4)
```

---

## ⚠️ 注意事项

- **绝对不要用多线程**调用 MLX，会触发 Metal Command Buffer 断言崩溃
- 首次运行需编译 Metal shader (~10-20s)，后续无此开销
- M1 内存带宽上限 68GB/s，4-bit 模型比 8-bit 快约 2x 的核心原因
- 未登录 HuggingFace 下载模型时限速明显，建议提前用 `setup.sh` 下载

---

## 📝 许可

本项目仅提供部署脚本和服务端代码。模型权重来自 [Qwen/Qwen3-TTS](https://huggingface.co/Qwen) 和 [mlx-community](https://huggingface.co/mlx-community)，请遵循其各自的许可协议。

*Built for Apple Silicon · Powered by MLX · 2026.03*
