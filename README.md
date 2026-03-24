# Qwen3-TTS Apple Silicon

> 完全离线的高性能本地语音合成服务，专为 Mac M 系列芯片深度优化  
> 基于 [MLX](https://github.com/ml-explore/mlx) 框架 + Metal GPU 加速 · **OpenAI API 兼容**

---

## ⚡ 快速开始

```bash
# 1. 一键安装 (创建环境 + 安装依赖 + 下载全部 3 个 8-bit 模型)
chmod +x setup.sh && ./setup.sh

# 2. 激活环境
source .venv/bin/activate

# 3. 启动服务
python fastapi_server.py

# 4. 打开浏览器
open http://127.0.0.1:8080
```

### 开机自启 + 崩溃恢复

```bash
./tts-service.sh install     # 安装为 macOS 系统服务
./tts-service.sh status      # 查看状态
./tts-service.sh logs        # 查看日志
./tts-service.sh uninstall   # 卸载服务
```

> 基于 `launchd` 实现，进程崩溃 5 秒后自动重启

---

## 🎯 Web UI

访问 `http://127.0.0.1:8080/` 即可使用浏览器试听界面：

| Tab | 功能 | 说明 |
|-----|------|------|
| 🎤 预设说话人 | 9 种预设声音 | 下拉选择，一键生成 |
| 🎨 设计音色 | 自然语言描述音色 | 输入描述文本即可定制声音 |
| 🧬 语音克隆 | 零样本声音克隆 | 上传音频文件或浏览器直接录音 |

生成结果直接在浏览器播放，支持下载。

---

## 📡 OpenAI 兼容 API

完全兼容 OpenAI `/v1/audio/speech` 标准，可直接对接支持 OpenAI TTS 的应用。

### 端点列表

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 浏览器试听界面 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 交互文档 |
| `/v1/models` | GET | 模型列表 (含能力描述) |
| `/v1/audio/voices` | GET | 可用声音列表 (OpenAI↔原生双向映射) |
| `/v1/audio/speech` | POST | 语音合成 (JSON) |
| `/v1/audio/speech/clone` | POST | 语音克隆 (文件上传) |

### 三种模型

| 模型 ID | 用途 | 必填参数 |
|---------|------|----------|
| `qwen3-tts-customvoice` | 预设说话人 (默认) | `voice` |
| `qwen3-tts-voicedesign` | 自然语言设计音色 | `instructions` |
| `qwen3-tts-base` | 零样本语音克隆 | `ref_audio` 或文件上传 |

### 调用示例

```bash
# 预设说话人
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-customvoice", "input":"你好世界！Hello World!", "voice":"alloy"}' \
  --output speech.wav

# 设计音色
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-voicedesign", "input":"你好世界", "voice":"alloy", "instructions":"A warm, deep male voice with a calm tone."}' \
  --output speech.wav

# 语音克隆 (文件上传)
curl -X POST http://127.0.0.1:8080/v1/audio/speech/clone \
  -F "file=@reference.wav" \
  -F "input=用这个声音说这句话" \
  --output clone.wav
```

### Voice 双向映射

`voice` 字段同时支持 OpenAI 标准名和原生名：

| OpenAI | 原生名 | 性别 | 语言 |
|--------|--------|------|------|
| alloy | vivian | 女 | 中英 |
| echo | ryan | 男 | 英 |
| fable | serena | 女 | 英 |
| nova | ono_anna | 女 | 日英 |
| onyx | eric | 男 | 英 |
| shimmer | sohee | 女 | 韩英 |
| ash | aiden | 男 | 英 |
| coral | dylan | 男 | 英 |
| sage | uncle_fu | 男 | 中英 |

---

## 🧠 模型详情

| 模型 | 大小 | 用途 | 适用场景 |
|------|------|------|----------|
| **Base-8bit** | ~2.9GB | 通用朗读 + 零样本克隆 | 有声书、克隆特定声音 |
| **VoiceDesign-8bit** | ~2.9GB | 文字描述→生成音色 | 角色扮演、个性化声音 |
| **CustomVoice-8bit** | ~2.6GB | 预设说话人切换 | 多角色对话、固定人设 |

> ⚠️ 同一时刻只加载一个模型，防止 OOM。首次加载或切换模型约需 10 秒。

---

## 📁 项目结构

```
Qwen3-TTS-MLX-Mac/
├── README.md                    # 本文档
├── setup.sh                     # 一键安装 + 下载模型
├── requirements.txt             # Python 依赖
├── fastapi_server.py            # FastAPI 服务端 (含 Web UI)
├── demo.py                      # CLI 演示脚本
├── tts-service.sh               # macOS 服务管理脚本
├── com.local.qwen3-tts.plist    # launchd 配置
└── logs/                        # 服务日志

~/Downloads/Qwen3-TTS-Models/    # 本地模型权重
├── Base-8bit/
├── VoiceDesign-8bit/
└── CustomVoice-8bit/
```

---

## 🏗️ 架构设计

```
┌───────────────────────────────────────┐
│        FastAPI (Uvicorn ASGI)         │
│       异步事件循环 · 永不阻塞          │
│                                       │
│  GET  /           → Web UI (HTML)     │
│  POST /v1/audio/* → run_in_executor() │
│                         │             │
│  ┌──────────────────────▼───────┐     │
│  │  ProcessPoolExecutor (1 wkr) │     │
│  │  ┌─────────────────────────┐ │     │
│  │  │ Worker (spawn 隔离进程)  │ │     │
│  │  │ 模型缓存 + 自动切换     │ │     │
│  │  │ Metal GPU 加速          │ │     │
│  │  └─────────────────────────┘ │     │
│  └──────────────────────────────┘     │
│         Apple Unified Memory          │
└───────────────────────────────────────┘
```

**关键设计：**

1. **`spawn` 模式** — macOS `fork` 会复制 Metal GPU 句柄导致崩溃
2. **单 Worker** — 防止多模型同时加载 OOM，自动卸载旧模型
3. **延迟导入** — `mlx_audio` 仅在 worker 进程内 import，隔离 Metal 上下文
4. **完整 WAV** — 生成完整 WAV 文件 (非流式)，兼容所有播放器
5. **ffmpeg 转换** — 浏览器录音 (WebM) 自动转换为 WAV

---

## 📊 性能参考 (M1)

| 指标 | 8-bit | 4-bit |
|------|-------|-------|
| RTF (实时率) | ~0.79x | ~0.30x |
| 峰值内存 | ~6.4GB | ~4.8GB |
| 10 秒音频耗时 | ~8s | ~3s |

---

## ⚠️ 注意事项

- **不要用多线程**调用 MLX，会触发 Metal Command Buffer 断言崩溃
- 首次运行需编译 Metal shader (~10-20s)
- `ref_text` 参数必须是参考音频的**准确转录**，不确定就留空
- 录音功能需通过 `127.0.0.1` 或 `localhost` 访问 (浏览器安全限制)
- 需要系统安装 `ffmpeg`：`brew install ffmpeg`

---

## 📝 许可

本项目提供部署脚本和服务端代码。模型权重来自 [Qwen/Qwen3-TTS](https://huggingface.co/Qwen) 和 [mlx-community](https://huggingface.co/mlx-community)，请遵循其各自的许可协议。

*Built for Apple Silicon · Powered by MLX · 2026.03*
