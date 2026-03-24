"""
TTS FastAPI + ProcessPool 生产级服务端
本地语音合成服务

Features:
  - CORS 跨域支持
  - 流式音频返回 (StreamingResponse + multiprocessing.Queue)
  - Health Check 端点
  - 后台临时文件自动清理
"""
import os
import sys
import asyncio
import multiprocessing
import struct
import time
import glob
from contextlib import asynccontextmanager

multiprocessing.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# ──────────────────────────────────────────
# 配置区：根据你的硬件修改
# ──────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/Qwen3-TTS-Models")
OUTPUT_DIR = "/tmp/qwen3-tts-output"
MAX_WORKERS = 1   # 单 worker 进程，保证同时只加载一个模型
MAX_QUEUE = 16    # 最大并发排队数，超出返回 429

# 临时文件清理配置
CLEANUP_INTERVAL_SEC = 300    # 每 5 分钟扫描一次
CLEANUP_MAX_AGE_SEC = 600     # 文件超过 10 分钟即删除

# 默认流式采样率
DEFAULT_SAMPLE_RATE = 24000

# ──────────────────────────────────────────
# 模型注册表
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen3-tts-base": {
        "internal_type": "Base",
        "local_dir": "Base-8bit",
        "path": f"{MODELS_DIR}/Base-8bit",
        "description": "Zero-shot voice cloning via reference audio",
        "capabilities": ["voice_cloning", "multilingual"],
        "required_params": ["ref_audio"],
        "optional_params": ["ref_text"],
    },
    "qwen3-tts-voicedesign": {
        "internal_type": "VoiceDesign",
        "local_dir": "VoiceDesign-8bit",
        "path": f"{MODELS_DIR}/VoiceDesign-8bit",
        "description": "Design any voice via natural language instructions",
        "capabilities": ["voice_design", "multilingual", "instructions"],
        "required_params": ["instructions"],
        "optional_params": [],
    },
    "qwen3-tts-customvoice": {
        "internal_type": "CustomVoice",
        "local_dir": "CustomVoice-8bit",
        "path": f"{MODELS_DIR}/CustomVoice-8bit",
        "description": "9 built-in speaker presets, instant switching",
        "capabilities": ["preset_voices", "multilingual"],
        "required_params": ["voice"],
        "optional_params": [],
    },
}

# 内部使用的 MODEL_MAP（从注册表派生）
MODEL_MAP = {v["internal_type"]: v["path"] for v in MODEL_REGISTRY.values()}

# CustomVoice 可用说话人
CUSTOM_SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]

# ──────────────────────────────────────────
# 声音注册表（含 OpenAI 标准名 + 原生名）
# ──────────────────────────────────────────
VOICE_REGISTRY = [
    {"openai_name": "alloy",   "native_name": "vivian",   "gender": "female", "language": "en/zh"},
    {"openai_name": "echo",    "native_name": "ryan",     "gender": "male",   "language": "en"},
    {"openai_name": "fable",   "native_name": "serena",   "gender": "female", "language": "en"},
    {"openai_name": "nova",    "native_name": "ono_anna", "gender": "female", "language": "ja/en"},
    {"openai_name": "onyx",    "native_name": "eric",     "gender": "male",   "language": "en"},
    {"openai_name": "shimmer", "native_name": "sohee",    "gender": "female", "language": "ko/en"},
    {"openai_name": "ash",     "native_name": "aiden",    "gender": "male",   "language": "en"},
    {"openai_name": "coral",   "native_name": "dylan",    "gender": "male",   "language": "en"},
    {"openai_name": "sage",    "native_name": "uncle_fu", "gender": "male",   "language": "zh/en"},
]

# 双向映射：OpenAI 名 <-> 原生名均可作为输入
_VOICE_LOOKUP = {}
for _v in VOICE_REGISTRY:
    _VOICE_LOOKUP[_v["openai_name"]] = _v["native_name"]
    _VOICE_LOOKUP[_v["native_name"]] = _v["native_name"]  # 原生名直接通过


def resolve_voice(voice_input: str) -> str:
    """将 OpenAI 标准名或原生名统一解析为 speaker 名"""
    return _VOICE_LOOKUP.get(voice_input, voice_input)

# ──────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────
tts_pool = None
_start_time = None
_cleanup_task = None
_mp_manager = None
_tts_semaphore = None  # asyncio.Semaphore, 限制并发排队数


# ──────────────────────────────────────────
# Lifespan（替代已弃用的 on_event）
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_pool, _start_time, _cleanup_task, _mp_manager, _tts_semaphore
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tts_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    _mp_manager = multiprocessing.Manager()
    _start_time = time.time()
    _tts_semaphore = asyncio.Semaphore(MAX_QUEUE)
    _cleanup_task = asyncio.get_event_loop().create_task(_cleanup_loop())
    print(f"🚀 TTS Server Started | Workers: {MAX_WORKERS}")
    yield
    if _cleanup_task:
        _cleanup_task.cancel()
    if tts_pool:
        tts_pool.shutdown(wait=True)
    if _mp_manager:
        _mp_manager.shutdown()


# ──────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────
app = FastAPI(
    title="TTS API",
    version="3.0",
    description="OpenAI-compatible local TTS service",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OpenAITTSRequest(BaseModel):
    """对标 OpenAI /v1/audio/speech 请求格式"""
    model: str = "qwen3-tts-customvoice"  # 默认使用预设说话人模型
    input: str                             # 要合成的文本
    voice: str = "alloy"                   # OpenAI 标准名或 Qwen3 原生 speaker 名
    response_format: str = "wav"           # wav / pcm
    speed: float = 1.0                     # 语速（保留字段）
    instructions: Optional[str] = None     # 用于 qwen3-tts-voicedesign
    ref_audio: Optional[str] = None        # 用于 qwen3-tts-base
    ref_text: Optional[str] = None


def _stream_audio_task(request_data: dict, queue: multiprocessing.Queue):
    """
    子进程 worker（流式）：逐段将 PCM 音频 bytes 通过 Queue 传回主进程。
    结束后放入 None 作为哨兵。出错则放入 Exception 对象。
    内置模型缓存：同一进程只保留一个模型，切换时显式释放旧模型防止 OOM。
    """
    global _worker_cached_model, _worker_cached_type
    try:
        from mlx_audio.tts.utils import load_model
        import numpy as np
        import gc

        model_type = request_data.get("model_type", "VoiceDesign")
        model_path = MODEL_MAP.get(model_type, MODEL_MAP["VoiceDesign"])

        if not os.path.isdir(model_path):
            queue.put(Exception(f"模型目录不存在: {model_path}，请先运行 setup.sh"))
            return

        # 模型缓存：如果当前进程已加载同类型模型则复用，否则卸载旧模型再加载新模型
        try:
            cached_model = _worker_cached_model
            cached_type = _worker_cached_type
        except NameError:
            cached_model = None
            cached_type = None

        if cached_type == model_type and cached_model is not None:
            model = cached_model
        else:
            if cached_model is not None:
                del cached_model
                gc.collect()
                print(f"♻️ 卸载旧模型 {cached_type}，准备加载 {model_type}")
            model = load_model(model_path=model_path)
            _worker_cached_model = model
            _worker_cached_type = model_type
            print(f"📦 已加载模型 {model_type}")

        # 先把采样率通知主进程
        queue.put(("sample_rate", model.sample_rate))

        gen_kwargs = {
            "text": request_data["text"],
            "stream": True,
            "streaming_interval": 0.5,
            "verbose": False,
            "temperature": 0.7,
        }

        if model_type == "VoiceDesign":
            gen_kwargs["instruct"] = request_data.get("instruct")
        elif model_type == "CustomVoice":
            voice = request_data.get("voice", "vivian")
            if voice not in CUSTOM_SPEAKERS:
                queue.put(Exception(f"不支持的说话人 '{voice}'，可用: {CUSTOM_SPEAKERS}"))
                return
            gen_kwargs["voice"] = voice
        elif model_type == "Base":
            if request_data.get("ref_audio"):
                from mlx_audio.utils import load_audio
                gen_kwargs["ref_audio"] = load_audio(
                    request_data["ref_audio"],
                    sample_rate=model.sample_rate,
                )
                # ref_text 必须是参考音频的准确转录文本
                # 模型会将 ref_text + text 拼接为连续语音
                # 如果 ref_text 不准确会导致输出混乱
                ref_text = request_data.get("ref_text", "").strip()
                if ref_text:
                    gen_kwargs["ref_text"] = ref_text

        results = model.generate(**gen_kwargs)

        for result in results:
            audio_np = np.array(result.audio).flatten()
            pcm_int16 = (audio_np * 32767).astype(np.int16)
            queue.put(("audio", pcm_int16.tobytes()))

        queue.put(None)  # 哨兵：结束

    except Exception as e:
        queue.put(e)


def _make_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """
    构造流式 WAV header（data size 设为 0xFFFFFFFF 表示未知长度）。
    仅用于 /stream 端点的实时流式播放。
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF
    riff_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        riff_size & 0xFFFFFFFF,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size & 0xFFFFFFFF,
    )
    return header


def _make_complete_wav(pcm_data: bytes, sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """
    构造完整 WAV 文件（带正确的 data size），可被所有播放器正常播放。
    用于 /v1/audio/speech 端点。
    """
    data_size = len(pcm_data)
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    riff_size = 36 + data_size

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        riff_size,
        b'WAVE',
        b'fmt ',
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size,
    )
    return header + pcm_data


# ──────────────────────────────────────────
# 临时文件清理
# ──────────────────────────────────────────
async def _cleanup_loop():
    """后台定期清理 OUTPUT_DIR 中过期的 WAV 文件"""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
            if not os.path.isdir(OUTPUT_DIR):
                continue
            now = time.time()
            cleaned = 0
            for f in glob.glob(os.path.join(OUTPUT_DIR, "tts_*.wav")):
                try:
                    if now - os.path.getmtime(f) > CLEANUP_MAX_AGE_SEC:
                        os.remove(f)
                        cleaned += 1
                except OSError:
                    pass
            if cleaned:
                print(f"🧹 清理了 {cleaned} 个过期临时文件")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️ 清理任务异常: {e}")




# ──────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────

# ─── Health Check ───
@app.get("/health")
async def health_check():
    """健康检查：返回服务状态、模型可用性、worker 数量"""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "models": {mid: os.path.isdir(info["path"]) for mid, info in MODEL_REGISTRY.items()},
        "workers": MAX_WORKERS,
    }


# ─── Web UI ───
@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """浏览器端 TTS 试听界面"""
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TTS Playground</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',-apple-system,system-ui,sans-serif;background:#0f0f1a;color:#e0e0f0;min-height:100vh;display:flex;justify-content:center;padding:30px 16px}
.container{max-width:720px;width:100%}
h1{font-size:1.8rem;text-align:center;margin-bottom:6px;background:linear-gradient(135deg,#7c3aed,#2563eb,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{text-align:center;color:#888;font-size:.85rem;margin-bottom:28px}
.tabs{display:flex;gap:6px;margin-bottom:20px}
.tab{flex:1;padding:10px 0;border:none;border-radius:10px;cursor:pointer;font-size:.85rem;font-weight:600;background:#1a1a2e;color:#888;transition:all .2s}
.tab.active{background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;box-shadow:0 4px 20px rgba(99,60,255,.3)}
.tab:hover:not(.active){background:#252540;color:#bbb}
.card{background:#1a1a2e;border-radius:14px;padding:24px;border:1px solid #2a2a45}
label{display:block;font-size:.8rem;color:#999;margin-bottom:6px;font-weight:500}
textarea,input[type=text],select{width:100%;padding:10px 14px;border-radius:8px;border:1px solid #333;background:#12121f;color:#e0e0f0;font-size:.9rem;outline:none;transition:border .2s}
textarea:focus,input:focus,select:focus{border-color:#7c3aed}
textarea{resize:vertical;min-height:80px;font-family:inherit}
select{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23888' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center}
.row{display:flex;gap:12px;margin-bottom:14px}
.row>div{flex:1}
.field{margin-bottom:14px}
.btn{width:100%;padding:12px;border:none;border-radius:10px;font-size:.95rem;font-weight:600;cursor:pointer;background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;transition:all .25s;margin-top:6px}
.btn:hover{transform:translateY(-1px);box-shadow:0 6px 24px rgba(99,60,255,.4)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}
.btn:disabled::after{content:'  ⏳'}
.result{margin-top:18px;display:none}
.result.show{display:block}
.result audio{width:100%;margin-top:8px;border-radius:8px}
.result .time{font-size:.75rem;color:#666;margin-top:4px;text-align:right}
.info{font-size:.75rem;color:#555;margin-top:6px;line-height:1.5}
.upload-area{border:2px dashed #333;border-radius:10px;padding:20px;text-align:center;cursor:pointer;transition:all .2s;position:relative}
.upload-area:hover{border-color:#7c3aed;background:#1f1f35}
.upload-area input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upload-area .icon{font-size:2rem;margin-bottom:6px}
.upload-area .name{color:#7c3aed;font-size:.85rem;margin-top:4px}
.hidden{display:none}
.audio-source{display:flex;gap:10px;margin-bottom:10px}
.audio-source .src-btn{flex:1;padding:8px;border:none;border-radius:8px;cursor:pointer;font-size:.8rem;font-weight:600;background:#252540;color:#888;transition:all .2s}
.audio-source .src-btn.active{background:#2a2a55;color:#7c3aed;border:1px solid #7c3aed}
.rec-area{text-align:center;padding:24px}
.rec-btn{width:64px;height:64px;border-radius:50%;border:3px solid #444;background:#1a1a2e;cursor:pointer;display:flex;align-items:center;justify-content:center;margin:0 auto 10px;transition:all .2s}
.rec-btn .dot{width:24px;height:24px;border-radius:50%;background:#ef4444;transition:all .2s}
.rec-btn.recording{border-color:#ef4444;animation:pulse 1.5s infinite}
.rec-btn.recording .dot{border-radius:4px;width:18px;height:18px}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 12px rgba(239,68,68,0)}}
.rec-timer{font-size:.85rem;color:#999;font-variant-numeric:tabular-nums}
.rec-preview{margin-top:10px}
.rec-preview audio{width:100%}
</style>
</head>
<body>
<div class="container">
<h1>🔊 TTS Playground</h1>
<p class="sub">本地离线 · OpenAI API 兼容</p>

<div class="tabs">
  <button class="tab active" onclick="switchTab('cv')">🎤 预设说话人</button>
  <button class="tab" onclick="switchTab('vd')">🎨 设计音色</button>
  <button class="tab" onclick="switchTab('base')">🧬 语音克隆</button>
</div>

<!-- CustomVoice -->
<div class="card" id="panel-cv">
  <div class="field">
    <label>要合成的文本</label>
    <textarea id="cv-text">大家好，Welcome to the TTS playground! 这是一个中英文混合语音合成的示例，you can try different voices here.</textarea>
  </div>
  <div class="field">
    <label>选择说话人</label>
    <select id="cv-voice">
      <option value="alloy">alloy (女, 中英)</option>
      <option value="echo">echo (男, 英)</option>
      <option value="fable">fable (女, 英)</option>
      <option value="nova">nova (女, 日/英)</option>
      <option value="onyx">onyx (男, 英)</option>
      <option value="shimmer">shimmer (女, 韩/英)</option>
      <option value="ash">ash (男, 英)</option>
      <option value="coral">coral (男, 英)</option>
      <option value="sage">sage (男, 中英)</option>
    </select>
  </div>
  <button class="btn" id="cv-btn" onclick="genCustomVoice()">生成语音</button>
  <div class="result" id="cv-result">
    <label>🎧 合成结果</label>
    <audio id="cv-audio" controls></audio>
    <div class="time" id="cv-time"></div>
  </div>
</div>

<!-- VoiceDesign -->
<div class="card hidden" id="panel-vd">
  <div class="field">
    <label>要合成的文本</label>
    <textarea id="vd-text">欢迎来到 AI 语音合成的世界。You can design any voice simply by describing it in natural language. 让我们开始吧！</textarea>
  </div>
  <div class="field">
    <label>音色描述 (instructions)</label>
    <textarea id="vd-inst" style="min-height:60px">A deep, warm, authoritative male voice with a slight British accent, speaking at a measured and elegant pace.</textarea>
  </div>
  <button class="btn" id="vd-btn" onclick="genVoiceDesign()">生成语音</button>
  <div class="result" id="vd-result">
    <label>🎧 合成结果</label>
    <audio id="vd-audio" controls></audio>
    <div class="time" id="vd-time"></div>
  </div>
</div>

<!-- Base (Clone) -->
<div class="card hidden" id="panel-base">
  <div class="field">
    <label>要合成的文本</label>
    <textarea id="base-text">This is a zero-shot voice cloning demo. The output voice will match the reference audio you uploaded.</textarea>
  </div>
  <div class="field">
    <label>参考音频</label>
    <div class="audio-source">
      <button class="src-btn active" onclick="switchSrc('upload')">📁 上传文件</button>
      <button class="src-btn" onclick="switchSrc('record')">🎙️ 录音</button>
    </div>
    <div id="src-upload">
      <div class="upload-area" id="upload-area">
        <input type="file" id="base-file" accept="audio/*" onchange="onFileSelect(this)">
        <div class="icon">📁</div>
        <div>点击或拖拽上传参考音频 (wav/mp3/flac)</div>
        <div class="name" id="file-name"></div>
      </div>
    </div>
    <div id="src-record" class="hidden">
      <div class="rec-area">
        <button class="rec-btn" id="rec-btn" onclick="toggleRecord()">
          <div class="dot"></div>
        </button>
        <div class="rec-timer" id="rec-timer">点击开始录音</div>
        <div class="rec-preview hidden" id="rec-preview">
          <audio id="rec-audio" controls></audio>
        </div>
      </div>
    </div>
  </div>
  <div class="field">
    <label>参考音频的准确转录文本（必须与音频内容完全一致，否则留空）</label>
    <input type="text" id="base-ref-text" placeholder="输入参考音频中说话人所说的原文，不确定就留空...">
  </div>
  <button class="btn" id="base-btn" onclick="genClone()">生成语音</button>
  <div class="result" id="base-result">
    <label>🎧 克隆结果</label>
    <audio id="base-audio" controls></audio>
    <div class="time" id="base-time"></div>
  </div>
</div>

<p class="info" style="margin-top:16px;text-align:center">⚠️ 同一时刻只加载一个模型，首次或切换模型需约 10s 加载时间&nbsp;&nbsp;|&nbsp;&nbsp;<a href="/docs" style="color:#7c3aed">API 文档</a></p>
</div>

<script>
function switchTab(id) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const panels = ['cv','vd','base'];
    t.classList.toggle('active', panels[i]===id);
  });
  ['cv','vd','base'].forEach(p => {
    document.getElementById('panel-'+p).classList.toggle('hidden', p!==id);
  });
}

function onFileSelect(input) {
  const name = input.files[0]?.name || '';
  document.getElementById('file-name').textContent = name ? '✅ ' + name : '';
}

async function doGenerate(btn, audioEl, timeEl, resultEl, fetchFn) {
  btn.disabled = true;
  const btnText = btn.textContent.replace('  ⏳','');
  resultEl.classList.remove('show');
  const t0 = Date.now();
  try {
    const blob = await fetchFn();
    const url = URL.createObjectURL(blob);
    audioEl.src = url;
    audioEl.load();
    resultEl.classList.add('show');
    timeEl.textContent = '耗时 ' + ((Date.now()-t0)/1000).toFixed(1) + 's';
  } catch(e) {
    alert('生成失败: ' + (e.message || e));
  } finally {
    btn.disabled = false;
    btn.textContent = btnText;
  }
}

function genCustomVoice() {
  doGenerate(
    document.getElementById('cv-btn'),
    document.getElementById('cv-audio'),
    document.getElementById('cv-time'),
    document.getElementById('cv-result'),
    async () => {
      const r = await fetch('/v1/audio/speech', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({model:'qwen3-tts-customvoice', input:document.getElementById('cv-text').value, voice:document.getElementById('cv-voice').value})
      });
      if(!r.ok) throw new Error((await r.json()).detail);
      return r.blob();
    }
  );
}

function genVoiceDesign() {
  doGenerate(
    document.getElementById('vd-btn'),
    document.getElementById('vd-audio'),
    document.getElementById('vd-time'),
    document.getElementById('vd-result'),
    async () => {
      const r = await fetch('/v1/audio/speech', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({model:'qwen3-tts-voicedesign', input:document.getElementById('vd-text').value, voice:'alloy', instructions:document.getElementById('vd-inst').value})
      });
      if(!r.ok) throw new Error((await r.json()).detail);
      return r.blob();
    }
  );
}

function genClone() {
  const file = document.getElementById('base-file').files[0] || recordedBlob;
  if(!file){alert('请先上传或录制参考音频');return}
  doGenerate(
    document.getElementById('base-btn'),
    document.getElementById('base-audio'),
    document.getElementById('base-time'),
    document.getElementById('base-result'),
    async () => {
      const fd = new FormData();
      fd.append('file', file, file.name || 'recording.wav');
      fd.append('input', document.getElementById('base-text').value);
      fd.append('ref_text', document.getElementById('base-ref-text').value);
      const r = await fetch('/v1/audio/speech/clone', {method:'POST', body:fd});
      if(!r.ok) throw new Error((await r.json()).detail);
      return r.blob();
    }
  );
}

// ── 录音相关 ──
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recStartTime = 0;
let recTimer = null;

function switchSrc(mode) {
  document.querySelectorAll('.src-btn').forEach((b,i) => b.classList.toggle('active', (i===0&&mode==='upload')||(i===1&&mode==='record')));
  document.getElementById('src-upload').classList.toggle('hidden', mode!=='upload');
  document.getElementById('src-record').classList.toggle('hidden', mode!=='record');
}

async function toggleRecord() {
  const btn = document.getElementById('rec-btn');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    btn.classList.remove('recording');
    clearInterval(recTimer);
    document.getElementById('rec-timer').textContent = '录音完成';
    return;
  }
  // 兼容性检查
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    // 尝试旧版 API
    const legacyGetUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!legacyGetUserMedia) {
      alert('当前浏览器不支持录音功能。\\n请使用 Chrome/Edge/Safari 并通过 localhost 或 127.0.0.1 访问，或改用 HTTPS。');
      return;
    }
    // 包装旧版 API 为 Promise
    navigator.mediaDevices = navigator.mediaDevices || {};
    navigator.mediaDevices.getUserMedia = (constraints) => new Promise((resolve, reject) => legacyGetUserMedia.call(navigator, constraints, resolve, reject));
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    recordedChunks = [];
    // 选择浏览器支持的格式
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : (MediaRecorder.isTypeSupported('audio/mp4') ? 'audio/mp4' : '');
    mediaRecorder = mimeType ? new MediaRecorder(stream, {mimeType}) : new MediaRecorder(stream);
    const ext = mimeType.includes('mp4') ? '.m4a' : '.webm';
    mediaRecorder.ondataavailable = e => { if(e.data.size>0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t=>t.stop());
      recordedBlob = new Blob(recordedChunks, {type: mediaRecorder.mimeType});
      recordedBlob.name = 'recording' + ext;
      const url = URL.createObjectURL(recordedBlob);
      const preview = document.getElementById('rec-preview');
      document.getElementById('rec-audio').src = url;
      preview.classList.remove('hidden');
      document.getElementById('base-file').value = '';
      document.getElementById('file-name').textContent = '';
    };
    mediaRecorder.start();
    btn.classList.add('recording');
    recStartTime = Date.now();
    updateRecTimer();
    recTimer = setInterval(updateRecTimer, 100);
  } catch(e) {
    alert('无法访问麦克风: ' + e.message + '\\n\\n提示: 请确保通过 localhost 或 127.0.0.1 访问本页面。');
  }
}

function updateRecTimer() {
  const s = ((Date.now()-recStartTime)/1000).toFixed(1);
  document.getElementById('rec-timer').textContent = '🔴 录音中 ' + s + 's';
}
</script>
</body>
</html>"""


# ─── GET /v1/models ───
@app.get("/v1/models")
async def get_openai_models():
    """
    返回 OpenAI 格式的模型列表。
    每个模型包含 capabilities、description、required_params。
    只返回本地实际存在的模型。
    """
    now = int(time.time())
    data = []
    for model_id, info in MODEL_REGISTRY.items():
        if os.path.isdir(info["path"]):
            data.append({
                "id": model_id,
                "object": "model",
                "created": now,
                "owned_by": "local",
                "description": info["description"],
                "capabilities": info["capabilities"],
                "required_params": info["required_params"],
                "optional_params": info.get("optional_params", []),
            })
    return {"object": "list", "data": data}


# ─── GET /v1/audio/voices ───
@app.get("/v1/audio/voices")
async def get_openai_voices():
    """
    列出所有可用声音。
    每个声音包含标准名、性别、支持语言。
    """
    voices = []
    for v in VOICE_REGISTRY:
        voices.append({
            "voice_id": v["openai_name"],
            "name": v["openai_name"],
            "gender": v["gender"],
            "language": v["language"],
            "preview_url": None,
        })
    return {"voices": voices}


# ─── POST /v1/audio/speech ───
@app.post("/v1/audio/speech")
async def openai_audio_speech(
    request: OpenAITTSRequest = Body(
        ...,
        openapi_examples={
            "预设说话人": {
                "summary": "使用预设说话人",
                "description": "9 种内置说话人，voice 可用标准名（alloy/echo/fable/nova/onyx/shimmer/ash/coral/sage）",
                "value": {
                    "model": "qwen3-tts-customvoice",
                    "input": "你好，这是预设说话人模式。Welcome to TTS!",
                    "voice": "alloy",
                },
            },
            "自定义音色": {
                "summary": "通过自然语言描述设计任意音色",
                "description": "用 instructions 字段描述你想要的声音特征，模型会据此生成",
                "value": {
                    "model": "qwen3-tts-voicedesign",
                    "input": "Welcome to the world of AI voice synthesis. You can design any voice simply by describing it.",
                    "voice": "alloy",
                    "instructions": "A deep, warm, authoritative male voice with a slight British accent, speaking at a measured and elegant pace.",
                },
            },
            "中文女主播": {
                "summary": "设计一个活力中文女主播音色",
                "description": "instructions 支持英文描述，模型会根据描述生成对应音色",
                "value": {
                    "model": "qwen3-tts-voicedesign",
                    "input": "亲爱的听众朋友们，欢迎收听今天的节目！今天我们来聊聊人工智能语音合成技术。",
                    "voice": "alloy",
                    "instructions": "A cheerful, energetic young Chinese female radio host voice, speaking with warmth and enthusiasm.",
                },
            },
            "零样本语音克隆": {
                "summary": "提供参考音频进行声音克隆",
                "description": "提供 ref_audio（参考音频文件路径）和可选的 ref_text（参考音频对应文本），模型会克隆该声音。",
                "value": {
                    "model": "qwen3-tts-base",
                    "input": "This is a zero-shot voice cloning demo. The output voice will match the reference audio.",
                    "voice": "alloy",
                    "ref_audio": "/path/to/reference_audio.wav",
                    "ref_text": "The transcript of the reference audio goes here.",
                },
            },
        },
    )
):
    """
    OpenAI 兼容语音合成接口。

    ## 三种模型

    | 模型 ID | 用途 | 必填参数 |
    |---------|------|----------|
    | `qwen3-tts-customvoice` | 9 种预设说话人 | `voice` |
    | `qwen3-tts-voicedesign` | 自然语言设计音色 | `instructions` |
    | `qwen3-tts-base` | 零样本语音克隆 | `ref_audio` |

    ## Voice 可用值

    `voice` 字段支持以下标准名：
    alloy / echo / fable / nova / onyx / shimmer / ash / coral / sage

    > ⚠️ 同一时刻只加载一个模型，切换模型时自动卸载旧模型防止 OOM。
    """

    # ── 1. 解析模型 ──
    model_id = request.model
    if model_id not in MODEL_REGISTRY:
        available = [k for k, v in MODEL_REGISTRY.items() if os.path.isdir(v["path"])]
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型 '{model_id}'，可用模型: {available}"
        )

    registry = MODEL_REGISTRY[model_id]
    internal_type = registry["internal_type"]
    model_path = registry["path"]

    if not os.path.isdir(model_path):
        raise HTTPException(
            status_code=400,
            detail=f"模型 '{model_id}' 目录不存在: {model_path}，请先运行 setup.sh"
        )

    # ── 2. 双向解析声音名 ──
    voice = resolve_voice(request.voice)

    # ── 3. 按模型类型构建内部请求并校验必要参数 ──
    internal_req = {
        "text": request.input,
        "model_type": internal_type,
    }

    if internal_type == "CustomVoice":
        if voice not in CUSTOM_SPEAKERS:
            all_accepted = [v["openai_name"] for v in VOICE_REGISTRY] + CUSTOM_SPEAKERS
            raise HTTPException(
                status_code=422,
                detail=f"不支持的声音 '{request.voice}'，可用声音: {all_accepted}"
            )
        internal_req["voice"] = voice

    elif internal_type == "VoiceDesign":
        if not request.instructions:
            raise HTTPException(
                status_code=422,
                detail="VoiceDesign 模型需要提供 'instructions' 参数来描述期望的音色，"
                       "例如: 'A warm, mature male voice with a calm and professional tone.'"
            )
        internal_req["instruct"] = request.instructions

    elif internal_type == "Base":
        if not request.ref_audio:
            raise HTTPException(
                status_code=422,
                detail="Base 模型需要提供 'ref_audio' 参数（参考音频文件路径）用于零样本语音克隆，"
                       "同时建议提供 'ref_text'（参考音频对应文本）"
            )
        internal_req["ref_audio"] = request.ref_audio
        if request.ref_text and request.ref_text.strip():
            internal_req["ref_text"] = request.ref_text.strip()

    # ── 4. 生成音频并收集全部 PCM 数据 ──
    # MAX_WORKERS=1 保证单 worker，模型缓存机制保证同时只加载一个模型
    # Semaphore 限制最多 MAX_QUEUE 个请求同时排队
    if not _tts_semaphore.locked() or _tts_semaphore._value > 0:
        pass  # 有空位
    else:
        raise HTTPException(status_code=429, detail=f"服务繁忙，当前已有 {MAX_QUEUE} 个请求排队，请稍后重试")
    async with _tts_semaphore:
        q = _mp_manager.Queue()
        loop = asyncio.get_running_loop()
        loop.run_in_executor(tts_pool, _stream_audio_task, internal_req, q)

        # 收集所有 PCM 数据块，构建完整 WAV（非流式，确保可播放）
        audio_chunks = []
        sample_rate = DEFAULT_SAMPLE_RATE
        while True:
            item = await loop.run_in_executor(None, q.get)
            if isinstance(item, Exception):
                raise HTTPException(status_code=500, detail=str(item))
            if item is None:
                break
            tag, data = item
            if tag == "sample_rate":
                sample_rate = data
            elif tag == "audio":
                audio_chunks.append(data)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="音频生成失败：未产生任何音频数据")

        pcm_data = b"".join(audio_chunks)

    # ── 5. 构建响应 ──
    if request.response_format == "pcm":
        return Response(content=pcm_data, media_type="audio/pcm")

    wav_data = _make_complete_wav(pcm_data, sample_rate=sample_rate)
    return Response(
        content=wav_data,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


# ─── POST /v1/audio/speech/clone ───
@app.post("/v1/audio/speech/clone")
async def openai_audio_speech_clone(
    file: UploadFile = File(..., description="参考音频文件（支持 wav/mp3/flac 等格式）"),
    input: str = Form(..., description="要合成的文本"),
    ref_text: str = Form("", description="参考音频的准确转录文本（必须与音频内容完全一致，否则留空）"),
    response_format: str = Form("wav", description="输出格式：wav / pcm"),
):
    """
    零样本语音克隆（支持上传音频文件）。

    上传一段参考音频，模型会克隆该声音来朗读你指定的文本。
    使用 Base 模型。

    - **file**: 参考音频文件（直接上传，支持 wav/mp3/flac 等）
    - **input**: 要用克隆声音朗读的文本
    - **ref_text**: 参考音频对应的文本（可选，提供可提高克隆效果）
    - **response_format**: 输出格式，默认 wav
    """
    import uuid
    import tempfile

    # 校验 Base 模型可用
    base_info = MODEL_REGISTRY.get("qwen3-tts-base")
    if not base_info or not os.path.isdir(base_info["path"]):
        raise HTTPException(status_code=400, detail="Base 模型不可用，请先确保模型已下载")

    # 保存上传文件到临时目录
    suffix = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"tts_ref_{uuid.uuid4().hex[:8]}{suffix}")
    wav_path = None  # 用于记录转换后的 WAV 路径
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        # 非 WAV 格式需要通过 ffmpeg 转换（浏览器录音通常是 WebM）
        audio_path = tmp_path
        if suffix.lower() not in (".wav", ".flac", ".ogg"):
            import subprocess
            wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"音频格式转换失败（{suffix}），请上传 WAV/MP3/FLAC 格式：{result.stderr.decode()[:200]}"
                )
            audio_path = wav_path

        # 构建内部请求
        internal_req = {
            "text": input,
            "model_type": "Base",
            "ref_audio": audio_path,
        }
        if ref_text and ref_text.strip():
            internal_req["ref_text"] = ref_text.strip()

        # 生成音频（Semaphore 限制最多 MAX_QUEUE 路并发排队）
        async with _tts_semaphore:
            q = _mp_manager.Queue()
            loop = asyncio.get_running_loop()
            loop.run_in_executor(tts_pool, _stream_audio_task, internal_req, q)

            audio_chunks = []
            sample_rate = DEFAULT_SAMPLE_RATE
            while True:
                item = await loop.run_in_executor(None, q.get)
                if isinstance(item, Exception):
                    raise HTTPException(status_code=500, detail=str(item))
                if item is None:
                    break
                tag, data = item
                if tag == "sample_rate":
                    sample_rate = data
                elif tag == "audio":
                    audio_chunks.append(data)

            if not audio_chunks:
                raise HTTPException(status_code=500, detail="音频生成失败：未产生任何音频数据")

            pcm_data = b"".join(audio_chunks)

        if response_format == "pcm":
            return Response(content=pcm_data, media_type="audio/pcm")

        wav_data = _make_complete_wav(pcm_data, sample_rate=sample_rate)
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=cloned_speech.wav"},
        )
    finally:
        # 清理临时文件
        for p in [tmp_path, wav_path]:
            if p and os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8080, reload=False)
