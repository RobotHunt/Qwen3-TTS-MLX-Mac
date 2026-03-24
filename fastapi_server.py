"""
Qwen3-TTS FastAPI + ProcessPool 生产级服务端
专为 Apple Silicon (M1/M2/M3/M4) 设计

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
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

# ──────────────────────────────────────────
# 配置区：根据你的硬件修改
# ──────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-Models")
OUTPUT_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-MLX-Mac/output")
MAX_WORKERS = 1  # 同时只加载一个模型，防止 OOM

# 临时文件清理配置
CLEANUP_INTERVAL_SEC = 300    # 每 5 分钟扫描一次
CLEANUP_MAX_AGE_SEC = 600     # 文件超过 10 分钟即删除

# 默认流式采样率（与 Qwen3-TTS 模型一致）
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
# 声音注册表（含 OpenAI 标准名 + Qwen3 原生名）
# ──────────────────────────────────────────
VOICE_REGISTRY = [
    {"openai_name": "alloy",   "qwen3_name": "vivian",   "gender": "female", "language": "en/zh"},
    {"openai_name": "echo",    "qwen3_name": "ryan",     "gender": "male",   "language": "en"},
    {"openai_name": "fable",   "qwen3_name": "serena",   "gender": "female", "language": "en"},
    {"openai_name": "nova",    "qwen3_name": "ono_anna", "gender": "female", "language": "ja/en"},
    {"openai_name": "onyx",    "qwen3_name": "eric",     "gender": "male",   "language": "en"},
    {"openai_name": "shimmer", "qwen3_name": "sohee",    "gender": "female", "language": "ko/en"},
    {"openai_name": "ash",     "qwen3_name": "aiden",    "gender": "male",   "language": "en"},
    {"openai_name": "coral",   "qwen3_name": "dylan",    "gender": "male",   "language": "en"},
    {"openai_name": "sage",    "qwen3_name": "uncle_fu", "gender": "male",   "language": "zh/en"},
]

# 双向映射：OpenAI 名 <-> Qwen3 名均可作为输入
_VOICE_LOOKUP = {}
for _v in VOICE_REGISTRY:
    _VOICE_LOOKUP[_v["openai_name"]] = _v["qwen3_name"]
    _VOICE_LOOKUP[_v["qwen3_name"]] = _v["qwen3_name"]  # 原生名直接通过


def resolve_voice(voice_input: str) -> str:
    """将 OpenAI 标准名或 Qwen3 原生名统一解析为 Qwen3 speaker 名"""
    return _VOICE_LOOKUP.get(voice_input, voice_input)

# ──────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────
tts_pool = None
_start_time = None
_cleanup_task = None
_mp_manager = None


# ──────────────────────────────────────────
# Lifespan（替代已弃用的 on_event）
# ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_pool, _start_time, _cleanup_task, _mp_manager
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tts_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    _mp_manager = multiprocessing.Manager()
    _start_time = time.time()
    _cleanup_task = asyncio.get_event_loop().create_task(_cleanup_loop())
    print(f"🚀 Qwen3-TTS Server Started | Workers: {MAX_WORKERS} | Models: {MODELS_DIR}")
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
    title="Qwen3-TTS Apple Silicon API",
    version="3.0",
    description="OpenAI-compatible local TTS service for Apple Silicon",
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


class TTSRequest(BaseModel):
    text: str
    model_type: str = "VoiceDesign"  # Base / VoiceDesign / CustomVoice
    instruct: Optional[str] = "A young female with a friendly, professional tone."
    voice: Optional[str] = None      # CustomVoice 说话人名
    ref_audio: Optional[str] = None  # Base 模型零样本克隆用
    ref_text: Optional[str] = None


class OpenAITTSRequest(BaseModel):
    """严格对标 OpenAI /v1/audio/speech 请求格式"""
    model: str                        # qwen3-tts-base / qwen3-tts-voicedesign / qwen3-tts-customvoice
    input: str                        # 要合成的文本
    voice: str = "alloy"              # OpenAI 标准名或 Qwen3 原生 speaker 名（双向均可）
    response_format: str = "wav"      # wav / pcm / mp3 / opus / aac / flac
    speed: float = 1.0                # 语速（保留字段）
    # 对齐 OpenAI gpt-4o-mini-tts 的 instructions 字段，用于 qwen3-tts-voicedesign
    instructions: Optional[str] = None
    # qwen3-tts-base 零样本克隆专用
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None


# ──────────────────────────────────────────
# Worker 函数（子进程中执行）
# ──────────────────────────────────────────
def _generate_audio_task(task_id: str, request_data: dict, output_dir: str):
    """
    子进程 worker：延迟导入 MLX 确保独立的 Metal GPU 上下文
    """
    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError as e:
        return False, str(e)

    model_type = request_data.get("model_type", "VoiceDesign")
    model_path = MODEL_MAP.get(model_type, MODEL_MAP["VoiceDesign"])

    if not os.path.isdir(model_path):
        return False, f"模型目录不存在: {model_path}，请先运行 setup.sh"

    kwargs = {
        "text": request_data["text"],
        "model": model_path,
        "output_path": output_dir,
        "file_prefix": task_id,
        "audio_format": "wav",
        "stream": False,
        "play": False,
    }

    # 根据模型类型设置对应参数
    if model_type == "VoiceDesign":
        kwargs["instruct"] = request_data.get("instruct")
    elif model_type == "CustomVoice":
        voice = request_data.get("voice", "vivian")
        if voice not in CUSTOM_SPEAKERS:
            return False, f"不支持的说话人 '{voice}'，可用: {CUSTOM_SPEAKERS}"
        kwargs["voice"] = voice
    elif model_type == "Base":
        if request_data.get("ref_audio"):
            kwargs["ref_audio"] = request_data["ref_audio"]
            kwargs["ref_text"] = request_data.get("ref_text", "")

    try:
        generate_audio(**kwargs)
        return True, f"{output_dir}/{task_id}_000.wav"
    except Exception as e:
        return False, str(e)


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
                gen_kwargs["ref_text"] = request_data.get("ref_text", "")

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
        "models": {k: os.path.isdir(v) for k, v in MODEL_MAP.items()},
        "workers": MAX_WORKERS,
    }


# ─── 完整生成（原有端点） ───
@app.post("/generate")
async def generate_tts(request: TTSRequest):
    """合成语音并返回完整音频文件"""
    import uuid
    task_id = f"tts_{uuid.uuid4().hex[:8]}"
    loop = asyncio.get_running_loop()

    try:
        success, result = await loop.run_in_executor(
            tts_pool, _generate_audio_task, task_id, request.dict(), OUTPUT_DIR
        )
        if success:
            return FileResponse(result, media_type="audio/wav", filename=f"{task_id}.wav")
        else:
            raise HTTPException(status_code=500, detail=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 流式生成（新端点） ───
@app.post("/stream")
async def stream_tts(request: TTSRequest):
    """流式合成语音，逐段返回 WAV 音频流（降低首字延迟）"""
    q = _mp_manager.Queue()
    loop = asyncio.get_running_loop()

    # 在进程池中启动流式生成（fire-and-forget）
    loop.run_in_executor(tts_pool, _stream_audio_task, request.dict(), q)

    async def audio_chunk_generator():
        sample_rate = DEFAULT_SAMPLE_RATE
        header_sent = False

        while True:
            # 非阻塞轮询 Queue
            item = await loop.run_in_executor(None, q.get)

            # 处理错误
            if isinstance(item, Exception):
                raise HTTPException(status_code=500, detail=str(item))

            # 哨兵：结束
            if item is None:
                break

            tag, data = item

            # 采样率元信息
            if tag == "sample_rate":
                sample_rate = data
                continue

            # 音频 chunk
            if tag == "audio":
                if not header_sent:
                    yield _make_wav_header(sample_rate=sample_rate)
                    header_sent = True
                yield data

    return StreamingResponse(audio_chunk_generator(), media_type="audio/wav")


# ─── 说话人列表 ───
@app.get("/speakers")
async def list_speakers():
    """列出 CustomVoice 可用说话人"""
    return {"speakers": CUSTOM_SPEAKERS}


# ─── 模型列表 ───
@app.get("/models")
async def list_models():
    """列出已安装的模型"""
    available = {k: os.path.isdir(v) for k, v in MODEL_MAP.items()}
    return {"models": available}


# ──────────────────────────────────────────
# OpenAI 兼容端点
# ──────────────────────────────────────────

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
                "owned_by": "qwen3-tts",
                "description": info["description"],
                "capabilities": info["capabilities"],
                "required_params": info["required_params"],
                "optional_params": info.get("optional_params", []),
                "local_dir": info["local_dir"],
            })
    return {"object": "list", "data": data}


# ─── GET /v1/audio/voices ───
@app.get("/v1/audio/voices")
async def get_openai_voices():
    """
    列出所有可用声音。
    每个声音包含 OpenAI 标准名、Qwen3 原生名、性别、支持语言。
    两种名称均可在 /v1/audio/speech 的 voice 字段中直接使用。
    """
    voices = []
    for v in VOICE_REGISTRY:
        voices.append({
            "voice_id": v["openai_name"],
            "name": v["openai_name"],
            "qwen3_name": v["qwen3_name"],
            "gender": v["gender"],
            "language": v["language"],
            "preview_url": None,
        })
    return {"voices": voices}


# ─── POST /v1/audio/speech ───
@app.post("/v1/audio/speech")
async def openai_audio_speech(request: OpenAITTSRequest):
    """
    兼容 OpenAI `POST /v1/audio/speech` 接口，流式传输音频。

    模型 ID：
      - qwen3-tts-customvoice : 需要 voice（预设说话人）
      - qwen3-tts-voicedesign : 需要 instructions（自然语言音色描述）
      - qwen3-tts-base        : 需要 ref_audio（+ 可选 ref_text，零样本克隆）

    voice 字段支持 OpenAI 标准名（alloy/echo/...）和 Qwen3 原生名（vivian/ryan/...），双向均可。
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
                detail="qwen3-tts-voicedesign 需要提供 'instructions' 参数来描述期望的音色，"
                       "例如: 'A warm, mature male voice with a calm and professional tone.'"
            )
        internal_req["instruct"] = request.instructions

    elif internal_type == "Base":
        if not request.ref_audio:
            raise HTTPException(
                status_code=422,
                detail="qwen3-tts-base 需要提供 'ref_audio' 参数（参考音频文件路径）用于零样本语音克隆，"
                       "同时建议提供 'ref_text'（参考音频对应文本）"
            )
        internal_req["ref_audio"] = request.ref_audio
        internal_req["ref_text"] = request.ref_text or ""

    # ── 4. 生成音频并收集全部 PCM 数据 ──
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


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8080, reload=False)
