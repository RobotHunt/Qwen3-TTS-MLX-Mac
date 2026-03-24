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

multiprocessing.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ──────────────────────────────────────────
# 配置区：根据你的硬件修改
# ──────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-Models")
OUTPUT_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-MLX-Mac/output")
MAX_WORKERS = 2  # M1 16GB 建议 2；M1 Pro/Max 32GB+ 可设 4

# 临时文件清理配置
CLEANUP_INTERVAL_SEC = 300    # 每 5 分钟扫描一次
CLEANUP_MAX_AGE_SEC = 600     # 文件超过 10 分钟即删除

# 默认流式采样率（与 Qwen3-TTS 模型一致）
DEFAULT_SAMPLE_RATE = 24000

MODEL_MAP = {
    "Base":        f"{MODELS_DIR}/Base-8bit",
    "VoiceDesign": f"{MODELS_DIR}/VoiceDesign-8bit",
    "CustomVoice": f"{MODELS_DIR}/CustomVoice-8bit",
}

# CustomVoice 可用说话人
CUSTOM_SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]

# ──────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────
app = FastAPI(
    title="Qwen3-TTS Apple Silicon API",
    version="2.0",
    description="本地离线 TTS 服务，支持 Base / VoiceDesign / CustomVoice 三种模型，含流式输出"
)

# CORS 中间件 —— 解决前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 生产环境应限制为实际前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts_pool = None
_start_time = None
_cleanup_task = None
_mp_manager = None


class TTSRequest(BaseModel):
    text: str
    model_type: str = "VoiceDesign"  # Base / VoiceDesign / CustomVoice
    instruct: Optional[str] = "A young female with a friendly, professional tone."
    voice: Optional[str] = None      # CustomVoice 说话人名
    ref_audio: Optional[str] = None  # Base 模型零样本克隆用
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
    """
    try:
        from mlx_audio.tts.utils import load_model
        import numpy as np

        model_type = request_data.get("model_type", "VoiceDesign")
        model_path = MODEL_MAP.get(model_type, MODEL_MAP["VoiceDesign"])

        if not os.path.isdir(model_path):
            queue.put(Exception(f"模型目录不存在: {model_path}，请先运行 setup.sh"))
            return

        model = load_model(model_path=model_path)

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
            # result.audio 是 numpy array 或 mlx array
            audio_np = np.array(result.audio).flatten()
            # 转为 16-bit PCM
            pcm_int16 = (audio_np * 32767).astype(np.int16)
            queue.put(("audio", pcm_int16.tobytes()))

        queue.put(None)  # 哨兵：结束

    except Exception as e:
        queue.put(e)


def _make_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """
    构造流式 WAV header（data size 设为 0xFFFFFFFF 表示未知长度）。
    大部分播放器/解码器能正确处理此格式。
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF
    riff_size = 36 + data_size  # 会溢出 uint32，但这里是流式标记

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        riff_size & 0xFFFFFFFF,
        b'WAVE',
        b'fmt ',
        16,                    # fmt chunk size
        1,                     # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size & 0xFFFFFFFF,
    )
    return header


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
# 生命周期事件
# ──────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    global tts_pool, _start_time, _cleanup_task, _mp_manager
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tts_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    _mp_manager = multiprocessing.Manager()
    _start_time = time.time()
    _cleanup_task = asyncio.get_event_loop().create_task(_cleanup_loop())
    print(f"🚀 Qwen3-TTS Server Started | Workers: {MAX_WORKERS} | Models: {MODELS_DIR}")


@app.on_event("shutdown")
def shutdown_event():
    global tts_pool, _cleanup_task, _mp_manager
    if _cleanup_task:
        _cleanup_task.cancel()
    if tts_pool:
        tts_pool.shutdown(wait=True)
    if _mp_manager:
        _mp_manager.shutdown()


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


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8080, reload=False)
