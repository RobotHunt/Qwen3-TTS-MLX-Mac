"""
Qwen3-TTS FastAPI + ProcessPool 生产级服务端
专为 Apple Silicon (M1/M2/M3/M4) 设计
"""
import os
import sys
import asyncio
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ──────────────────────────────────────────
# 配置区：根据你的硬件修改
# ──────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-Models")
OUTPUT_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-MLX-Mac/output")
MAX_WORKERS = 2  # M1 16GB 建议 2；M1 Pro/Max 32GB+ 可设 4

MODEL_MAP = {
    "Base":        f"{MODELS_DIR}/Base-8bit",
    "VoiceDesign": f"{MODELS_DIR}/VoiceDesign-8bit",
    "CustomVoice": f"{MODELS_DIR}/CustomVoice-8bit",
}

# CustomVoice 可用说话人
CUSTOM_SPEAKERS = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]

app = FastAPI(
    title="Qwen3-TTS Apple Silicon API",
    version="1.0",
    description="本地离线 TTS 服务，支持 Base / VoiceDesign / CustomVoice 三种模型"
)

tts_pool = None


class TTSRequest(BaseModel):
    text: str
    model_type: str = "VoiceDesign"  # Base / VoiceDesign / CustomVoice
    instruct: Optional[str] = "A young female with a friendly, professional tone."
    voice: Optional[str] = None      # CustomVoice 说话人名
    ref_audio: Optional[str] = None  # Base 模型零样本克隆用
    ref_text: Optional[str] = None


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


@app.on_event("startup")
def startup_event():
    global tts_pool
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tts_pool = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    print(f"🚀 Qwen3-TTS Server Started | Workers: {MAX_WORKERS} | Models: {MODELS_DIR}")


@app.on_event("shutdown")
def shutdown_event():
    global tts_pool
    if tts_pool:
        tts_pool.shutdown(wait=True)


@app.post("/generate")
async def generate_tts(request: TTSRequest):
    """合成语音并返回音频文件路径"""
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


@app.get("/speakers")
async def list_speakers():
    """列出 CustomVoice 可用说话人"""
    return {"speakers": CUSTOM_SPEAKERS}


@app.get("/models")
async def list_models():
    """列出已安装的模型"""
    available = {k: os.path.isdir(v) for k, v in MODEL_MAP.items()}
    return {"models": available}


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)
