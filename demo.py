"""
Qwen3-TTS Apple Silicon 综合演示脚本
支持模式：基础合成 / 极速流式 / 音色定制 / 多语言混合 / 多角色切换 / 并发压测
"""
import os
import sys
import time
import argparse

try:
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    print("❌ 请先安装依赖: pip install mlx mlx-audio soundfile")
    sys.exit(1)

# 默认模型路径（本地下载目录）
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-Models")
MODEL_MAP = {
    "base":        f"{MODELS_DIR}/Base-8bit",
    "voicedesign": f"{MODELS_DIR}/VoiceDesign-8bit",
    "customvoice": f"{MODELS_DIR}/CustomVoice-8bit",
}
# 如果本地不存在则 fallback 到 HuggingFace Hub 自动下载
HF_FALLBACK = {
    "base":        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "voicedesign": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "customvoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
}
OUTPUT_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS-MLX-Mac/output")

def resolve_model(model_type: str) -> str:
    local = MODEL_MAP.get(model_type)
    if local and os.path.isdir(local):
        return local
    print(f"⚠️ 本地模型 {local} 不存在，将从 HuggingFace 自动下载...")
    return HF_FALLBACK[model_type]

def demo_basic(args):
    """基础合成"""
    print("\n🔹 基础合成 (Base-8bit)")
    generate_audio(
        text=args.text or "您好，这是基础版模型。发音字正腔圆，适合通用朗读。Hello, this is a base model test.",
        model=resolve_model("base"),
        output_path=OUTPUT_DIR, file_prefix="demo_basic", audio_format="wav",
        play=args.play
    )

def demo_fast(args):
    """极速流式 (4-bit + streaming)"""
    print("\n🔹 极速流式合成 (VoiceDesign-4bit + Stream)")
    model = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
    generate_audio(
        text=args.text or "流式模式已开启！首字延迟被压缩到一百毫秒内。This is streaming mode with ultra-low latency!",
        model=model,
        instruct="A young female with a friendly, energetic tone.",
        output_path=OUTPUT_DIR, file_prefix="demo_fast", audio_format="wav",
        stream=True, streaming_interval=0.5, play=args.play
    )

def demo_voicedesign(args):
    """音色定制"""
    print("\n🔹 音色定制 (VoiceDesign-8bit)")
    generate_audio(
        text=args.text or "You fool! How dare you summon me from my slumber? Leave this tower at once!",
        model=resolve_model("voicedesign"),
        instruct=args.instruct or "A very old, grumpy, and raspy male voice, speaking like a furious medieval wizard.",
        output_path=OUTPUT_DIR, file_prefix="demo_voicedesign", audio_format="wav",
        play=args.play
    )

def demo_multilingual(args):
    """跨语言混合 (中/英/日)"""
    print("\n🔹 跨语言无缝切换 (中/英/日)")
    generate_audio(
        text=args.text or "今天的天气真好啊！It's a beautiful day outside! 素晴らしい天気ですね、一緒に遊びに行きましょう！",
        model=resolve_model("voicedesign"),
        instruct="A cheerful young female anime voice actor with an energetic and sweet tone.",
        output_path=OUTPUT_DIR, file_prefix="demo_multilingual", audio_format="wav",
        play=args.play
    )

def demo_customvoice(args):
    """多角色切换 (CustomVoice 预设说话人)"""
    speakers = {
        "vivian":   "Hello! I am Vivian. CustomVoice allows instant speaker switching.",
        "uncle_fu": "大家好，我是傅叔叔。这是中文男声效果展示。",
        "ryan":     "Hey there, I am Ryan. Multiple unique vocal identities, out of the box!",
        "ono_anna": "こんにちは、私はAnnaです。日本語の音声合成もとても自然ですよ！",
    }
    speaker = args.voice or "vivian"
    if speaker not in speakers:
        print(f"⚠️ 可用角色: {', '.join(speakers.keys())}, serena, aiden, sohee, eric, dylan")
        return
    print(f"\n🔹 CustomVoice 角色切换 [{speaker}]")
    generate_audio(
        text=args.text or speakers[speaker],
        model=resolve_model("customvoice"),
        voice=speaker,
        output_path=OUTPUT_DIR, file_prefix=f"demo_customvoice_{speaker}", audio_format="wav",
        play=args.play
    )

def demo_concurrent(args):
    """4 路并发极压测试"""
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    from concurrent.futures import ProcessPoolExecutor

    def _worker(task_id, text):
        from mlx_audio.tts.generate import generate_audio as _gen
        model = resolve_model("voicedesign")
        start = time.time()
        _gen(text=text, model=model,
             instruct="A friendly female voice.",
             output_path=OUTPUT_DIR, file_prefix=f"demo_concurrent_{task_id}",
             audio_format="wav", play=False)
        return task_id, time.time() - start

    texts = [
        "Concurrent test one: testing M1 throughput.",
        "并发测试第二段：苹果统一内存架构多进程表现。",
        "Concurrent test three: simulating a busy TTS server.",
        "并发测试第四段：高强度四路并发压力测试。",
    ]
    print("\n🔹 4 路并发极压测试")
    start = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_worker, i, t) for i, t in enumerate(texts)]
        for f in futures:
            tid, elapsed = f.result()
            print(f"  [{tid}] ✅ {elapsed:.2f}s")
    print(f"  总计: {time.time()-start:.2f}s")

DEMOS = {
    "basic":        demo_basic,
    "fast":         demo_fast,
    "voicedesign":  demo_voicedesign,
    "multilingual": demo_multilingual,
    "customvoice":  demo_customvoice,
    "concurrent":   demo_concurrent,
    "all":          None,
}

def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Apple Silicon Demo")
    parser.add_argument("mode", choices=DEMOS.keys(), default="basic", nargs="?",
                        help="演示模式 (默认: basic)")
    parser.add_argument("--text", type=str, help="自定义合成文本")
    parser.add_argument("--instruct", type=str, help="VoiceDesign 音色描述")
    parser.add_argument("--voice", type=str, help="CustomVoice 说话人名")
    parser.add_argument("--play", action="store_true", help="生成后自动播放")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.time()

    if args.mode == "all":
        for name, fn in DEMOS.items():
            if fn:
                fn(args)
    else:
        DEMOS[args.mode](args)

    print(f"\n🎉 完成！总耗时: {time.time()-start:.2f}s")
    print(f"📂 输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
