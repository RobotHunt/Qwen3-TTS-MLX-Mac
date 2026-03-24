#!/bin/bash
set -e

echo "🚀 Qwen3-TTS Apple Silicon 一键安装"
echo "======================================"

# 1. 创建虚拟环境
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 安装依赖
echo "📦 安装依赖..."
pip install -q -r requirements.txt

# 3. 下载模型
MODELS_DIR="$HOME/Downloads/Qwen3-TTS-Models"
mkdir -p "$MODELS_DIR"

download_model() {
    local name=$1 repo=$2 dir="$MODELS_DIR/$name"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "✅ $name 已存在，跳过"
    else
        echo "⬇️  下载 $name..."
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo', local_dir='$dir', max_workers=4)
print('✅ $name 下载完成')
"
    fi
}

download_model "Base-8bit"        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
download_model "VoiceDesign-8bit" "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
download_model "CustomVoice-8bit" "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

echo ""
echo "🎉 安装完成！"
echo ""
echo "使用方式："
echo "  source .venv/bin/activate"
echo "  python demo.py basic --play       # 基础合成"
echo "  python demo.py fast --play        # 极速流式"
echo "  python demo.py voicedesign --play # 音色定制"
echo "  python demo.py all --play         # 运行全部演示"
echo "  python fastapi_server.py          # 启动 API 服务器"
