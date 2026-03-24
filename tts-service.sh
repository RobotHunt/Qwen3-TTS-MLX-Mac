#!/bin/bash
# ──────────────────────────────────────────
# TTS Server 管理脚本
# 用法: ./tts-service.sh [install|uninstall|start|stop|restart|status|logs]
# ──────────────────────────────────────────

PLIST_NAME="com.local.qwen3-tts"
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/${PLIST_NAME}.plist"
PLIST_DST="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

case "$1" in
  install)
    mkdir -p "$LOG_DIR"
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl load "$PLIST_DST"
    echo -e "${GREEN}✅ 已安装并启动 TTS 服务${NC}"
    echo -e "   开机自启: ${GREEN}已开启${NC}"
    echo -e "   崩溃恢复: ${GREEN}已开启${NC} (5秒后自动重启)"
    echo -e "   日志目录: $LOG_DIR"
    echo -e "   访问地址: ${YELLOW}http://127.0.0.1:8080/${NC}"
    ;;
  uninstall)
    launchctl unload "$PLIST_DST" 2>/dev/null
    rm -f "$PLIST_DST"
    echo -e "${GREEN}✅ 已卸载 TTS 服务${NC} (开机自启已关闭)"
    ;;
  start)
    mkdir -p "$LOG_DIR"
    if [ ! -f "$PLIST_DST" ]; then
      cp "$PLIST_SRC" "$PLIST_DST"
      launchctl load "$PLIST_DST"
    else
      launchctl start "$PLIST_NAME"
    fi
    echo -e "${GREEN}✅ TTS 服务已启动${NC}"
    ;;
  stop)
    launchctl stop "$PLIST_NAME"
    echo -e "${YELLOW}⏹ TTS 服务已停止${NC} (KeepAlive 会在 5s 后自动重启)"
    echo -e "   如需彻底停止请用: $0 uninstall"
    ;;
  restart)
    launchctl stop "$PLIST_NAME"
    sleep 2
    launchctl start "$PLIST_NAME"
    echo -e "${GREEN}🔄 TTS 服务已重启${NC}"
    ;;
  status)
    echo "── TTS Service Status ──"
    if launchctl list | grep -q "$PLIST_NAME"; then
      PID=$(launchctl list | grep "$PLIST_NAME" | awk '{print $1}')
      if [ "$PID" != "-" ] && [ -n "$PID" ]; then
        echo -e "状态: ${GREEN}运行中${NC} (PID: $PID)"
      else
        echo -e "状态: ${YELLOW}已注册但未运行${NC}"
      fi
    else
      echo -e "状态: ${RED}未安装${NC}"
    fi
    # 检查端口
    if lsof -i:8080 -sTCP:LISTEN &>/dev/null; then
      echo -e "端口: ${GREEN}8080 已监听${NC}"
    else
      echo -e "端口: ${RED}8080 未监听${NC}"
    fi
    # 日志文件大小
    if [ -f "$LOG_DIR/stdout.log" ]; then
      SIZE=$(du -h "$LOG_DIR/stdout.log" | cut -f1)
      echo "日志: stdout.log ($SIZE)"
    fi
    ;;
  logs)
    echo "── 最近 30 行日志 ──"
    tail -30 "$LOG_DIR/stdout.log" "$LOG_DIR/stderr.log" 2>/dev/null || echo "暂无日志"
    echo ""
    echo -e "实时跟踪: ${YELLOW}tail -f $LOG_DIR/stdout.log${NC}"
    ;;
  *)
    echo "TTS Server 管理脚本"
    echo ""
    echo "用法: $0 <命令>"
    echo ""
    echo "命令:"
    echo "  install    安装服务 (开机自启 + 崩溃恢复)"
    echo "  uninstall  卸载服务 (彻底停止 + 关闭开机自启)"
    echo "  start      启动服务"
    echo "  stop       停止服务 (KeepAlive 会自动重启)"
    echo "  restart    重启服务"
    echo "  status     查看服务状态"
    echo "  logs       查看日志"
    ;;
esac
