#!/bin/bash
# Ray worker日志实时查看脚本

echo "🔍 Ray Worker 日志监控脚本"
echo "=================================="

# 检查Ray是否运行
if ! pgrep -f "ray" > /dev/null; then
    echo "❌ Ray似乎没有运行"
    echo "请先启动Ray训练任务"
    exit 1
fi

# 查找Ray session目录
RAY_SESSION_DIR="/tmp/ray/session_latest"
if [ ! -d "$RAY_SESSION_DIR" ]; then
    echo "❌ 找不到Ray session目录"
    echo "请检查Ray是否正常运行"
    exit 1
fi

echo "📂 Ray session目录: $RAY_SESSION_DIR"
echo "📊 可用的worker日志文件:"

# 列出有内容的日志文件
find "$RAY_SESSION_DIR/logs" -name "worker-*.out" -size +0 | head -10 | while read logfile; do
    size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null)
    echo "  📄 $(basename "$logfile") (${size} bytes)"
done

echo ""
echo "🚀 选择查看方式:"
echo "1) 实时监控所有worker输出"
echo "2) 查看特定worker的完整日志"
echo "3) 只显示最新的worker输出"

read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo "📺 实时监控所有worker输出 (Ctrl+C 停止)..."
        tail -f "$RAY_SESSION_DIR/logs"/worker-*.out | grep -v "^$" | while read line; do
            echo "[$(date '+%H:%M:%S')] $line"
        done
        ;;
    2)
        echo "📋 选择要查看的worker日志文件:"
        logfiles=($(find "$RAY_SESSION_DIR/logs" -name "worker-*.out" -size +0 | head -10))
        for i in "${!logfiles[@]}"; do
            echo "$((i+1))) $(basename "${logfiles[$i]}")"
        done
        read -p "请选择文件编号: " filenum
        if [[ $filenum -ge 1 && $filenum -le ${#logfiles[@]} ]]; then
            selected_file="${logfiles[$((filenum-1))]}"
            echo "📖 查看文件: $(basename "$selected_file")"
            echo "============================================"
            cat "$selected_file"
        else
            echo "❌ 无效的文件编号"
        fi
        ;;
    3)
        echo "📊 最新的worker输出:"
        echo "============================================"
        find "$RAY_SESSION_DIR/logs" -name "worker-*.out" -size +0 -exec tail -n 20 {} \; | tail -50
        ;;
    *)
        echo "❌ 无效的选择"
        exit 1
        ;;
esac 