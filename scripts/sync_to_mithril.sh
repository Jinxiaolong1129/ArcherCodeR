#!/bin/bash
# 同步脚本 - 同步 Archer-Intuitor 检查点到 mithril-h100 服务器
# 用法: ./sync_to_mithril.sh

# 配置
SSH_KEY="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/ssh/mithril_jxl"
REMOTE_USER="ubuntu"
REMOTE_HOST="35.95.65.28"
REMOTE_BASE="/mnt/selfrl/ArcherCodeR/output"
LOCAL_BASE="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

# 要同步的目录列表 (相对于LOCAL_BASE)
DIRS=(
    "Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-epoch-10-lr/global_step_50"
    "Archer-TrajectoryEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-epoch-10-lr/global_step_40"
    "Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-epoch-10-lr/global_step_100"
)

# SSH 选项 - 添加保活设置防止连接断开
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes -c aes128-gcm@openssh.com"

# rsync 选项 - 添加超时和部分传输支持
RSYNC_OPTS="-avzP --inplace --partial --timeout=120"

# 重试次数
MAX_RETRIES=3

echo "=========================================="
echo "🚀 批量同步开始"
echo "=========================================="
echo "目标服务器: ${REMOTE_USER}@${REMOTE_HOST}"
echo "目标路径: ${REMOTE_BASE}/"
echo "待同步目录数: ${#DIRS[@]}"
echo "=========================================="
echo ""

# 显示所有目录的大小
echo "📦 待同步目录:"
for dir in "${DIRS[@]}"; do
    if [ -d "${LOCAL_BASE}/${dir}" ]; then
        SIZE=$(du -sh "${LOCAL_BASE}/${dir}" 2>/dev/null | cut -f1)
        echo "  - ${dir}: ${SIZE}"
    else
        echo "  - ${dir}: ❌ 不存在"
    fi
done
echo ""

# 开始计时
START_TIME=$(date +%s)

# 在目标服务器创建基础目录
echo "📂 确保目标服务器路径存在..."
ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}"
if [ $? -eq 0 ]; then
    echo "✅ 目标路径已就绪: ${REMOTE_BASE}"
else
    echo "❌ 无法创建目标路径，退出"
    exit 1
fi
echo ""

# 同步每个目录
SUCCESS_COUNT=0
FAIL_COUNT=0

for dir in "${DIRS[@]}"; do
    echo "=========================================="
    echo "📁 正在同步: ${dir}"
    echo "=========================================="
    
    SOURCE_DIR="${LOCAL_BASE}/${dir}"
    
    if [ ! -d "${SOURCE_DIR}" ]; then
        echo "❌ 目录不存在，跳过"
        ((FAIL_COUNT++))
        continue
    fi
    
    DIR_START=$(date +%s)
    
    # 获取父目录路径并在远程创建
    PARENT_DIR=$(dirname "${dir}")
    ssh ${SSH_OPTS} "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}/${PARENT_DIR}"
    
    # 执行同步 - 带重试机制
    RETRY_COUNT=0
    SYNC_SUCCESS=false
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SYNC_SUCCESS" = false ]; do
        ((RETRY_COUNT++))
        
        if [ $RETRY_COUNT -gt 1 ]; then
            echo "🔄 重试第 ${RETRY_COUNT}/${MAX_RETRIES} 次..."
            sleep 5  # 重试前等待5秒
        fi
        
        rsync ${RSYNC_OPTS} -e "ssh ${SSH_OPTS}" \
            "${SOURCE_DIR}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${PARENT_DIR}/"
        
        if [ $? -eq 0 ]; then
            SYNC_SUCCESS=true
        fi
    done
    
    if [ "$SYNC_SUCCESS" = true ]; then
        DIR_END=$(date +%s)
        DIR_DURATION=$((DIR_END - DIR_START))
        echo "✅ ${dir} 同步完成 (耗时: ${DIR_DURATION}秒)"
        ((SUCCESS_COUNT++))
    else
        echo "❌ ${dir} 同步失败 (已重试 ${MAX_RETRIES} 次)"
        ((FAIL_COUNT++))
    fi
    echo ""
done

# 结束计时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "🎉 批量同步完成!"
echo "=========================================="
echo "✅ 成功: ${SUCCESS_COUNT} 个目录"
echo "❌ 失败: ${FAIL_COUNT} 个目录"
echo "⏱️  总耗时: ${DURATION} 秒 ($((DURATION / 60)) 分钟)"
echo "=========================================="

