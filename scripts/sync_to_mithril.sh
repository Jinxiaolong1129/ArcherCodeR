#!/bin/bash
# 同步脚本 - 同步 Archer-Intuitor 检查点到 mithril-h100 服务器
# 用法: ./sync_to_mithril.sh

# 配置
SSH_KEY="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/ssh/mithril_jxl"
REMOTE_USER="ubuntu"
REMOTE_HOST="18.236.82.4"
REMOTE_BASE="/mnt/selfrl/ArcherCodeR/output/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-v2"
LOCAL_BASE="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR/Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple-v2"

# 要同步的目录列表
DIRS=(
    "global_step_105"
    "global_step_80"
)

# 高速 SSH 选项
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o Compression=no -c aes128-gcm@openssh.com"

# rsync 高速选项
RSYNC_OPTS="-avzP --inplace"

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
    
    # 执行同步 (rsync 会自动在目标创建目录)
    rsync ${RSYNC_OPTS} -e "ssh ${SSH_OPTS}" \
        "${SOURCE_DIR}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
    
    if [ $? -eq 0 ]; then
        DIR_END=$(date +%s)
        DIR_DURATION=$((DIR_END - DIR_START))
        echo "✅ ${dir} 同步完成 (耗时: ${DIR_DURATION}秒)"
        ((SUCCESS_COUNT++))
    else
        echo "❌ ${dir} 同步失败"
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

