#!/bin/bash
# 高速同步脚本 - 同步实验到 trojai4 服务器 (通过跳板机)
# 用法: ./sync_to_trojai4.sh

# 配置
SSH_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/ssh"
SSH_KEY="${SSH_DIR}/trojai4"

PROXY_USER="jin509"
PROXY_HOST="data.cs.purdue.edu"

REMOTE_USER="jin509"
REMOTE_HOST="trojai4.cs.purdue.edu"
REMOTE_BASE="/data4/user/jin509/ArcherCodeR/output/ArcherCodeR"
LOCAL_BASE="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

# 要同步的实验目录 (Pure-GRPO 系列)
DIRS=(
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-probdisparity-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-trajentropy-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-step10"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step50-batch64"
)

# SSH 跳板机代理命令
PROXY_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}"

# rsync 高速选项
RSYNC_OPTS="-avzP --inplace"

echo "=========================================="
echo "🚀 批量同步开始 (trojai4 服务器)"
echo "=========================================="
echo "跳板机: ${PROXY_USER}@${PROXY_HOST}"
echo "目标服务器: ${REMOTE_USER}@${REMOTE_HOST}"
echo "目标路径: ${REMOTE_BASE}/"
echo "待同步目录数: ${#DIRS[@]}"
echo "=========================================="
echo ""

# 检查 SSH key 文件
echo "🔑 检查 SSH 密钥文件..."
if [ ! -f "${SSH_KEY}" ]; then
    echo "❌ SSH 密钥不存在: ${SSH_KEY}"
    echo "   请将 trojai4 密钥放到 ${SSH_DIR}/ 目录"
    exit 1
fi
echo "✅ SSH 密钥文件就绪"
echo ""

# 设置密钥权限
chmod 600 "${SSH_KEY}" 2>/dev/null

# 显示所有目录的大小
echo "📦 待同步目录 (Pure-GRPO 系列):"
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

# 测试 SSH 连接
echo "🔗 测试 SSH 连接..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no \
    -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    "${REMOTE_USER}@${REMOTE_HOST}" "echo '连接成功'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ SSH 连接失败，请检查密钥和网络"
    exit 1
fi
echo "✅ SSH 连接测试通过"
echo ""

# 在目标服务器创建基础目录
echo "📂 确保目标服务器路径存在..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no \
    -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}"
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
    
    # 执行同步 (通过跳板机)
    rsync ${RSYNC_OPTS} \
        -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o ProxyCommand=\"ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}\"" \
        "${SOURCE_DIR}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
    
    if [ $? -eq 0 ]; then
        DIR_END=$(date +%s)
        DIR_DURATION=$((DIR_END - DIR_START))
        MINUTES=$((DIR_DURATION / 60))
        SECONDS=$((DIR_DURATION % 60))
        echo "✅ ${dir} 同步完成 (耗时: ${MINUTES}分${SECONDS}秒)"
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
echo "⏱️  总耗时: ${DURATION} 秒 ($((DURATION / 60)) 分 $((DURATION % 60)) 秒)"
echo "=========================================="

