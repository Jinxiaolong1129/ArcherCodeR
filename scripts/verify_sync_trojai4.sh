#!/bin/bash
# 验证 trojai4 同步完成脚本（通过跳板机）
# 用法: ./verify_sync_trojai4.sh [目录名]
#       ./verify_sync_trojai4.sh all  # 验证所有实验目录

# 配置
SSH_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/ssh"
SSH_KEY="${SSH_DIR}/trojai4"

PROXY_USER="jin509"
PROXY_HOST="data.cs.purdue.edu"

REMOTE_USER="jin509"
REMOTE_HOST="trojai4.cs.purdue.edu"
REMOTE_BASE="/data4/user/jin509/ArcherCodeR/output/ArcherCodeR"
LOCAL_BASE="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

# 所有实验目录 (Pure-GRPO 系列)
ALL_DIRS=(
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-probdisparity-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-trajentropy-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-step10"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-intuitor-step50"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-from-tokenentropy-step50-batch64"
)

# 验证单个目录的函数
verify_single_dir() {
    local DIR_NAME="$1"
    local LOCAL_DIR="${LOCAL_BASE}/${DIR_NAME}"
    local REMOTE_DIR="${REMOTE_BASE}/${DIR_NAME}"

    echo "=========================================="
    echo "🔍 验证: ${DIR_NAME}"
    echo "=========================================="

    # 检查本地目录是否存在
    if [ ! -d "${LOCAL_DIR}" ]; then
        echo "❌ 本地目录不存在: ${LOCAL_DIR}"
        return 1
    fi

    # 1. 比较文件数量
    LOCAL_COUNT=$(find "${LOCAL_DIR}" -type f | wc -l)
    REMOTE_COUNT=$(ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
        ${REMOTE_USER}@${REMOTE_HOST} "find '${REMOTE_DIR}' -type f 2>/dev/null | wc -l" 2>/dev/null)

    if [ -z "${REMOTE_COUNT}" ] || [ "${REMOTE_COUNT}" = "0" ]; then
        echo "📊 本地: ${LOCAL_COUNT} 文件 | 远程: 未开始/不存在"
        echo "⏳ 状态: 等待传输"
        return 2
    fi

    # 2. 比较大小
    LOCAL_SIZE=$(du -sb "${LOCAL_DIR}" | cut -f1)
    LOCAL_SIZE_H=$(du -sh "${LOCAL_DIR}" | cut -f1)
    REMOTE_SIZE=$(ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
        ${REMOTE_USER}@${REMOTE_HOST} "du -sb '${REMOTE_DIR}' 2>/dev/null | cut -f1" 2>/dev/null)
    REMOTE_SIZE_H=$(ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
        ${REMOTE_USER}@${REMOTE_HOST} "du -sh '${REMOTE_DIR}' 2>/dev/null | cut -f1" 2>/dev/null)

    # 计算进度百分比
    if [ -n "${REMOTE_SIZE}" ] && [ "${LOCAL_SIZE}" -gt 0 ]; then
        PROGRESS=$((REMOTE_SIZE * 100 / LOCAL_SIZE))
    else
        PROGRESS=0
    fi

    echo "📊 文件数: 本地 ${LOCAL_COUNT} | 远程 ${REMOTE_COUNT}"
    echo "📊 大小:   本地 ${LOCAL_SIZE_H} | 远程 ${REMOTE_SIZE_H}"
    echo "📊 进度:   ${PROGRESS}%"

    if [ "${LOCAL_COUNT}" -eq "${REMOTE_COUNT}" ] && [ "${PROGRESS}" -ge 99 ]; then
        echo "✅ 状态: 传输完成"
        return 0
    else
        echo "🔄 状态: 传输中..."
        return 2
    fi
}

# 主逻辑
echo "=========================================="
echo "🚀 trojai4 同步验证"
echo "=========================================="
echo "跳板机: ${PROXY_USER}@${PROXY_HOST}"
echo "目标: ${REMOTE_USER}@${REMOTE_HOST}"
echo "远程路径: ${REMOTE_BASE}/"
echo "=========================================="
echo ""

# 测试连接
echo "🔗 测试 SSH 连接..."
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    -o ProxyCommand="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    "${REMOTE_USER}@${REMOTE_HOST}" "echo '连接成功'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ SSH 连接失败"
    exit 1
fi
echo "✅ SSH 连接成功"
echo ""

# 处理参数
if [ "$1" = "all" ] || [ -z "$1" ]; then
    # 验证所有目录
    echo "验证所有实验目录 (${#ALL_DIRS[@]} 个)..."
    echo ""
    
    COMPLETE=0
    IN_PROGRESS=0
    NOT_STARTED=0
    
    for dir in "${ALL_DIRS[@]}"; do
        verify_single_dir "$dir"
        result=$?
        if [ $result -eq 0 ]; then
            ((COMPLETE++))
        elif [ $result -eq 2 ]; then
            ((IN_PROGRESS++))
        else
            ((NOT_STARTED++))
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "📋 总结"
    echo "=========================================="
    echo "✅ 完成: ${COMPLETE}"
    echo "🔄 进行中: ${IN_PROGRESS}"
    echo "⏳ 未开始/失败: ${NOT_STARTED}"
    echo "=========================================="
else
    # 验证单个目录
    verify_single_dir "$1"
fi

