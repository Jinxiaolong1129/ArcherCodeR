#!/bin/bash
# 验证 Purdue 同步完成脚本（通过跳板机）
# 用法: ./verify_sync_purdue.sh [目录名]

# 配置
SSH_DIR="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/ssh"
PROXY_KEY="${SSH_DIR}/trojai4"
TARGET_KEY="${SSH_DIR}/purcl_cs"

PROXY_USER="jin509"
PROXY_HOST="data.cs.purdue.edu"

REMOTE_USER="jin509"
REMOTE_HOST="purcl.cs.purdue.edu"
REMOTE_BASE="/scratch/jin509/ArcherCodeR/output/ArcherCodeR"
LOCAL_BASE="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"

# 默认验证的目录
DIR_NAME="${1:-Archer-ProbDisparity-Qwen2.5-1.5B-2k-8k-batch64-kl005}"

LOCAL_DIR="${LOCAL_BASE}/${DIR_NAME}"
REMOTE_DIR="${REMOTE_BASE}/${DIR_NAME}"

# SSH 选项（通过跳板机）
SSH_CMD="ssh -i ${TARGET_KEY} -o StrictHostKeyChecking=no -o ProxyCommand=\"ssh -i ${PROXY_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}\""

echo "=========================================="
echo "🔍 验证 Purdue 同步: ${DIR_NAME}"
echo "=========================================="
echo ""

# 检查本地目录是否存在
if [ ! -d "${LOCAL_DIR}" ]; then
    echo "❌ 本地目录不存在: ${LOCAL_DIR}"
    exit 1
fi

echo "📂 本地目录: ${LOCAL_DIR}"
echo "📂 远程目录: ${REMOTE_DIR}"
echo "🔗 跳板机: ${PROXY_USER}@${PROXY_HOST}"
echo ""

# 1. 比较文件数量
echo "=========================================="
echo "📊 文件数量对比"
echo "=========================================="
LOCAL_COUNT=$(find "${LOCAL_DIR}" -type f | wc -l)
REMOTE_COUNT=$(ssh -i ${TARGET_KEY} -o StrictHostKeyChecking=no \
    -o ProxyCommand="ssh -i ${PROXY_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    ${REMOTE_USER}@${REMOTE_HOST} "find '${REMOTE_DIR}' -type f 2>/dev/null | wc -l")

echo "本地文件数: ${LOCAL_COUNT}"
echo "远程文件数: ${REMOTE_COUNT}"

if [ "${LOCAL_COUNT}" -eq "${REMOTE_COUNT}" ]; then
    echo "✅ 文件数量一致"
else
    echo "❌ 文件数量不一致! 差异: $((LOCAL_COUNT - REMOTE_COUNT))"
fi
echo ""

# 2. 比较总大小
echo "=========================================="
echo "📊 总大小对比"
echo "=========================================="
LOCAL_SIZE=$(du -sb "${LOCAL_DIR}" | cut -f1)
REMOTE_SIZE=$(ssh -i ${TARGET_KEY} -o StrictHostKeyChecking=no \
    -o ProxyCommand="ssh -i ${PROXY_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    ${REMOTE_USER}@${REMOTE_HOST} "du -sb '${REMOTE_DIR}' 2>/dev/null | cut -f1")

LOCAL_SIZE_H=$(du -sh "${LOCAL_DIR}" | cut -f1)
REMOTE_SIZE_H=$(ssh -i ${TARGET_KEY} -o StrictHostKeyChecking=no \
    -o ProxyCommand="ssh -i ${PROXY_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}" \
    ${REMOTE_USER}@${REMOTE_HOST} "du -sh '${REMOTE_DIR}' 2>/dev/null | cut -f1")

echo "本地大小: ${LOCAL_SIZE_H} (${LOCAL_SIZE} bytes)"
echo "远程大小: ${REMOTE_SIZE_H} (${REMOTE_SIZE} bytes)"

if [ "${LOCAL_SIZE}" -eq "${REMOTE_SIZE}" ]; then
    echo "✅ 大小完全一致"
else
    DIFF=$((LOCAL_SIZE - REMOTE_SIZE))
    DIFF_MB=$((DIFF / 1024 / 1024))
    echo "⚠️ 大小差异: ${DIFF_MB} MB"
fi
echo ""

# 3. 使用 rsync dry-run 检查差异
echo "=========================================="
echo "📊 Rsync 差异检查 (dry-run)"
echo "=========================================="
DIFF_FILES=$(rsync -avnc --delete \
    -e "ssh -i ${TARGET_KEY} -o StrictHostKeyChecking=no -o ProxyCommand=\"ssh -i ${PROXY_KEY} -o StrictHostKeyChecking=no -W %h:%p ${PROXY_USER}@${PROXY_HOST}\"" \
    "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" 2>/dev/null | \
    grep -v "^$" | grep -v "sending incremental" | grep -v "^sent\|^total\|bytes/sec")

if [ -z "${DIFF_FILES}" ]; then
    echo "✅ 没有差异，完全同步"
else
    echo "⚠️ 发现差异文件:"
    echo "${DIFF_FILES}" | head -20
    DIFF_COUNT=$(echo "${DIFF_FILES}" | wc -l)
    if [ "${DIFF_COUNT}" -gt 20 ]; then
        echo "... 还有 $((DIFF_COUNT - 20)) 个文件"
    fi
fi
echo ""

# 4. 总结
echo "=========================================="
echo "📋 验证总结"
echo "=========================================="

if [ "${LOCAL_COUNT}" -eq "${REMOTE_COUNT}" ]; then
    if [ -z "${DIFF_FILES}" ]; then
        echo "🎉 ${DIR_NAME} 传输完成且验证通过!"
        echo "   - 文件数量: ${LOCAL_COUNT} ✅"
        echo "   - Rsync校验: 无差异 ✅"
        exit 0
    else
        echo "⚠️ ${DIR_NAME} 存在文件差异，请检查"
        exit 1
    fi
else
    echo "❌ ${DIR_NAME} 文件数量不一致，传输可能不完整"
    exit 1
fi
