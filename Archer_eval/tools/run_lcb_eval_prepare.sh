#!/bin/bash
#
# LCB 评估环境准备脚本
# 克隆 Archer_LiveCodeBench 仓库
#

set -e

BASE_DIR=/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/Archer_eval
LCB_DIR=${BASE_DIR}/LiveCodeBench
LCB_REPO="https://github.com/Jinxiaolong1129/Archer_LiveCodeBench.git"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "LCB 评估环境准备"
log "=============================================="

if [ -d "$LCB_DIR" ]; then
    log "LiveCodeBench 目录已存在: $LCB_DIR"
    log "是否要删除并重新克隆? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log "删除现有目录..."
        rm -rf "$LCB_DIR"
        log "克隆仓库: $LCB_REPO"
        git clone "$LCB_REPO" "$LCB_DIR"
        log "✓ 克隆完成!"
    else
        log "保留现有目录，跳过克隆"
        
        # 可选：拉取最新更新
        log "是否要拉取最新更新? (y/n)"
        read -r update_response
        if [[ "$update_response" =~ ^[Yy]$ ]]; then
            cd "$LCB_DIR"
            log "拉取最新更新..."
            git pull
            log "✓ 更新完成!"
        fi
    fi
else
    log "克隆仓库: $LCB_REPO"
    git clone "$LCB_REPO" "$LCB_DIR"
    log "✓ 克隆完成!"
fi

log "=============================================="
log "LiveCodeBench 目录: $LCB_DIR"
log "=============================================="

# 显示目录结构
log "目录结构:"
ls -la "$LCB_DIR"






