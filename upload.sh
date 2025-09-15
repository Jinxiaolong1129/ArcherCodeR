#!/bin/bash

# 上传并清理脚本
# 安全地上传文件到 OneDrive 并删除本地文件

set -e  # 遇到错误立即退出

# 配置变量
BASE_LOCAL_PATH="/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR"
BASE_REMOTE_PATH="onedrive:ArcherCodeR"

# 要处理的目录列表
DIRECTORIES=(
    "Archer-Qwen2.5-3B-2K-8K-16resp"
    "Archer-Qwen2.5-3B-2K-16K-16resp"
)

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查 rclone 是否安装
if ! command -v rclone &> /dev/null; then
    log "错误: rclone 未安装"
    exit 1
fi

# 上传并验证函数
upload_and_verify() {
    local local_dir="$1"
    local remote_dir="$2"
    local dir_name="$3"
    
    log "开始处理目录: $dir_name"
    
    # 检查本地目录是否存在
    if [ ! -d "$local_dir" ]; then
        log "错误: 本地目录不存在: $local_dir"
        return 1
    fi
    
    # 获取本地文件总数和大小
    local_count=$(find "$local_dir" -type f | wc -l)
    local_size=$(du -sh "$local_dir" | cut -f1)
    log "本地目录统计 - 文件数: $local_count, 大小: $local_size"
    
    # 执行上传
    log "开始上传: $local_dir -> $remote_dir"
    if rclone copy "$local_dir" "$remote_dir" -P --stats=1s -v; then
        log "上传完成: $dir_name"
    else
        log "错误: 上传失败: $dir_name"
        return 1
    fi
    
    # 验证上传
    log "验证上传完整性..."
    if rclone check "$local_dir" "$remote_dir" --one-way; then
        log "验证成功: 远程文件与本地文件匹配"
    else
        log "错误: 验证失败，远程文件与本地文件不匹配"
        return 1
    fi
    
    # 获取远程文件统计
    remote_count=$(rclone size "$remote_dir" --json | jq -r '.count')
    remote_size=$(rclone size "$remote_dir" --json | jq -r '.sizeByte')
    remote_size_human=$(numfmt --to=iec "$remote_size")
    
    log "远程目录统计 - 文件数: $remote_count, 大小: $remote_size_human"
    
    # 比较文件数量
    if [ "$local_count" -eq "$remote_count" ]; then
        log "文件数量验证通过: $local_count = $remote_count"
    else
        log "错误: 文件数量不匹配 - 本地: $local_count, 远程: $remote_count"
        return 1
    fi
    
    return 0
}

# 安全删除函数
safe_delete() {
    local dir_path="$1"
    local dir_name="$2"
    
    log "准备删除本地目录: $dir_path"
    
    # 二次确认
    read -p "确认删除本地目录 '$dir_name'? (输入 'YES' 确认): " confirm
    if [ "$confirm" != "YES" ]; then
        log "取消删除操作"
        return 1
    fi
    
    # 删除目录
    log "正在删除: $dir_path"
    if rm -rf "$dir_path"; then
        log "成功删除: $dir_name"
    else
        log "错误: 删除失败: $dir_name"
        return 1
    fi
}

# 主程序
main() {
    log "开始执行上传和清理任务"
    
    for dir_name in "${DIRECTORIES[@]}"; do
        local_path="$BASE_LOCAL_PATH/$dir_name"
        remote_path="$BASE_REMOTE_PATH/$dir_name"
        
        log "=================================="
        log "处理目录: $dir_name"
        log "本地路径: $local_path"
        log "远程路径: $remote_path"
        log "=================================="
        
        # 上传并验证
        if upload_and_verify "$local_path" "$remote_path" "$dir_name"; then
            log "上传和验证成功，准备删除本地文件"
            
            # 安全删除
            if safe_delete "$local_path" "$dir_name"; then
                log "目录处理完成: $dir_name"
            else
                log "警告: 删除操作被取消或失败: $dir_name"
            fi
        else
            log "错误: 上传或验证失败，跳过删除操作: $dir_name"
            continue
        fi
        
        log "等待 5 秒后处理下一个目录..."
        sleep 5
    done
    
    log "所有任务完成"
}

# 运行主程序
main

# 最终报告
log "脚本执行完成"
log "请检查日志确认所有操作是否成功"