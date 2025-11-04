#!/usr/bin/env python3
"""
简化的JSON数据分析脚本
"""

import json
import os
from typing import Dict, Any

def analyze_json_file_simple(file_path: str) -> Dict[str, Any]:
    """简化版本的JSON文件分析"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试读取第一行来判断文件格式
            first_line = f.readline().strip()
            f.seek(0)  # 重置文件指针
            
            if first_line.startswith('['):
                # JSON数组格式
                print("📋 格式: JSON数组")
                data = json.load(f)
                total_entries = len(data)
                
                # 分析第一个条目的结构
                if data:
                    sample = data[0]
                    print(f"📊 数据总量: {total_entries:,} 条")
                    print("🔍 数据结构 (第一条样本):")
                    print_structure(sample, indent=2)
                    
                    # 显示前2个样本的简化内容
                    print("\n📝 样本数据:")
                    for i, item in enumerate(data[:2], 1):
                        print(f"  样本 {i}:")
                        print_sample_content(item, indent=4)
                        if i < 2:
                            print()
                
            else:
                # JSONL格式
                print("📋 格式: JSONL (每行一个JSON)")
                line_count = 0
                sample_data = []
                
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            line_count += 1
                            if len(sample_data) < 2:
                                sample_data.append(entry)
                        except json.JSONDecodeError:
                            continue
                
                print(f"📊 数据总量: {line_count:,} 条")
                
                if sample_data:
                    print("🔍 数据结构 (第一条样本):")
                    print_structure(sample_data[0], indent=2)
                    
                    print("\n📝 样本数据:")
                    for i, item in enumerate(sample_data, 1):
                        print(f"  样本 {i}:")
                        print_sample_content(item, indent=4)
                        if i < len(sample_data):
                            print()
                
                return {"total_entries": line_count, "file_size_mb": round(file_size / (1024*1024), 2)}
                
        return {"total_entries": total_entries, "file_size_mb": round(file_size / (1024*1024), 2)}
        
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}

def print_structure(obj, indent=0, max_depth=3, current_depth=0):
    """递归打印数据结构"""
    if current_depth >= max_depth:
        return
    
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:5]:  # 只显示前5个键
            if isinstance(value, dict):
                print(f"{prefix}- {key}: dict")
                print_structure(value, indent + 2, max_depth, current_depth + 1)
            elif isinstance(value, list):
                print(f"{prefix}- {key}: list[{len(value)}]")
                if value and current_depth < max_depth - 1:
                    print_structure(value[0], indent + 2, max_depth, current_depth + 1)
            else:
                print(f"{prefix}- {key}: {type(value).__name__}")
        
        if len(obj) > 5:
            print(f"{prefix}... ({len(obj) - 5} more fields)")
    
    elif isinstance(obj, list) and obj:
        print_structure(obj[0], indent, max_depth, current_depth)

def print_sample_content(obj, indent=0):
    """打印样本内容的简化版本"""
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:3]:  # 只显示前3个字段
            if isinstance(value, str):
                # 截断长字符串
                display_value = value[:100] + "..." if len(value) > 100 else value
                print(f"{prefix}{key}: {repr(display_value)}")
            elif isinstance(value, list):
                print(f"{prefix}{key}: list[{len(value)}] {value[:2] if len(value) <= 2 else str(value[:2]) + '...'}")
            elif isinstance(value, dict):
                print(f"{prefix}{key}: dict with {len(value)} keys")
            else:
                print(f"{prefix}{key}: {value}")
        
        if len(obj) > 3:
            print(f"{prefix}... ({len(obj) - 3} more fields)")

def main():
    """主函数"""
    print("🚀 JSON数据文件分析")
    
    files_to_analyze = [
        "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/train/archercoder-1.5b-train.json",
        "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/test/livecodebench_v5.json"
    ]
    
    results = {}
    total_entries = 0
    total_size = 0
    
    for file_path in files_to_analyze:
        result = analyze_json_file_simple(file_path)
        if result:
            results[file_path] = result
            total_entries += result.get("total_entries", 0)
            total_size += result.get("file_size_mb", 0)
    
    print(f"\n{'='*50}")
    print("📊 总结报告")
    print(f"{'='*50}")
    print(f"🎯 总计: {total_entries:,} 条数据")
    print(f"💾 总大小: {total_size:.2f} MB")
    
    for file_path, result in results.items():
        filename = os.path.basename(file_path)
        print(f"📁 {filename}: {result.get('total_entries', 0):,} 条, {result.get('file_size_mb', 0):.2f} MB")

if __name__ == "__main__":
    main()

