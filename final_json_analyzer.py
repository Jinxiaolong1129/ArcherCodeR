#!/usr/bin/env python3
"""
最终版JSON数据分析脚本 - 正确处理特殊格式
"""

import json
import os
from typing import Dict, Any

def analyze_archercoder_file(file_path: str):
    """专门分析archercoder文件的特殊格式"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("=" * 50)
    
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取整个文件内容
            content = f.read().strip()
            
            # 解析JSON
            data = json.loads(content)
            
            print("📋 格式: 特殊JSON格式 (字典的字典)")
            
            # 这个文件实际上是一个大字典，每个键对应一个数据条目
            total_entries = len(data)
            print(f"📊 数据总量: {total_entries:,} 条")
            
            # 分析数据结构 - 查看第一个条目
            first_key = list(data.keys())[0]
            first_entry = data[first_key]
            
            print("🔍 数据结构 (第一条样本):")
            print_structure_simple(first_entry, indent=2)
            
            # 显示几个样本
            print("\n📝 样本数据:")
            sample_keys = list(data.keys())[:2]
            for i, key in enumerate(sample_keys, 1):
                print(f"  样本 {i} (key: {key}):")
                entry = data[key]
                print_sample_simple(entry, indent=4)
                if i < len(sample_keys):
                    print()
            
            return {"total_entries": total_entries, "file_size_mb": round(file_size / (1024*1024), 2)}
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}

def analyze_livecodebench_file(file_path: str):
    """分析livecodebench文件"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("=" * 50)
    
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            print("📋 格式: JSON数组")
            total_entries = len(data)
            print(f"📊 数据总量: {total_entries:,} 条")
            
            # 分析第一个条目的结构
            if data:
                print("🔍 数据结构 (第一条样本):")
                print_structure_simple(data[0], indent=2)
                
                # 显示前2个样本
                print("\n📝 样本数据:")
                for i, item in enumerate(data[:2], 1):
                    print(f"  样本 {i}:")
                    print_sample_simple(item, indent=4)
                    if i < 2:
                        print()
            
            return {"total_entries": total_entries, "file_size_mb": round(file_size / (1024*1024), 2)}
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}

def print_structure_simple(obj, indent=0, max_depth=2, current_depth=0):
    """简化的结构打印"""
    if current_depth >= max_depth:
        return
    
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:5]:
            if isinstance(value, dict):
                print(f"{prefix}- {key}: dict ({len(value)} keys)")
                if current_depth < max_depth - 1:
                    print_structure_simple(value, indent + 2, max_depth, current_depth + 1)
            elif isinstance(value, list):
                print(f"{prefix}- {key}: list[{len(value)}]")
                if value and current_depth < max_depth - 1:
                    print(f"{prefix}  └─ 元素类型: {type(value[0]).__name__}")
            else:
                print(f"{prefix}- {key}: {type(value).__name__}")
        
        if len(obj) > 5:
            print(f"{prefix}... ({len(obj) - 5} more fields)")

def print_sample_simple(obj, indent=0):
    """简化的样本内容打印"""
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, value in list(obj.items())[:4]:
            if isinstance(value, str):
                display_value = value[:80] + "..." if len(value) > 80 else value
                print(f"{prefix}{key}: '{display_value}'")
            elif isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], dict):
                    # 特殊处理单元素字典列表
                    inner_dict = value[0]
                    print(f"{prefix}{key}: [{{")
                    for k, v in list(inner_dict.items())[:2]:
                        v_display = v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v
                        print(f"{prefix}    {k}: {repr(v_display)}")
                    if len(inner_dict) > 2:
                        print(f"{prefix}    ... ({len(inner_dict) - 2} more)")
                    print(f"{prefix}}}]")
                else:
                    print(f"{prefix}{key}: list[{len(value)}]")
            elif isinstance(value, dict):
                print(f"{prefix}{key}: dict ({len(value)} keys)")
            else:
                print(f"{prefix}{key}: {value}")
        
        if len(obj) > 4:
            print(f"{prefix}... ({len(obj) - 4} more fields)")

def main():
    """主函数"""
    print("🚀 JSON数据文件详细分析")
    
    # 分别处理两个文件
    archercoder_file = "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/train/archercoder-1.5b-train.json"
    livecodebench_file = "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/test/livecodebench_v5.json"
    
    results = {}
    
    # 分析archercoder文件
    result1 = analyze_archercoder_file(archercoder_file)
    if result1:
        results["archercoder"] = result1
    
    # 分析livecodebench文件
    result2 = analyze_livecodebench_file(livecodebench_file)
    if result2:
        results["livecodebench"] = result2
    
    # 总结报告
    print(f"\n{'='*50}")
    print("📊 总结报告")
    print(f"{'='*50}")
    
    total_entries = sum(r.get("total_entries", 0) for r in results.values())
    total_size = sum(r.get("file_size_mb", 0) for r in results.values())
    
    print(f"🎯 总计: {total_entries:,} 条数据")
    print(f"💾 总大小: {total_size:.2f} MB")
    print()
    
    for name, result in results.items():
        filename = "archercoder-1.5b-train.json" if name == "archercoder" else "livecodebench_v5.json"
        print(f"📁 {filename}:")
        print(f"   - 数据条数: {result.get('total_entries', 0):,}")
        print(f"   - 文件大小: {result.get('file_size_mb', 0):.2f} MB")
        if name == "archercoder":
            print(f"   - 格式: 字典格式，每个键对应一个训练样本")
        else:
            print(f"   - 格式: JSON数组，每个元素是一个测试样本")
        print()

if __name__ == "__main__":
    main()

