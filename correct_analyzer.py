#!/usr/bin/env python3
"""
正确的JSON数据分析脚本
"""

import json
import os

def analyze_archercoder_correctly(file_path: str):
    """正确分析archercoder文件"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("=" * 60)
    
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            print("📋 格式: JSON对象")
            print(f"📊 顶层字段数: {len(data)}")
            
            # 检查数据结构
            print("\n🔍 顶层字段:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, dict):
                    print(f"  - {key}: dict ({len(value)} 个条目)")
                elif isinstance(value, list):
                    print(f"  - {key}: list ({len(value)} 个元素)")
                else:
                    print(f"  - {key}: {type(value).__name__}")
            
            # 假设这是按字段组织的数据，每个字段包含多个样本
            # 检查第一个字段来确定实际的数据条数
            first_field = list(data.keys())[0]
            if isinstance(data[first_field], dict):
                actual_data_count = len(data[first_field])
                print(f"\n📊 实际数据条数: {actual_data_count:,}")
                
                # 分析单个数据条目的结构
                print("\n🔍 单个数据条目结构:")
                sample_key = list(data[first_field].keys())[0]
                
                for field_name, field_data in data.items():
                    if isinstance(field_data, dict) and sample_key in field_data:
                        sample_value = field_data[sample_key]
                        if isinstance(sample_value, str):
                            print(f"  - {field_name}: str (长度: {len(sample_value)})")
                        elif isinstance(sample_value, list):
                            print(f"  - {field_name}: list[{len(sample_value)}]")
                            if sample_value and isinstance(sample_value[0], dict):
                                inner_keys = list(sample_value[0].keys())
                                print(f"    └─ 包含字段: {inner_keys}")
                        elif isinstance(sample_value, dict):
                            print(f"  - {field_name}: dict ({len(sample_value)} keys)")
                            inner_keys = list(sample_value.keys())[:3]
                            print(f"    └─ 包含字段: {inner_keys}{'...' if len(sample_value) > 3 else ''}")
                        else:
                            print(f"  - {field_name}: {type(sample_value).__name__}")
                
                # 显示样本数据
                print(f"\n📝 样本数据 (条目 {sample_key}):")
                for field_name, field_data in list(data.items())[:4]:  # 只显示前4个字段
                    if isinstance(field_data, dict) and sample_key in field_data:
                        sample_value = field_data[sample_key]
                        if isinstance(sample_value, str):
                            display_value = sample_value[:100] + "..." if len(sample_value) > 100 else sample_value
                            print(f"  {field_name}: '{display_value}'")
                        elif isinstance(sample_value, list) and len(sample_value) == 1 and isinstance(sample_value[0], dict):
                            inner_dict = sample_value[0]
                            print(f"  {field_name}: [{{")
                            for k, v in list(inner_dict.items())[:2]:
                                v_display = v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v
                                print(f"      {k}: {repr(v_display)}")
                            if len(inner_dict) > 2:
                                print(f"      ... ({len(inner_dict) - 2} more)")
                            print("  }]")
                        else:
                            print(f"  {field_name}: {type(sample_value).__name__} - {sample_value}")
                
                return {"total_entries": actual_data_count, "file_size_mb": round(file_size / (1024*1024), 2)}
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}

def analyze_livecodebench_correctly(file_path: str):
    """正确分析livecodebench文件"""
    print(f"\n分析文件: {os.path.basename(file_path)}")
    print("=" * 60)
    
    file_size = os.path.getsize(file_path)
    print(f"📁 文件大小: {file_size / (1024*1024):.2f} MB")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            print("📋 格式: JSON数组")
            total_entries = len(data)
            print(f"📊 数据条数: {total_entries:,}")
            
            if data:
                print("\n🔍 数据结构 (基于第一条):")
                sample = data[0]
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"  - {key}: str")
                    elif isinstance(value, list):
                        print(f"  - {key}: list[{len(value)}]")
                        if value and isinstance(value[0], dict):
                            inner_keys = list(value[0].keys())
                            print(f"    └─ 元素字段: {inner_keys}")
                    elif isinstance(value, dict):
                        print(f"  - {key}: dict")
                        inner_keys = list(value.keys())
                        print(f"    └─ 包含字段: {inner_keys}")
                    else:
                        print(f"  - {key}: {type(value).__name__}")
                
                # 显示样本数据
                print(f"\n📝 样本数据:")
                for i, item in enumerate(data[:2], 1):
                    print(f"  样本 {i}:")
                    for key, value in list(item.items())[:4]:
                        if isinstance(value, str):
                            display_value = value[:80] + "..." if len(value) > 80 else value
                            print(f"    {key}: '{display_value}'")
                        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                            inner_dict = value[0]
                            print(f"    {key}: [{{")
                            for k, v in list(inner_dict.items())[:2]:
                                v_display = v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v
                                print(f"        {k}: {repr(v_display)}")
                            if len(inner_dict) > 2:
                                print(f"        ... ({len(inner_dict) - 2} more)")
                            print("    }]")
                        else:
                            print(f"    {key}: {type(value).__name__}")
                    if i < 2:
                        print()
            
            return {"total_entries": total_entries, "file_size_mb": round(file_size / (1024*1024), 2)}
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return {}

def main():
    """主函数"""
    print("🚀 JSON数据文件正确分析")
    
    archercoder_file = "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/train/archercoder-1.5b-train.json"
    livecodebench_file = "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/data/test/livecodebench_v5.json"
    
    results = {}
    
    # 分析两个文件
    result1 = analyze_archercoder_correctly(archercoder_file)
    if result1:
        results["archercoder"] = result1
    
    result2 = analyze_livecodebench_correctly(livecodebench_file)
    if result2:
        results["livecodebench"] = result2
    
    # 总结报告
    print(f"\n{'='*60}")
    print("📊 最终总结报告")
    print(f"{'='*60}")
    
    total_entries = sum(r.get("total_entries", 0) for r in results.values())
    total_size = sum(r.get("file_size_mb", 0) for r in results.values())
    
    print(f"🎯 总数据量: {total_entries:,} 条")
    print(f"💾 总文件大小: {total_size:.2f} MB")
    print()
    
    for name, result in results.items():
        if name == "archercoder":
            print(f"📁 archercoder-1.5b-train.json (训练数据):")
            print(f"   - 数据条数: {result.get('total_entries', 0):,}")
            print(f"   - 文件大小: {result.get('file_size_mb', 0):.2f} MB")
            print(f"   - 数据类型: 代码生成训练数据")
            print(f"   - 格式说明: 按字段组织的JSON对象，每个字段包含所有样本的对应数据")
        else:
            print(f"📁 livecodebench_v5.json (测试数据):")
            print(f"   - 数据条数: {result.get('total_entries', 0):,}")
            print(f"   - 文件大小: {result.get('file_size_mb', 0):.2f} MB") 
            print(f"   - 数据类型: 代码生成测试数据")
            print(f"   - 格式说明: 标准JSON数组，每个元素是一个完整的测试样本")
        print()

if __name__ == "__main__":
    main()

