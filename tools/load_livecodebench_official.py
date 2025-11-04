#!/usr/bin/env python3
"""
使用官方LiveCodeBench代码加载数据的脚本
基于官方的load_code_generation_dataset函数
"""

import json
import os
from datetime import datetime
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class CodeGenerationProblem:
    """LiveCodeBench问题的数据结构"""
    problem_id: str
    contest_date: datetime
    problem_statement: str
    starter_code: str
    function_name: str
    test_cases: List[dict]
    # 可能还有其他字段，根据实际数据调整
    
    def __init__(self, **kwargs):
        # 灵活处理字段，避免缺少字段的错误
        for key, value in kwargs.items():
            if key == 'contest_date' and isinstance(value, str):
                # 处理日期字符串
                try:
                    value = datetime.strptime(value, "%Y-%m-%d")
                except:
                    # 如果日期格式不同，尝试其他格式
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        print(f"Warning: Could not parse date {value}")
                        value = datetime.now()
            setattr(self, key, value)

def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> List[CodeGenerationProblem]:
    """
    官方的LiveCodeBench数据加载函数
    
    Args:
        release_version: 版本标签，如 "release_v5", "release_v6"
        start_date: 开始日期，格式 "YYYY-MM-DD"
        end_date: 结束日期，格式 "YYYY-MM-DD"
    
    Returns:
        List[CodeGenerationProblem]: 问题列表
    """
    print(f"🚀 加载 LiveCodeBench {release_version}")
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    
    try:
        dataset = load_dataset("livecodebench/code_generation_lite", 
                             split="test", 
                             version_tag=release_version, 
                             trust_remote_code=True)
        
        print(f"📊 原始数据量: {len(dataset)}")
        
        # 显示数据结构
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"🔍 数据字段: {list(sample.keys())}")
        
        # 转换为CodeGenerationProblem对象
        try:
            dataset = [CodeGenerationProblem(**p) for p in dataset]
        except Exception as e:
            print(f"⚠️  转换为CodeGenerationProblem时出错: {e}")
            print("使用原始字典格式...")
            # 如果转换失败，保持原始格式
            dataset = list(dataset)
        
        # 按日期筛选
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if hasattr(dataset[0], 'contest_date'):
                dataset = [e for e in dataset if p_start_date <= e.contest_date]
            else:
                print("⚠️  数据中没有contest_date字段，跳过开始日期筛选")

        if end_date is not None:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            if hasattr(dataset[0], 'contest_date'):
                dataset = [e for e in dataset if e.contest_date <= p_end_date]
            else:
                print("⚠️  数据中没有contest_date字段，跳过结束日期筛选")

        print(f"✅ 筛选后数据量: {len(dataset)}")
        return dataset
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

def save_filtered_dataset(dataset, output_path):
    """保存筛选后的数据集"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换为可序列化的格式
    serializable_data = []
    for item in dataset:
        if hasattr(item, '__dict__'):
            # 如果是对象，转换为字典
            item_dict = item.__dict__.copy()
            # 处理datetime对象
            for key, value in item_dict.items():
                if isinstance(value, datetime):
                    item_dict[key] = value.isoformat()
            serializable_data.append(item_dict)
        else:
            # 如果已经是字典，直接使用
            serializable_data.append(dict(item))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path) / (1024*1024)
    print(f"💾 已保存到: {output_path} ({file_size:.2f} MB)")

def main():
    """主函数 - 复现你的查询"""
    print("🚀 使用官方方法加载LiveCodeBench数据")
    print("=" * 50)
    
    # 测试不同的版本和日期范围
    test_cases = [
        {
            "version": "release_v5",
            "start_date": "2024-08-01",
            "end_date": "2025-02-01",
            "output": "./data/test/livecodebench_v5_official_filtered.json",
            "description": "v5版本，2024.08.01-2025.02.01"
        },
        {
            "version": "release_v5", 
            "start_date": None,
            "end_date": None,
            "output": "./data/test/livecodebench_v5_official_full.json",
            "description": "v5版本，完整数据"
        },
        {
            "version": "release_v6",
            "start_date": "2025-02-01", 
            "end_date": "2025-05-01",
            "output": "./data/test/livecodebench_v6_official_filtered.json",
            "description": "v6版本，2025.02.01-2025.05.01"
        }
    ]
    
    results = {}
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"📋 测试: {test_case['description']}")
        print(f"{'='*50}")
        
        try:
            dataset = load_code_generation_dataset(
                release_version=test_case["version"],
                start_date=test_case["start_date"],
                end_date=test_case["end_date"]
            )
            
            if dataset:
                save_filtered_dataset(dataset, test_case["output"])
                results[test_case["version"]] = len(dataset)
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 结果总结")
    print(f"{'='*50}")
    
    for version, count in results.items():
        print(f"📁 {version}: {count} 条数据")
    
    # 对比分析
    if "release_v5" in results:
        print(f"\n💡 分析:")
        print(f"  - 你现有的数据: 279 条")
        print(f"  - 官方v5完整版: {results.get('release_v5', 'N/A')} 条")
        
        if results.get("release_v5", 0) > 279:
            print(f"  - 差异: 官方版本包含更多数据")
            print(f"  - 原因: ArcherCodeR使用的是精选子集")

if __name__ == "__main__":
    main()
