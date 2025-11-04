#!/usr/bin/env python3
"""
测试最简单的LiveCodeBench数据加载
"""

from datasets import load_dataset
import json

def test_basic_loading():
    """测试基本的数据加载"""
    print("🧪 测试LiveCodeBench数据加载")
    print("=" * 50)
    
    # 测试不同的版本
    versions_to_test = [
        "release_v2",
        "release_v5", 
        "release_v6"
    ]
    
    for version in versions_to_test:
        print(f"\n📋 测试版本: {version}")
        print("-" * 30)
        
        try:
            # 最简单的加载方式
            print(f"🚀 加载 {version}...")
            lcb_codegen = load_dataset("livecodebench/code_generation_lite", version_tag=version)
            
            print(f"✅ 成功加载 {version}")
            print(f"📊 可用splits: {list(lcb_codegen.keys())}")
            
            # 检查每个split的数据量
            for split_name, split_data in lcb_codegen.items():
                print(f"  - {split_name}: {len(split_data)} 条数据")
                
                # 显示第一条数据的结构
                if len(split_data) > 0:
                    sample = split_data[0]
                    print(f"  - 字段: {list(sample.keys())}")
                    
                    # 检查是否有日期字段
                    date_fields = [k for k in sample.keys() if 'date' in k.lower()]
                    if date_fields:
                        print(f"  - 日期字段: {date_fields}")
                        for field in date_fields:
                            print(f"    {field}: {sample[field]}")
            
        except Exception as e:
            print(f"❌ 加载 {version} 失败: {e}")
    
    print(f"\n{'='*50}")
    print("📊 总结")
    print(f"{'='*50}")

def test_v5_specifically():
    """专门测试v5版本"""
    print(f"\n🎯 专门测试v5版本")
    print("-" * 30)
    
    try:
        # 尝试加载v5
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")
        
        if 'test' in dataset:
            test_data = dataset['test']
            print(f"📊 v5测试数据: {len(test_data)} 条")
            
            # 分析日期分布
            if len(test_data) > 0:
                sample = test_data[0]
                print(f"🔍 样本字段: {list(sample.keys())}")
                
                # 查找日期相关字段
                for key, value in sample.items():
                    if 'date' in key.lower() or 'time' in key.lower():
                        print(f"📅 {key}: {value}")
                
                # 如果有contest_date字段，分析日期范围
                if 'contest_date' in sample:
                    dates = [item['contest_date'] for item in test_data if 'contest_date' in item]
                    if dates:
                        print(f"📅 日期范围: {min(dates)} 到 {max(dates)}")
                        
                        # 筛选2024-08-01到2025-02-01的数据
                        filtered = [item for item in test_data 
                                  if 'contest_date' in item and 
                                  "2024-08-01" <= item['contest_date'] <= "2025-02-01"]
                        print(f"🎯 2024-08-01到2025-02-01范围: {len(filtered)} 条")
                        
                        if len(filtered) == 279:
                            print("✅ 匹配！这就是你现有数据的来源")
                        else:
                            print(f"📊 与你的数据(279条)差异: {len(filtered) - 279}")
        
    except Exception as e:
        print(f"❌ v5测试失败: {e}")

def save_sample_data():
    """保存样本数据用于分析"""
    try:
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")
        if 'test' in dataset:
            # 保存前5条数据作为样本
            sample_data = [dict(item) for item in dataset['test'][:5]]
            
            with open('./livecodebench_v5_sample.json', 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 已保存样本数据到: ./livecodebench_v5_sample.json")
    except Exception as e:
        print(f"❌ 保存样本失败: {e}")

if __name__ == "__main__":
    test_basic_loading()
    test_v5_specifically()
    save_sample_data()
