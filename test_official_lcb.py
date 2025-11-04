#!/usr/bin/env python3
"""
测试官方LiveCodeBench数据加载方法
复现你提供的代码逻辑
"""

from datasets import load_dataset
from datetime import datetime

def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None):
    """
    官方的LiveCodeBench数据加载函数（简化版）
    """
    print(f"🚀 加载 {release_version}")
    if start_date or end_date:
        print(f"📅 日期筛选: {start_date} 到 {end_date}")
    
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
            
            # 检查是否有contest_date字段
            if 'contest_date' in sample:
                print(f"📅 样本日期: {sample['contest_date']}")
            else:
                print("⚠️  没有找到contest_date字段")
                print(f"📝 样本数据: {dict(list(sample.items())[:3])}")
        
        # 简化版本：不转换为对象，直接使用字典
        dataset_list = list(dataset)
        
        # 按日期筛选（如果有contest_date字段）
        if start_date is not None and len(dataset_list) > 0 and 'contest_date' in dataset_list[0]:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            original_count = len(dataset_list)
            dataset_list = [e for e in dataset_list if datetime.strptime(e['contest_date'], "%Y-%m-%d") >= p_start_date]
            print(f"📅 开始日期筛选: {original_count} -> {len(dataset_list)}")

        if end_date is not None and len(dataset_list) > 0 and 'contest_date' in dataset_list[0]:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            original_count = len(dataset_list)
            dataset_list = [e for e in dataset_list if datetime.strptime(e['contest_date'], "%Y-%m-%d") <= p_end_date]
            print(f"📅 结束日期筛选: {original_count} -> {len(dataset_list)}")

        print(f"✅ 最终数据量: {len(dataset_list)}")
        return dataset_list
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return []

def main():
    """测试你的具体查询"""
    print("🧪 测试官方LiveCodeBench数据加载")
    print("=" * 50)
    
    # 你的查询
    print("\n📋 执行你的查询:")
    print('lcb_dataset = load_code_generation_dataset(release_version="release_v5", start_date="2024-08-01", end_date="2025-02-01")')
    
    lcb_dataset = load_code_generation_dataset(
        release_version="release_v5", 
        start_date="2024-08-01", 
        end_date="2025-02-01"
    )
    
    print(f"\n📊 结果分析:")
    print(f"  - 筛选后数据量: {len(lcb_dataset)}")
    print(f"  - 你现有数据量: 279")
    
    if len(lcb_dataset) == 279:
        print("✅ 数量匹配！你的数据可能就是用这种方法筛选的")
    elif len(lcb_dataset) > 279:
        print(f"📈 官方数据更多 (+{len(lcb_dataset) - 279})")
        print("💡 你的数据可能是进一步筛选的结果")
    else:
        print(f"📉 官方数据较少 (-{279 - len(lcb_dataset)})")
    
    # 额外测试：不筛选日期的完整v5数据
    print(f"\n{'='*50}")
    print("📋 测试完整v5数据（无日期筛选）:")
    
    full_dataset = load_code_generation_dataset(release_version="release_v5")
    print(f"  - 完整v5数据量: {len(full_dataset)}")
    
    if len(full_dataset) > 0:
        # 分析日期分布
        dates = []
        for item in full_dataset:
            if 'contest_date' in item:
                dates.append(item['contest_date'])
        
        if dates:
            print(f"  - 日期范围: {min(dates)} 到 {max(dates)}")
            
            # 统计在你指定日期范围内的数据
            in_range = [d for d in dates if "2024-08-01" <= d <= "2025-02-01"]
            print(f"  - 2024-08-01到2025-02-01范围内: {len(in_range)} 条")

if __name__ == "__main__":
    main()
