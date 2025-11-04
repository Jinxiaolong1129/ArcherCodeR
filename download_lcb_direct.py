#!/usr/bin/env python3
"""
直接下载LiveCodeBench的jsonl文件并分析
"""

import json
import os
import requests
from datetime import datetime
from urllib.parse import urlparse

def download_file(url, local_path):
    """下载文件"""
    try:
        print(f"📥 下载 {os.path.basename(local_path)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(local_path) / (1024*1024)
        print(f"✅ 下载完成: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def analyze_jsonl_file(file_path):
    """分析单个jsonl文件"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return {}
    
    print(f"\n🔍 分析 {os.path.basename(file_path)}")
    print("-" * 30)
    
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  第{line_num}行JSON解析错误: {e}")
        
        print(f"📊 数据条数: {len(data)}")
        
        if data:
            # 分析数据结构
            sample = data[0]
            print(f"🔍 数据字段: {list(sample.keys())}")
            
            # 分析日期分布
            if 'contest_date' in sample:
                dates = [item['contest_date'] for item in data if 'contest_date' in item]
                if dates:
                    print(f"📅 日期范围: {min(dates)} 到 {max(dates)}")
                    
                    # 统计在特定日期范围内的数据
                    in_range = [d for d in dates if "2024-08-01" <= d <= "2025-02-01"]
                    print(f"📅 2024-08-01到2025-02-01: {len(in_range)} 条")
            
            # 分析平台分布
            if 'platform' in sample:
                platforms = {}
                for item in data:
                    if 'platform' in item:
                        platform = item['platform']
                        platforms[platform] = platforms.get(platform, 0) + 1
                print(f"🏛️  平台分布: {platforms}")
            
            # 分析难度分布
            if 'difficulty' in sample:
                difficulties = {}
                for item in data:
                    if 'difficulty' in item:
                        difficulty = item['difficulty']
                        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
                print(f"⭐ 难度分布: {difficulties}")
        
        return {
            'file': os.path.basename(file_path),
            'count': len(data),
            'data': data
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return {}

def filter_by_date_range(data_list, start_date, end_date):
    """按日期范围筛选数据"""
    filtered = []
    for item in data_list:
        if 'contest_date' in item:
            contest_date = item['contest_date']
            if start_date <= contest_date <= end_date:
                filtered.append(item)
    return filtered

def main():
    """主函数"""
    print("🚀 直接下载LiveCodeBench数据文件")
    print("=" * 50)
    
    # 定义下载文件
    base_url = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
    files_to_download = [
        "test.jsonl",
        "test2.jsonl", 
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl"
    ]
    
    download_dir = "./data/lcb_raw"
    
    # 下载文件
    downloaded_files = []
    for file_name in files_to_download:
        url = base_url + file_name
        local_path = os.path.join(download_dir, file_name)
        
        if os.path.exists(local_path):
            print(f"✅ {file_name} 已存在，跳过下载")
            downloaded_files.append(local_path)
        else:
            if download_file(url, local_path):
                downloaded_files.append(local_path)
    
    # 分析每个文件
    print(f"\n{'='*50}")
    print("📊 文件分析")
    print(f"{'='*50}")
    
    all_data = []
    file_stats = []
    
    for file_path in downloaded_files:
        result = analyze_jsonl_file(file_path)
        if result:
            file_stats.append(result)
            all_data.extend(result['data'])
    
    # 综合分析
    print(f"\n{'='*50}")
    print("📊 综合分析")
    print(f"{'='*50}")
    
    print(f"📁 总文件数: {len(file_stats)}")
    print(f"📊 总数据量: {len(all_data)}")
    
    # 按日期筛选（复现你的查询）
    if all_data:
        filtered_data = filter_by_date_range(all_data, "2024-08-01", "2025-02-01")
        print(f"🎯 2024-08-01到2025-02-01筛选结果: {len(filtered_data)} 条")
        
        if len(filtered_data) == 279:
            print("✅ 完美匹配！这就是你279条数据的来源")
        elif abs(len(filtered_data) - 279) <= 5:
            print(f"🎯 非常接近！差异: {len(filtered_data) - 279}")
        else:
            print(f"📊 与你的数据差异: {len(filtered_data) - 279}")
        
        # 保存筛选结果
        output_path = "./data/test/livecodebench_v5_reconstructed.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_path) / (1024*1024)
        print(f"💾 已保存重建数据到: {output_path} ({file_size:.2f} MB)")
    
    # 文件统计
    print(f"\n📋 各文件统计:")
    for stat in file_stats:
        print(f"  {stat['file']}: {stat['count']} 条")

if __name__ == "__main__":
    main()
