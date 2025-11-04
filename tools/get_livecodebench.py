#!/usr/bin/env python3
"""
简单的LiveCodeBench数据获取脚本
支持命令行参数指定版本
"""

import os
import json
import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description='下载LiveCodeBench数据集')
    parser.add_argument('--version', '-v', 
                       choices=['v5', 'v6', 'both'], 
                       default='both',
                       help='要下载的版本 (v5, v6, 或 both)')
    parser.add_argument('--output-dir', '-o',
                       default='./data/test',
                       help='输出目录 (默认: ./data/test)')
    
    args = parser.parse_args()
    
    # 版本映射
    version_mapping = {
        'v5': {
            'tag': 'release_v5',
            'filename': 'livecodebench_v5.json',
            'period': '2024.08.01-2025.02.01'
        },
        'v6': {
            'tag': 'release_v6', 
            'filename': 'livecodebench_v6.json',
            'period': '2025.02.01-2025.05.01'
        }
    }
    
    # 确定要下载的版本
    if args.version == 'both':
        versions_to_download = ['v5', 'v6']
    else:
        versions_to_download = [args.version]
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 LiveCodeBench 数据集下载")
    print("=" * 40)
    
    for version in versions_to_download:
        version_info = version_mapping[version]
        output_path = os.path.join(args.output_dir, version_info['filename'])
        
        print(f"\n📥 下载 LiveCodeBench {version.upper()}")
        print(f"📅 时间范围: {version_info['period']}")
        print(f"💾 保存路径: {output_path}")
        
        try:
            # 加载数据集
            dataset = load_dataset("livecodebench/code_generation_lite", 
                                 version_tag=version_info['tag'])
            
            # 获取数据
            if 'test' in dataset:
                data = dataset['test']
            else:
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
            
            print(f"📊 数据条数: {len(data)}")
            
            # 转换并保存
            data_list = [dict(item) for item in data]
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(output_path) / (1024*1024)
            print(f"✅ 下载完成! 文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
    
    print(f"\n🎉 所有下载任务完成!")

if __name__ == "__main__":
    main()

