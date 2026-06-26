#!/usr/bin/env python3
"""
下载LiveCodeBench v5和v6数据集的脚本
"""

import os
import json
from datasets import load_dataset
from datetime import datetime

def download_livecodebench_data(version_tag, save_path):
    """
    从Hugging Face下载LiveCodeBench数据集
    
    Args:
        version_tag: 版本标签，如 "release_v5" 或 "release_v6"
        save_path: 保存路径
    """
    print(f"🚀 开始下载 LiveCodeBench {version_tag}...")
    
    try:
        # 加载数据集
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version_tag, trust_remote_code=True)
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"📁 创建目录: {save_dir}")
        
        # 获取数据集信息
        if 'test' in dataset:
            data = dataset['test']
        elif 'train' in dataset:
            data = dataset['train']
        else:
            # 如果没有明确的split，取第一个可用的
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
            print(f"📋 使用数据分割: {split_name}")
        
        print(f"📊 数据条数: {len(data)}")
        
        # 转换为列表格式并保存
        data_list = []
        for item in data:
            data_list.append(dict(item))
        
        # 保存为JSON文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 成功保存到: {save_path}")
        print(f"📁 文件大小: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def show_dataset_info(version_tag):
    """显示数据集信息"""
    try:
        print(f"\n🔍 查看 {version_tag} 数据集信息...")
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag=version_tag, trust_remote_code=True)
        
        print(f"📋 可用分割: {list(dataset.keys())}")
        
        # 获取第一个可用的分割
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"📊 数据条数: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"🔍 数据字段: {list(sample.keys())}")
            
            # 显示样本数据
            print(f"\n📝 样本数据:")
            for key, value in sample.items():
                if isinstance(value, str):
                    display_value = value[:100] + "..." if len(value) > 100 else value
                    print(f"  {key}: {repr(display_value)}")
                else:
                    print(f"  {key}: {type(value).__name__} - {value}")
        
    except Exception as e:
        print(f"❌ 获取信息失败: {e}")

def main():
    """主函数"""
    print("🚀 LiveCodeBench 数据集下载工具")
    print("=" * 50)
    
    # 定义版本和保存路径
    versions = {
        "release_v5": {
            "path": "./data/test/livecodebench_v5.json",
            "description": "2024.08.01-2025.02.01"
        },
        "release_v6": {
            "path": "./data/test/livecodebench_v6.json", 
            "description": "2025.02.01-2025.05.01"
        }
    }
    
    # 首先显示可用版本信息
    print("📋 可用版本:")
    for version, info in versions.items():
        print(f"  - {version}: {info['description']}")
    print()
    
    # 检查是否已安装datasets库
    try:
        import datasets
        print(f"✅ datasets库版本: {datasets.__version__}")
    except ImportError:
        print("❌ 请先安装datasets库: pip install datasets")
        return
    
    # 下载各个版本
    success_count = 0
    for version_tag, info in versions.items():
        print(f"\n{'='*50}")
        
        # 显示数据集信息
        show_dataset_info(version_tag)
        
        # 下载数据集
        if download_livecodebench_data(version_tag, info["path"]):
            success_count += 1
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 下载总结")
    print(f"{'='*50}")
    print(f"✅ 成功下载: {success_count}/{len(versions)} 个版本")
    
    if success_count > 0:
        print(f"\n📁 下载的文件:")
        for version, info in versions.items():
            if os.path.exists(info["path"]):
                file_size = os.path.getsize(info["path"]) / (1024*1024)
                print(f"  - {info['path']}: {file_size:.2f} MB")
    
    print(f"\n💡 使用说明:")
    print(f"  - v5版本包含2024年8月到2025年2月的数据")
    print(f"  - v6版本包含2025年2月到2025年5月的数据")
    print(f"  - 数据格式与现有的livecodebench_v5.json兼容")

if __name__ == "__main__":
    main()

