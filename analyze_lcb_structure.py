#!/usr/bin/env python3
"""
基于LiveCodeBench数据集脚本分析其结构
"""

# 从脚本中提取的版本映射信息
ALLOWED_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}

def analyze_version_structure():
    """分析版本结构"""
    print("🔍 LiveCodeBench 版本结构分析")
    print("=" * 50)
    
    print("📋 各版本包含的文件:")
    for version, files in ALLOWED_FILES.items():
        print(f"  {version}: {len(files)} 个文件")
        for file in files:
            print(f"    - {file}")
        print()
    
    print("💡 关键发现:")
    print("  - 每个版本都是累积的（包含之前版本的所有文件）")
    print("  - release_v5 包含 test.jsonl 到 test5.jsonl")
    print("  - release_v6 包含 test.jsonl 到 test6.jsonl")
    print("  - 每个 testX.jsonl 可能对应不同的时间段")

def explain_279_data():
    """解释279条数据的来源"""
    print(f"\n🎯 你的279条数据分析")
    print("=" * 50)
    
    print("基于脚本分析，可能的原因:")
    print("1. 📁 文件组合:")
    print("   - release_v5 包含 5 个 jsonl 文件")
    print("   - 你的数据可能来自其中某个特定文件")
    print("   - 或者是多个文件的特定时间段筛选结果")
    
    print("\n2. 📅 时间筛选:")
    print("   - 每个文件可能对应不同的时间段")
    print("   - 2024-08-01 到 2025-02-01 的筛选可能只涉及部分文件")
    
    print("\n3. 🔍 数据字段:")
    print("   - contest_date: 比赛日期（用于时间筛选）")
    print("   - platform: 平台（LeetCode/AtCoder/Codeforces）")
    print("   - difficulty: 难度等级")
    print("   - 可能还有其他筛选条件")

def suggest_solutions():
    """建议解决方案"""
    print(f"\n🚀 解决方案建议")
    print("=" * 50)
    
    print("1. 📥 直接下载原始文件:")
    print("   - 从 Hugging Face 仓库直接下载 jsonl 文件")
    print("   - 绕过 datasets 库的脚本限制")
    
    print("\n2. 🔧 使用旧版本 datasets 库:")
    print("   - 降级到支持数据集脚本的版本")
    print("   - pip install datasets==2.14.0")
    
    print("\n3. 🎯 手动重建筛选逻辑:")
    print("   - 下载所有相关的 jsonl 文件")
    print("   - 按照时间范围手动筛选")
    print("   - 复现 279 条数据的生成过程")

def create_download_urls():
    """生成下载链接"""
    print(f"\n🔗 直接下载链接")
    print("=" * 50)
    
    base_url = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
    
    print("release_v5 相关文件:")
    for file in ALLOWED_FILES["release_v5"]:
        url = base_url + file
        print(f"  {file}: {url}")
    
    print(f"\n💡 使用方法:")
    print("  wget <URL> 或在浏览器中直接下载")

if __name__ == "__main__":
    analyze_version_structure()
    explain_279_data()
    suggest_solutions()
    create_download_urls()
