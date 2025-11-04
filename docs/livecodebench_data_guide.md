# LiveCodeBench 数据获取指南

## 概述

LiveCodeBench 是一个持续更新的代码评测基准，定期从 LeetCode、AtCoder 和 Codeforces 等竞赛平台收集新的编程问题。本指南介绍如何获取 LiveCodeBench v5 和 v6 版本的数据。

## 版本信息

| 版本 | 时间范围 | 描述 |
|------|----------|------|
| v5 | 2024.08.01 - 2025.02.01 | 包含约880个编程问题 |
| v6 | 2025.02.01 - 2025.05.01 | 最新版本，包含更多问题 |

## 获取方式

### 方法1: 使用项目提供的脚本

#### 快速下载（推荐）
```bash
# 下载所有版本
python tools/get_livecodebench.py

# 只下载v5版本
python tools/get_livecodebench.py --version v5

# 只下载v6版本  
python tools/get_livecodebench.py --version v6

# 指定输出目录
python tools/get_livecodebench.py --output-dir ./my_data
```

#### 详细下载（包含数据分析）
```bash
python tools/download_livecodebench.py
```

### 方法2: 直接使用 Hugging Face datasets

```python
from datasets import load_dataset

# 加载 v5 版本
dataset_v5 = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")

# 加载 v6 版本
dataset_v6 = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")

# 保存为本地文件
import json

data_v5 = [dict(item) for item in dataset_v5['test']]
with open('./data/test/livecodebench_v5.json', 'w') as f:
    json.dump(data_v5, f, indent=2)
```

### 方法3: 从 Hugging Face Hub 直接下载

访问 [LiveCodeBench 数据集页面](https://huggingface.co/datasets/livecodebench/code_generation_lite) 直接下载所需版本。

## 数据格式

LiveCodeBench 数据集采用标准的JSON格式，每个条目包含以下字段：

```json
{
  "data_source": "livecodebench",
  "prompt": [
    {
      "role": "user", 
      "content": "编程问题描述..."
    }
  ],
  "ability": "code",
  "reward_model": {
    "style": "rule",
    "ground_truth": "测试用例和预期输出..."
  },
  "extra_info": {
    "split": "test",
    "index": 0,
    "reference": null
  }
}
```

## 使用要求

### 环境依赖
```bash
pip install datasets
pip install huggingface_hub
```

### 注意事项

1. **网络要求**: 需要稳定的网络连接访问 Hugging Face
2. **存储空间**: 每个版本约3-4GB，请确保有足够的磁盘空间
3. **使用协议**: 请遵守 LiveCodeBench 的使用协议和许可条款
4. **数据更新**: LiveCodeBench 会定期发布新版本，建议关注官方更新

## 常见问题

### Q: 下载速度慢怎么办？
A: 可以设置 Hugging Face 镜像或使用代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 如何验证数据完整性？
A: 使用项目提供的分析脚本：
```bash
python correct_analyzer.py
```

### Q: v5和v6版本有什么区别？
A: 
- v5: 2024年8月到2025年2月的问题，约880个
- v6: 2025年2月到2025年5月的问题，包含更新的编程挑战

### Q: 数据格式与现有数据兼容吗？
A: 是的，新下载的数据格式与项目中现有的 `livecodebench_v5.json` 完全兼容。

## 相关资源

- [LiveCodeBench 官方论文](https://arxiv.org/abs/2403.07974)
- [Hugging Face 数据集页面](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- [项目 GitHub 仓库](https://github.com/wizard-III/ArcherCodeR)

## 技术支持

如果在数据获取过程中遇到问题，可以：
1. 检查网络连接和防火墙设置
2. 确认 Hugging Face 账户权限
3. 查看项目 Issues 页面寻找解决方案
4. 联系项目维护者获取帮助

