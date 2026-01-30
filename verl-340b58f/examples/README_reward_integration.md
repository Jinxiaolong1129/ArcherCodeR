# VERL Reward Integration with rewards/ Package

## 概述

本文档说明如何在 VERL 中使用 `rewards/` 包中的 reward 函数，同时获得丰富的分析功能。

## 核心配置

### 使用 WizardRewardManager + rewards/general_reward.py

```yaml
reward_model:
  enable: False
  reward_manager: wizard           # 使用 WizardRewardManager 获得丰富分析
  use_general_reward: True         # 使用 rewards/general_reward.py 中的函数
```

## 功能对比

| 配置 | Reward Manager | Compute Score | 分析功能 |
|------|----------------|---------------|----------|
| **推荐配置** | `wizard` | `rewards/general_reward.py` | ✅ 思维词汇分析<br>✅ 重复率检测<br>✅ 响应长度分析<br>✅ 准确率统计 |
| 简单配置 | `naive` | `rewards/general_reward.py` | ❌ 无额外分析 |
| VERL默认 | `naive` | `verl/utils/reward_score` | ❌ 无额外分析 |

## 算法支持

### GRPO 算法
- **训练时**: 使用 `rewards/general_reward.py` 计算真实 reward
- **验证时**: 使用 `rewards/general_reward.py` 计算准确率
- **分析**: WizardRewardManager 提供丰富的思维词汇和重复率分析

### Intuitor 算法
- **训练时**: 使用自信度 (self-certainty)，外部 reward 设为 0
- **验证时**: 自动使用 `rewards/general_reward.py` 计算准确率
- **分析**: WizardRewardManager 提供相同的丰富分析

## 支持的数据源

`rewards/general_reward.py` 支持以下数据源：

### 代码任务
- `code`: 通用代码题
- `livecodebench`: LiveCodeBench 实时代码基准
- `livecodebench_v5`: LiveCodeBench v5
- `livecodebench_v6`: LiveCodeBench v6  
- `humanevalplus`: HumanEval+ 扩展版

### 数学任务
- `math`: 通用数学题
- `aime2024`: AIME 2024 竞赛题
- `aime2025`: AIME 2025 竞赛题
- `math500`: Math500 题集

## 使用示例

### 1. GRPO 实验
```bash
bash verl/examples/grpo_with_rewards.sh
```

### 2. Intuitor 实验  
```bash
bash verl/examples/intuitor_with_rewards.sh
```

### 3. 对比实验
```bash
bash verl/examples/compare_intuitor_grpo.sh
```

## 分析功能详解

### 思维词汇分析 (Thinking Tokens Analysis)
WizardRewardManager 分析 7 大类思维词汇：
- **思考与推理**: analyze, reasoning, think, logic 等
- **计划与策略**: plan, strategy, method, approach 等  
- **评估与验证**: evaluate, assess, validate, verify 等
- **决策与问题解决**: decide, resolve, problem, solution 等
- **反思与回顾**: reflect, contemplate, review, ponder 等
- **概念与理论**: concept, theory, hypothesis, model 等
- **逻辑连接**: because, therefore, however, perhaps 等

### 重复率检测 (Repetition Analysis)
- 使用 20-gram 滑动窗口检测重复模式
- 计算每个序列的重复率
- 记录最高频次的 n-gram
- 分析批次级别的重复模式

### 响应长度分析 (Response Length Analysis)
- 统计有效响应长度
- 提供平均值、最大值、最小值
- 支持超长响应惩罚机制

## 数据流

```
配置: reward_manager=wizard + use_general_reward=True
    ↓
load_reward_manager() 检测配置
    ↓
创建 WizardRewardManager(compute_score=general_reward_fn)
    ↓
训练/验证时调用
    ↓
WizardRewardManager.__call__()
    ├── 调用 general_reward_fn 计算正确性
    ├── 分析思维词汇使用情况  
    ├── 检测重复率模式
    └── 统计响应长度信息
    ↓
返回丰富的分析结果
```

## 优势

1. **统一评估**: Intuitor 和 GRPO 使用相同的验证函数
2. **丰富分析**: 获得思维词汇、重复率等深度分析
3. **代码复用**: 直接使用 DAPO 的 reward 实现
4. **灵活配置**: 可以轻松切换不同的分析级别
5. **向后兼容**: 不影响现有的 VERL 配置

## 注意事项

1. **WizardRewardManager**: 需要确保 `verl/workers/reward_manager/wizard.py` 可用
2. **数据格式**: 确保数据中包含正确的 `data_source` 字段
3. **性能**: WizardRewardManager 会增加一些计算开销，但提供更丰富的分析
4. **内存**: 思维词汇和重复率分析会使用额外内存

## 故障排除

### 导入错误
```python
ImportError: No module named 'rewards.general_reward'
```
**解决**: 确保 `rewards/` 目录在 Python 路径中

### WizardRewardManager 不可用
```python
ValueError: Unknown reward manager: wizard
```
**解决**: 确保 `verl/workers/reward_manager/wizard.py` 存在并正确注册

### 数据源不支持
```python
ValueError: Current supports for data_source are [...] -- No idea what's: your_data_source
```
**解决**: 检查数据中的 `data_source` 字段，确保使用支持的数据源名称
