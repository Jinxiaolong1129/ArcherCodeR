# 修复 token_entropy 和 prob_disparity 验证准确率为0的问题

## 问题描述

`token_entropy` 和 `prob_disparity` 两种方法的验证准确率 `val-core/livecodebench/acc/` 一直是0。

## 根本原因

这两种方法使用**纯内在奖励(intrinsic reward)**进行训练：
- `token_entropy`: 使用负平均熵作为训练信号
- `prob_disparity`: 使用top-1和top-2概率差作为训练信号

但在 `verl/trainer/ppo/reward.py` 的 `load_reward_manager` 函数中，只有以下算法在验证时会使用真实的奖励函数：
- `intuitor`
- `intuitor_selective`
- `intuitor_entropy`

而 `token_entropy` 和 `prob_disparity` 不在这个列表中，导致它们在验证时使用了 `default_compute_score`。

由于数据的 `data_source` 字段是 `"code"` 或 `"intuitor"`，`default_compute_score` 会返回固定值 0.0（见 `verl/utils/reward_score/__init__.py` 第86-89行）：

```python
elif data_source == "code" or data_source == "intuitor" or data_source == "dummy":
    # Dummy reward function for intuitor algorithm which uses self-certainty
    # Return a fixed score to maintain compatibility with the training pipeline
    res = 0.0
```

这导致验证时所有样本的奖励都是0，因此准确率也是0。

## 解决方案

修改 `verl/trainer/ppo/reward.py` 的第98-127行，将 `token_entropy` 和 `prob_disparity` 加入到内在奖励算法列表中：

### 修改前
```python
# Special handling for Intuitor algorithm
if hasattr(config, 'algorithm') and hasattr(config.algorithm, 'adv_estimator') and config.algorithm.adv_estimator in ["intuitor", "intuitor_selective", "intuitor_entropy"]:
    algo_name = config.algorithm.adv_estimator
    ...
```

### 修改后
```python
# Special handling for Intuitor and intrinsic reward algorithms
# These algorithms use intrinsic rewards (self-certainty, entropy, etc.) for training
# but need actual reward functions for validation
intrinsic_reward_algos = [
    "intuitor", 
    "intuitor_selective", 
    "intuitor_entropy",
    "token_entropy",
    "prob_disparity",
    "trajectory_entropy"
]

if hasattr(config, 'algorithm') and hasattr(config.algorithm, 'adv_estimator') and config.algorithm.adv_estimator in intrinsic_reward_algos:
    algo_name = config.algorithm.adv_estimator
    ...
```

## 关键改进

1. **训练阶段**：`token_entropy` 和 `prob_disparity` 使用 dummy reward（返回0.0），因为它们依赖内在奖励信号
2. **验证阶段**：这些算法现在会使用 `general_reward_fn` 来评估代码的真实正确性

## 验证修复

重新运行训练脚本后，验证阶段应该能够正确计算代码准确率：

```bash
# Token Entropy
bash scripts-iclr-server/token_entropy/token_entropy_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh

# Probability Disparity
bash scripts-iclr-server/prob_disparity/prob_disparity_Qwen-1.5B-2k-8k-8k-batch64-no-kl-simple_ray.sh
```

## 预期结果

修复后，你应该能看到：
- 训练日志中：`📊 Final reward stats: avg=0.000` (训练时仍然使用dummy reward，这是正确的)
- 验证日志中：`val-core/livecodebench/acc/` 应该显示真实的准确率（非0值）

## 技术细节

### 为什么训练时使用 dummy reward？

对于内在奖励算法（如 `token_entropy`、`prob_disparity`），训练时的优势估计完全基于内在信号（熵、概率差等），不需要外部奖励。使用 dummy reward 可以：
1. 避免不必要的代码执行开销
2. 保持训练流程的一致性
3. 确保优势计算只依赖内在信号

### 为什么验证时使用真实 reward？

验证的目的是评估模型生成代码的真实质量，因此必须使用真实的奖励函数（代码执行结果）来计算准确率。

## 相关文件

- `verl/trainer/ppo/reward.py` - 奖励管理器加载逻辑（已修复）
- `verl/utils/reward_score/__init__.py` - 默认奖励计算函数
- `rewards/general_reward.py` - 通用奖励函数（支持代码和数学任务）
- `verl/workers/reward_manager/wizard.py` - Wizard奖励管理器

## 日期

2025-01-XX

