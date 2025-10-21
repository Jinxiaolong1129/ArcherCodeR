# DACE Implementation Summary

## 📦 实现概述

DACE (Difficulty-Aware Certainty Exploration) 已成功集成到代码库中。

---

## 📁 修改的文件

### 1. `verl/trainer/ppo/core_algos.py`

**修改内容**:
- ✅ 在 `AdvantageEstimator` 枚举中添加 `DACE = "dace"` (第92行)
- ✅ 实现 `compute_dace_advantage()` 函数 (第311-445行)

**核心功能**:
```python
@register_adv_est(AdvantageEstimator.DACE)
def compute_dace_advantage(
    token_level_rewards: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    alpha_scale: float = 0.1,
    beta_threshold: float = 0.5,
    ...
)
```

**Difficulty计算位置** (第392-397行):
```python
for prompt_id in id2rewards:
    rewards = torch.stack(id2rewards[prompt_id])
    success_rate = torch.mean(rewards)
    difficulty = 1.0 - success_rate  # 💡 核心公式
    id2difficulty[prompt_id] = difficulty
```

---

### 2. `verl/trainer/ppo/ray_trainer.py`

**修改内容**:
- ✅ 在 `compute_advantage()` 中添加 DACE 分支处理 (第472-522行)
- ✅ 在 `_validate_config()` 中将 DACE 添加到不使用 critic 的算法列表 (第609行)

**核心功能**:
```python
elif adv_estimator == AdvantageEstimator.DACE:
    # 获取DACE超参数
    alpha_scale = config.get("dace_alpha_scale", 0.1)
    beta_threshold = config.get("dace_beta_threshold", 0.5)
    
    # 计算DACE优势
    advantages, returns = core_algos.compute_dace_advantage(
        token_level_rewards=token_level_rewards,
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        index=data.non_tensor_batch["uid"],
        alpha_scale=alpha_scale,
        beta_threshold=beta_threshold,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
```

---

## 🚀 训练脚本

### 1. `scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh`

**完整的ArcherCodeR训练脚本**，包含：
- 所有ArcherCodeR的配置（序列长度、批次大小等）
- DACE特定的超参数配置
- 详细的配置说明和训练日志

**使用方法**:
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

**关键配置**:
```bash
adv_estimator=dace
dace_alpha_scale=0.1
dace_beta_threshold=0.5
n_resp_per_prompt=16  # 重要：用于difficulty估计
```

---

### 2. `scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-gsm8k-math.sh`

**通用的GSM8K/MATH训练脚本**（基于verl标准配置）

**使用方法**:
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-gsm8k-math.sh
```

---

## 📚 文档

### 1. `docs/DACE_ALGORITHM_EXPLANATION.md` (详细英文版)

包含：
- ✅ 算法的5个关键步骤详解
- ✅ 每个步骤的公式和代码
- ✅ 具体的数值例子
- ✅ Difficulty计算的完整流程
- ✅ 超参数调优指南
- ✅ 与其他方法的对比

### 2. `docs/DACE_简明教程.md` (简明中文版)

包含：
- ✅ 核心思想一句话总结
- ✅ Difficulty计算位置和逻辑
- ✅ 完整数据流图
- ✅ 具体例子（简单题vs困难题）
- ✅ 使用方法和调参建议
- ✅ 快速检查清单

### 3. `scripts-iclr-server/train/README_DACE.md`

包含：
- ✅ 问题动机
- ✅ 方法描述
- ✅ 实现细节
- ✅ 超参数说明
- ✅ 故障排除指南

---

## 🎯 快速开始

### 最简使用方式

```bash
# 1. 确保环境变量已设置（在.env文件中）
# 2. 直接运行训练脚本
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

### 自定义配置

```bash
python3 -m dapo.main_dapo \
    algorithm.adv_estimator=dace \
    algorithm.dace_alpha_scale=0.1 \
    algorithm.dace_beta_threshold=0.5 \
    actor_rollout_ref.rollout.n=16 \
    # ... 其他配置
```

---

## 📊 关键概念

### Difficulty (任务难度)

**计算公式**: `difficulty = 1 - success_rate`

**计算位置**: `verl/trainer/ppo/core_algos.py` 第392-397行

**含义**:
- `difficulty = 0.0`: 所有响应都正确 → **简单任务**
- `difficulty = 0.5`: 一半正确 → **中等任务**
- `difficulty = 1.0`: 所有响应都错误 → **困难任务**

### Certainty (确定性)

**计算公式**: `certainty = -mean(log_prob)`

**含义**:
- **低certainty值**: 高确定性（模型用高概率token）
- **高certainty值**: 低确定性（模型在探索低概率token）

### Adaptive Coefficient (自适应系数)

**计算公式**: `α = α_scale × sign(β_threshold - difficulty)`

**行为**:
- **简单任务**: `α > 0` → 鼓励高确定性（exploitation）
- **困难任务**: `α < 0` → 鼓励低确定性（exploration）

---

## ⚙️ 超参数说明

### dace_alpha_scale

**推荐值**: 0.1 - 0.3

**作用**: 控制内在奖励的强度

| 值 | 效果 |
|----|------|
| 0.05 | 内在奖励影响较小，接近纯GRPO |
| 0.1 | 平衡（推荐起始值） |
| 0.2-0.3 | 较强的探索/利用信号 |
| >0.5 | 内在奖励可能压倒外部奖励（不推荐） |

### dace_beta_threshold

**推荐值**: 0.4 - 0.6

**作用**: 区分"简单任务"和"困难任务"的阈值

| 值 | 效果 |
|----|------|
| 0.3-0.4 | 更多任务被视为"简单" → 更多exploitation |
| 0.5 | 平衡（推荐） |
| 0.6-0.7 | 更多任务被视为"困难" → 更多exploration |

### n_resp_per_prompt

**推荐值**: 8 - 16

**作用**: 每个prompt生成的响应数量，用于估计difficulty

| 值 | 效果 |
|----|------|
| 4 | 统计不够可靠 |
| 8 | 基本够用 |
| 16 | 推荐值，较好的统计估计 |
| 32 | 更准确，但计算开销大 |

---

## 🔍 训练日志示例

当DACE正常工作时，你会看到：

```
-------------------------------- This is DACE --------------------------------
DACE Hyperparameters:
  α_scale (intrinsic reward scaling): 0.1
  β_threshold (difficulty threshold): 0.5
data.batch['token_level_rewards'].shape: torch.Size([512, 1024])
data.batch['old_log_probs'].shape: torch.Size([512, 1024])
External reward range: [0.0000, 1.0000]
Certainty range: [0.5234, 3.8471]
Advantage range: [2.1543, -1.8765]
-------------------------------- End of DACE --------------------------------
```

**检查要点**:
- ✅ External reward应该在[0.0, 1.0]范围（二元奖励）
- ✅ Certainty应该有一定范围（说明有不同确定性的响应）
- ✅ Advantage应该有正有负（说明奖励分配合理）

---

## 🐛 故障排除

### 问题1: 训练不稳定

**症状**: Loss震荡剧烈，指标不收敛

**解决方案**:
```bash
# 减小alpha_scale
algorithm.dace_alpha_scale=0.05

# 确保启用advantage normalization
algorithm.norm_adv_by_std_in_grpo=True
```

### 问题2: 模型不探索

**症状**: 困难任务上卡住，不尝试新方法

**解决方案**:
```bash
# 增大alpha_scale
algorithm.dace_alpha_scale=0.2

# 提高threshold（更多任务被视为困难）
algorithm.dace_beta_threshold=0.6

# 增加响应数（更准确的difficulty估计）
actor_rollout_ref.rollout.n=16
```

### 问题3: Difficulty估计不准

**症状**: 所有任务都被判定为同一难度

**解决方案**:
```bash
# 增加每个prompt的响应数
actor_rollout_ref.rollout.n=16  # 或更多

# 检查数据集是否有难度混合
# 如果数据集全是简单题或全是难题，difficulty会缺乏区分度
```

---

## 📈 预期效果

使用DACE后，应该观察到：

1. **简单任务**: 
   - 快速收敛到高准确率
   - 响应的确定性逐渐提高
   - 不会浪费时间探索不必要的路径

2. **困难任务**:
   - 持续探索不同的推理方法
   - 响应的多样性保持较高
   - 逐步提高成功率

3. **整体训练**:
   - 比纯GRPO更快收敛（简单任务上）
   - 比纯GRPO更好的最终性能（困难任务上）
   - 训练过程更稳定

---

## ✅ 验证清单

在运行训练前，确认：

- [ ] **代码集成**: `AdvantageEstimator.DACE` 存在
- [ ] **配置正确**: `algorithm.adv_estimator=dace`
- [ ] **超参数设置**: `dace_alpha_scale` 和 `dace_beta_threshold`
- [ ] **响应数足够**: `n_resp_per_prompt >= 8`
- [ ] **数据集准备**: 包含不同难度的任务
- [ ] **环境变量**: WANDB_API_KEY 等已设置

---

## 🎓 总结

DACE通过以下方式改进RL训练：

1. ✅ **自动识别**任务难度（通过多次采样统计）
2. ✅ **自适应调整**探索/利用策略
3. ✅ **密集反馈**（利用log_prob提供连续信号）
4. ✅ **任务感知**（不同难度用不同策略）

**核心创新**: Difficulty计算
```python
# 在 verl/trainer/ppo/core_algos.py 第392-397行
success_rate = torch.mean(rewards)
difficulty = 1.0 - success_rate
```

这使得模型能够：
- 在**简单任务**上快速高效地收敛
- 在**困难任务**上持续探索创新
- **自动平衡**，无需手动调整

---

## 📞 需要帮助？

1. 查看详细文档: `docs/DACE_ALGORITHM_EXPLANATION.md`
2. 查看中文教程: `docs/DACE_简明教程.md`
3. 查看配置说明: `scripts-iclr-server/train/README_DACE.md`
4. 检查训练日志中的DACE输出部分

