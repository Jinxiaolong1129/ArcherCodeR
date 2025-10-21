# DACE Metrics Tracking Summary

## 概述

为DACE算法添加了全面的指标跟踪系统，类似于INTUITOR的certainty指标，可以通过WandB等工具监控训练过程。

---

## 📁 修改的文件

### 1. `verl/trainer/ppo/ray_trainer.py`

#### 添加的数据收集 (第509-581行)

在`compute_advantage()`函数的DACE分支中，添加了详细的统计数据收集：

```python
# 存储到 batch.non_tensor_batch 中的数据：
data.non_tensor_batch["dace_external_rewards"]      # 外部奖励（正确性）
data.non_tensor_batch["dace_certainty"]             # 确定性度量 C(y,x) = -mean(log_prob)
data.non_tensor_batch["dace_difficulty"]            # 任务难度 diff(x) = 1 - success_rate
data.non_tensor_batch["dace_alpha"]                 # 自适应系数 α = α_scale × sign(β - diff)
data.non_tensor_batch["dace_intrinsic_rewards"]    # 内在奖励 R_int = α × C
data.non_tensor_batch["dace_total_rewards"]        # 总奖励 R_total = R_ext + R_int
data.non_tensor_batch["dace_hard_task_mask"]       # Hard任务的mask (diff > β)
data.non_tensor_batch["dace_easy_task_mask"]       # Easy任务的mask (diff ≤ β)
```

#### 增强的输出信息 (第591-636行)

添加了详细的终端输出，包括：
- 🎯 **任务分布**: Hard/Easy任务比例
- 📊 **难度&Alpha**: 难度和自适应系数统计
- 💰 **奖励组成**: 外部/内在/总奖励的详细分析
- 🔍 **确定性分析**: 整体和分组的确定性统计
- 📈 **优势函数**: 最终的advantage分布

---

### 2. `verl/trainer/ppo/metric_utils.py`

#### 添加的指标 (第218-269行)

在`compute_data_metrics()`函数中添加了48个DACE相关指标：

#### 📊 **基础统计指标** (8个)

**Certainty指标** - 模型对自己预测的确定程度
```python
"dace/certainty/mean"          # 平均确定性
"dace/certainty/max"           # 最大确定性
"dace/certainty/min"           # 最小确定性
"dace/certainty/std"           # 确定性标准差
```

**Difficulty指标** - 任务难度评估
```python
"dace/difficulty/mean"         # 平均难度 (0=简单, 1=困难)
"dace/difficulty/max"          # 最大难度
"dace/difficulty/min"          # 最小难度
"dace/difficulty/std"          # 难度标准差
```

#### 🎛️ **自适应系数指标** (4个)

```python
"dace/alpha/mean"              # 平均α (负=探索, 正=利用)
"dace/alpha/max"               # 最大α
"dace/alpha/min"               # 最小α
"dace/alpha/std"               # α标准差
```

#### 💰 **奖励组成指标** (12个)

**外部奖励** (正确性)
```python
"dace/external_reward/mean"    # 平均外部奖励
"dace/external_reward/max"     # 最大外部奖励
"dace/external_reward/min"     # 最小外部奖励
"dace/external_reward/std"     # 外部奖励标准差
```

**内在奖励** (α × certainty)
```python
"dace/intrinsic_reward/mean"   # 平均内在奖励
"dace/intrinsic_reward/max"    # 最大内在奖励
"dace/intrinsic_reward/min"    # 最小内在奖励
"dace/intrinsic_reward/std"    # 内在奖励标准差
```

**总奖励** (外部 + 内在)
```python
"dace/total_reward/mean"       # 平均总奖励
"dace/total_reward/max"        # 最大总奖励
"dace/total_reward/min"        # 最小总奖励
"dace/total_reward/std"        # 总奖励标准差
```

#### 🎯 **任务分布指标** (4个)

```python
"dace/hard_task_count"         # Hard任务数量 (diff > β)
"dace/easy_task_count"         # Easy任务数量 (diff ≤ β)
"dace/hard_task_ratio"         # Hard任务比例
"dace/easy_task_ratio"         # Easy任务比例
```

#### 🔍 **Hard任务专属指标** (4个)

```python
"dace/hard_task_certainty_mean"         # Hard任务的平均确定性
"dace/hard_task_difficulty_mean"        # Hard任务的平均难度
"dace/hard_task_intrinsic_reward_mean"  # Hard任务的平均内在奖励
"dace/hard_task_external_reward_mean"   # Hard任务的平均外部奖励
```

#### 🎓 **Easy任务专属指标** (4个)

```python
"dace/easy_task_certainty_mean"         # Easy任务的平均确定性
"dace/easy_task_difficulty_mean"        # Easy任务的平均难度
"dace/easy_task_intrinsic_reward_mean"  # Easy任务的平均内在奖励
"dace/easy_task_external_reward_mean"   # Easy任务的平均外部奖励
```

#### 📈 **分析指标** (1个)

```python
"dace/intrinsic_reward_contribution"    # 内在奖励的贡献比例
                                         # = |mean(R_int)| / |mean(R_total)|
```

---

## 🎯 关键指标解读

### 1️⃣ **监控训练健康度**

```python
# 在WandB中查看这些指标的变化
dace/difficulty/mean              # 应该在训练中逐渐下降（任务变简单）
dace/external_reward/mean         # 应该在训练中逐渐上升（正确率提高）
dace/hard_task_ratio              # 应该在训练中逐渐下降（困难任务减少）
```

### 2️⃣ **验证DACE机制**

```python
# 确认自适应机制是否工作
dace/hard_task_ratio              # 如果 > 50%，大部分任务是hard
dace/alpha/mean                   # 如果 < 0，倾向于探索；如果 > 0，倾向于利用
dace/intrinsic_reward_contribution # 内在奖励的影响程度（应该在5-20%）
```

### 3️⃣ **Hard vs Easy 任务对比**

```python
# Hard任务应该：
dace/hard_task_certainty_mean     # 较低（模型不确定）
dace/hard_task_difficulty_mean    # 较高（> β_threshold）
dace/hard_task_intrinsic_reward_mean  # 负值（鼓励探索低确定性）

# Easy任务应该：
dace/easy_task_certainty_mean     # 较高（模型确定）
dace/easy_task_difficulty_mean    # 较低（< β_threshold）
dace/easy_task_intrinsic_reward_mean  # 正值（鼓励利用高确定性）
```

### 4️⃣ **确定性变化**

```python
# 随训练进行，期望看到：
dace/certainty/mean               # 整体上升（模型更自信）
dace/hard_task_certainty_mean     # 探索阶段可能保持低位或波动
dace/easy_task_certainty_mean     # 利用阶段应该持续上升
```

---

## 📊 WandB监控建议

### 创建自定义Dashboard

#### Panel 1: 任务分布变化
```
Plot: Line chart
Metrics:
  - dace/hard_task_ratio
  - dace/easy_task_ratio
Title: "Task Distribution Over Time"
```

#### Panel 2: 奖励组成分析
```
Plot: Multi-line chart
Metrics:
  - dace/external_reward/mean
  - dace/intrinsic_reward/mean
  - dace/total_reward/mean
Title: "Reward Composition"
```

#### Panel 3: 确定性对比
```
Plot: Multi-line chart
Metrics:
  - dace/certainty/mean
  - dace/hard_task_certainty_mean
  - dace/easy_task_certainty_mean
Title: "Certainty: Overall vs Hard/Easy Tasks"
```

#### Panel 4: 难度&Alpha动态
```
Plot: Dual-axis line chart
Metrics (Left Y):
  - dace/difficulty/mean
Metrics (Right Y):
  - dace/alpha/mean
Title: "Difficulty and Adaptive Coefficient"
```

#### Panel 5: 内在奖励贡献
```
Plot: Line chart
Metrics:
  - dace/intrinsic_reward_contribution
Title: "Intrinsic Reward Contribution (%)"
```

---

## 🔧 调试技巧

### 问题1: 内在奖励影响太小
**症状**: `dace/intrinsic_reward_contribution < 0.05`

**解决方案**:
- 增加 `dace_alpha_scale` (从0.05到0.1或0.15)
- 检查 `dace/certainty/mean` 是否在合理范围（通常2-5）

### 问题2: 所有任务都是Hard
**症状**: `dace/hard_task_ratio > 0.8`

**解决方案**:
- 降低 `dace_beta_threshold` (从0.4到0.3或0.2)
- 检查 `dace/difficulty/mean` 的值
- 可能模型当前确实表现不好，需要更多训练

### 问题3: 所有任务都是Easy
**症状**: `dace/easy_task_ratio > 0.8`

**解决方案**:
- 提高 `dace_beta_threshold` (从0.4到0.5或0.6)
- 检查 `dace/external_reward/mean` 是否已经很高
- 可能模型已经很好，可以考虑更难的数据集

### 问题4: Alpha没有自适应
**症状**: `dace/alpha/std` 接近0

**解决方案**:
- 检查数据集是否有足够的难度多样性
- 增加 `n_resp_per_prompt` 以获得更好的难度估计
- 检查 `dace/difficulty/std` 是否足够大

---

## 🔍 日志查询命令

### 在训练日志中快速查找DACE统计

```bash
# 查看任务分布
grep "Hard tasks" your_training.log

# 查看奖励组成
grep "Reward Components" -A 10 your_training.log

# 查看确定性分析
grep "Certainty Analysis" -A 5 your_training.log

# 查看完整的DACE统计
grep "Final Statistics Summary" -A 30 your_training.log
```

### 提取关键数值到CSV

```bash
# 提取hard task ratio
grep "Hard tasks" your_training.log | \
  sed 's/.*Hard tasks.*: \([0-9]*\)\/\([0-9]*\).*/\1,\2/' > hard_task_stats.csv

# 提取intrinsic reward contribution
grep "Contribution:" your_training.log | \
  sed 's/.*Contribution: \([0-9.]*\)%.*/\1/' > intrinsic_contribution.csv
```

---

## 📈 预期的训练曲线

### 初期 (Steps 1-50)
```
dace/difficulty/mean:              0.6-0.8 (大部分任务困难)
dace/hard_task_ratio:              0.6-0.8 (大部分是hard任务)
dace/alpha/mean:                   -0.03 to -0.01 (倾向探索)
dace/external_reward/mean:         0.2-0.3 (正确率低)
dace/intrinsic_reward_contribution: 10-20% (探索阶段影响较大)
```

### 中期 (Steps 50-200)
```
dace/difficulty/mean:              0.4-0.6 (难度下降)
dace/hard_task_ratio:              0.4-0.6 (hard/easy平衡)
dace/alpha/mean:                   -0.01 to 0.01 (探索-利用平衡)
dace/external_reward/mean:         0.4-0.6 (正确率提升)
dace/intrinsic_reward_contribution: 5-15% (影响逐渐减小)
```

### 后期 (Steps 200+)
```
dace/difficulty/mean:              0.2-0.4 (大部分任务简单)
dace/hard_task_ratio:              0.2-0.4 (大部分是easy任务)
dace/alpha/mean:                   0.01 to 0.03 (倾向利用)
dace/external_reward/mean:         0.6-0.8 (正确率高)
dace/intrinsic_reward_contribution: 5-10% (影响较小但仍有用)
```

---

## ✅ 修改清单

- ✅ `verl/trainer/ppo/ray_trainer.py`: 添加DACE数据收集和详细输出
- ✅ `verl/trainer/ppo/metric_utils.py`: 添加48个DACE指标到metrics系统
- ✅ 所有指标自动记录到WandB/Console logger
- ✅ 类似INTUITOR的certainty指标结构
- ✅ Hard/Easy任务分组分析
- ✅ 奖励组成分解和贡献分析

---

## 🚀 开始使用

现在可以运行DACE训练，所有指标将自动记录：

```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

指标将显示在：
1. **终端输出**: 每个step的详细统计
2. **WandB**: 在项目 `ArcherCodeR-DACE` 下查看所有曲线
3. **日志文件**: `./output/ArcherCodeR-DACE/*/..._dace.log`

所有`dace/*`开头的指标都会自动在WandB中显示，方便监控和对比实验！

