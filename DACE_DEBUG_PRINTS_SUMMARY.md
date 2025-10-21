# DACE Debug Print Statements Summary

## 添加的调试信息概览

为了方便调试和了解DACE训练进度，我在代码中添加了详细的print语句。这些print语句会在运行时输出关键信息。

---

## 📁 文件1: `verl/trainer/ppo/core_algos.py`

### 函数: `compute_dace_advantage()` (第311-522行)

#### 添加的调试输出：

### 🚀 启动信息
```
='*80
🚀 DACE ADVANTAGE COMPUTATION STARTED
='*80
📊 DACE Hyperparameters:
  ├─ α_scale (intrinsic reward scaling): 0.05
  ├─ β_threshold (difficulty threshold): 0.4
  ├─ norm_adv_by_std_in_grpo: True
  └─ epsilon: 1e-6
📦 Batch info:
  ├─ Batch size: 1024
  └─ Response length: 8192
```

### 📝 Step 1: 外部奖励计算
```
📝 Step 1: Computing external rewards...
  ├─ External reward range: [0.0000, 1.0000]
  ├─ External reward mean: 0.3450
  └─ Correct samples: 350/1024
```
- 显示外部奖励（正确性）的范围、均值和正确样本数

### 📝 Step 2: 确定性度量计算
```
📝 Step 2: Computing certainty metric...
  ├─ Certainty range: [2.1234, 3.5678]
  ├─ Certainty mean: 2.8456
  └─ Lower certainty = more uncertain, Higher certainty = more confident
```
- Certainty = -mean(log_prob)
- 较低的值表示模型更不确定，较高的值表示模型更确定

### 📝 Step 3: 每个prompt的难度估计
```
📝 Step 3: Estimating difficulty per prompt...
  ├─ Number of unique prompts: 64
  └─ Responses per prompt: [16, 16, 16]... (showing first 3)
```
- 显示有多少个唯一的prompt和每个prompt的响应数量

### 📝 Step 4: 难度和自适应系数计算
```
📝 Step 4: Computing difficulty and adaptive coefficients...
  ├─ Difficulty range: [0.1000, 0.9000]
  ├─ Difficulty mean: 0.5500
  ├─ β_threshold: 0.4
  ├─ Hard tasks (diff > β): 35/64 (54.7%)
  ├─ Easy tasks (diff < β): 29/64 (45.3%)
  ├─ Alpha range: [-0.0500, 0.0500]
  └─ Negative α → explore (low certainty), Positive α → exploit (high certainty)
```
**关键信息解读：**
- **Difficulty**: 1 - success_rate，范围[0,1]
  - 接近0 = 简单任务（成功率高）
  - 接近1 = 困难任务（成功率低）
- **Hard tasks**: difficulty > β_threshold → α为负 → 鼓励探索（低确定性）
- **Easy tasks**: difficulty < β_threshold → α为正 → 鼓励利用（高确定性）

### 📝 Step 5: 内在奖励计算
```
📝 Step 5: Computing intrinsic rewards...
  ├─ Intrinsic reward range: [-0.1500, 0.1200]
  ├─ Intrinsic reward mean: -0.0200
  └─ Intrinsic reward std: 0.0800
```
- R_int = α(x) * C(y,x)
- 负值表示鼓励探索，正值表示鼓励利用

### 📝 Step 6: 组合奖励
```
📝 Step 6: Combining external and intrinsic rewards...
  ├─ Total reward range: [-0.1000, 1.1000]
  ├─ Total reward mean: 0.3250
  └─ Total reward std: 0.4500
```
- R_total = R_external + R_intrinsic

### 📝 Step 7: 优势函数计算
```
📝 Step 7: Computing GRPO-style advantages...
  ├─ Advantage range: [-2.1000, 2.3000]
  ├─ Advantage mean: 0.0000
  ├─ Advantage std: 1.0000
  ├─ Positive advantages: 512/1024
  └─ Negative advantages: 512/1024
```

### ✅ 完成信息
```
-'*80
✅ DACE ADVANTAGE COMPUTATION COMPLETED
='*80
```

---

## 📁 文件2: `verl/trainer/ppo/ray_trainer.py`

### 函数: `compute_advantage()` - DACE分支 (第472-543行)

#### 添加的调试输出：

### 🚀 Ray Trainer中的DACE启动信息
```
='*80
🚀 DACE ADVANTAGE ESTIMATION - RAY TRAINER
='*80
📊 Input Data:
  ├─ token_level_rewards.shape: (1024, 8192)
  ├─ old_log_probs.shape: (1024, 8192)
  ├─ response_mask.shape: (1024, 8192)
  └─ Number of unique prompts: 64
📊 DACE Hyperparameters:
  ├─ α_scale (intrinsic reward scaling): 0.05
  ├─ β_threshold (difficulty threshold): 0.4
  └─ norm_adv_by_std_in_grpo: True
-'*80
```

### 📊 最终统计信息
```
📊 Computing statistics for logging...
-'*80
📈 Final Statistics:
  ├─ External reward range: [0.0000, 1.0000]
  ├─ External reward mean: 0.3450
  ├─ Certainty range: [2.1234, 3.5678]
  ├─ Certainty mean: 2.8456
  ├─ Advantage (per sample) range: [-2.1000, 2.3000]
  ├─ Advantage (per sample) mean: 0.0000
  ├─ Positive advantage samples: 512/1024
  └─ Negative advantage samples: 512/1024
-'*80
✅ DACE ADVANTAGE ESTIMATION COMPLETED
='*80
```

---

## 🎯 如何使用这些Debug信息

### 1. 监控训练健康状态
- **External rewards**: 应该看到正确样本的比例随训练增加
- **Certainty**: 观察模型确定性的变化
- **Difficulty distribution**: 了解任务难度分布

### 2. 验证DACE机制
- **Hard vs Easy tasks分布**: 应该动态变化
- **Alpha values**: 确认自适应系数的正负符号是否符合预期
- **Intrinsic rewards**: 观察内在奖励的贡献

### 3. 诊断问题
- 如果所有任务都是hard → 可能需要调整β_threshold
- 如果intrinsic rewards太小 → 可能需要增加α_scale
- 如果advantages分布异常 → 检查reward normalization

### 4. 日志文件位置
训练日志会保存在：
```
./output/ArcherCodeR-DACE/DACE-Qwen2.5-1.5B-2K-8K-16resp/ArcherCodeR-DACE_DACE-Qwen2.5-1.5B-2K-8K-16resp_dace.log
```

---

## 📊 预期输出示例

在每个训练步骤中，你会看到：

1. **Ray Trainer启动DACE** (一次)
2. **Core Algos执行DACE计算** (包含7个详细步骤)
3. **Ray Trainer完成统计** (一次)

总共每个batch会输出约30-40行的详细信息，帮助你：
- ✅ 了解当前训练进度
- ✅ 监控DACE机制是否正常工作
- ✅ 调试任何潜在问题
- ✅ 验证超参数设置是否合理

---

## 🔧 调试技巧

### 如果训练很慢
可以通过查看log找到瓶颈：
```bash
grep "Step [0-9]:" your_log_file.log | tail -20
```

### 如果想只看DACE相关信息
```bash
grep -A 30 "DACE ADVANTAGE" your_log_file.log
```

### 如果想看难度分布变化
```bash
grep "Hard tasks" your_log_file.log
```

### 如果想看奖励统计
```bash
grep "External reward" your_log_file.log
grep "Intrinsic reward" your_log_file.log
```

---

## 💡 关键概念提醒

### DACE工作原理
1. **Difficulty**: `diff(x) = 1 - success_rate`
2. **Certainty**: `C(y,x) = -mean(log_prob)`  
3. **Adaptive α**: `α = α_scale × sign(β_threshold - diff)`
   - Hard task (diff > β): α < 0 → 鼓励探索（低certainty获得更高奖励）
   - Easy task (diff < β): α > 0 → 鼓励利用（高certainty获得更高奖励）
4. **Intrinsic Reward**: `R_int = α × C`
5. **Total Reward**: `R_total = R_external + R_int`

### 超参数调整建议
- **α_scale**: 默认0.05，如果内在奖励影响太小可增加到0.1
- **β_threshold**: 默认0.4，根据任务难度分布调整
  - 如果大部分任务都是hard，可以降低到0.3
  - 如果大部分任务都是easy，可以提高到0.5

---

## ✅ 修改完成

所有debug print语句已成功添加到：
- ✅ `verl/trainer/ppo/core_algos.py` - compute_dace_advantage()
- ✅ `verl/trainer/ppo/ray_trainer.py` - compute_advantage() DACE分支

现在可以运行训练脚本了：
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

