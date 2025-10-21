# DACE实验对比：v1 vs v2

## 📊 实验配置对比

| 参数 | DACE v1 (Baseline) | DACE v2 (Enhanced) | 变化说明 |
|------|-------------------|-------------------|---------|
| **α_scale** | 0.05 | 0.1 | ⬆️ 翻倍，更强的内在奖励信号 |
| **β_threshold** | 0.4 | 0.5 | ⬆️ 提高，更严格的hard任务定义 |
| **Experiment Name** | `DACE-Qwen2.5-1.5B-2K-8K-16resp` | `DACE-v2-Qwen2.5-1.5B-alpha0.1-beta0.5` | 不同的实验名称 |
| **Save Path** | `./output/ArcherCodeR-DACE/DACE-Qwen2.5-1.5B-2K-8K-16resp` | `./output/ArcherCodeR-DACE/DACE-v2-Qwen2.5-1.5B-alpha0.1-beta0.5` | 独立的保存路径 |

### 其他配置（保持一致）

| 配置项 | 值 |
|-------|-----|
| Model | DeepSeek-R1-Distill-Qwen-1.5B |
| Batch Size | 64 |
| Responses per prompt | 16 |
| Max prompt length | 2K |
| Max response length | 8K |
| Tensor parallel | 2 |
| Total epochs | 10 |

---

## 🎯 预期效果对比

### v1 (α=0.05, β=0.4) - 论文默认参数

```python
# 从第一个step的实际数据
Intrinsic reward contribution: ~8.5%
Hard tasks: ~77.3% (diff > 0.4)
Easy tasks: ~22.7% (diff ≤ 0.4)

特点：
✓ 保守的内在奖励影响
✓ 较宽松的hard任务定义（40%成功率即为easy）
✓ 适合初步探索DACE机制
```

### v2 (α=0.1, β=0.5) - 增强版本

```python
# 预期效果
Intrinsic reward contribution: ~15-20% (翻倍)
Hard tasks: 预计 >80% (diff > 0.5)
Easy tasks: 预计 <20% (diff ≤ 0.5)

特点：
✓ 更激进的内在奖励影响
✓ 更严格的hard任务定义（50%成功率才算easy）
✓ 更强的探索-利用对比
```

---

## 💡 参数变化的影响分析

### 📈 α_scale: 0.05 → 0.1 (翻倍)

#### 对Hard任务（α = -0.1）
```python
# v1: α = -0.05
R_int = -0.05 × certainty
例: certainty=1.0 → R_int=-0.05

# v2: α = -0.1 (翻倍!)
R_int = -0.1 × certainty
例: certainty=1.0 → R_int=-0.1

影响：
✓ 对不确定响应的惩罚加倍
✓ 更强烈地鼓励"有把握再说"
✓ 可能加快在困难任务上的学习
⚠️ 风险：过度惩罚可能导致过于保守
```

#### 对Easy任务（α = +0.1）
```python
# v1: α = +0.05
R_int = +0.05 × certainty

# v2: α = +0.1 (翻倍!)
R_int = +0.1 × certainty

影响：
✓ 对已掌握任务的利用信号加倍
✓ 更快地巩固已学到的模式
⚠️ 风险：可能过早陷入局部最优
```

### 📈 β_threshold: 0.4 → 0.5 (提高)

```python
# v1: β = 0.4
# 40%成功率 → easy (开始exploit)
# 60%失败率 → hard (继续explore)

# v2: β = 0.5
# 50%成功率 → easy (开始exploit)
# 50%失败率 → hard (继续explore)

影响：
✓ 需要更高的成功率才能进入exploitation模式
✓ 更多任务被归类为hard → 更长时间保持探索
✓ 避免过早exploitation

预期变化：
• v1: 77% hard → 23% easy (实际观察)
• v2: 预计 82% hard → 18% easy (更多hard)
```

---

## 📊 在WandB中如何对比

### 关键指标对比

```python
# 1. 内在奖励贡献
"dace/intrinsic_reward_contribution"
v1: ~8-10%
v2: 预期 15-20%

# 2. Hard/Easy任务分布
"dace/hard_task_ratio"
v1: ~77%
v2: 预期 >80%

# 3. 内在奖励统计
"dace/intrinsic_reward/mean"
v1: ~-0.03
v2: 预期 ~-0.06 (翻倍)

# 4. 最终性能
"val-core/*/acc/*"  # 验证集准确率
v1 vs v2: 哪个更高？
```

### WandB查询命令

```python
# 在WandB UI中，添加比较：
runs = [
    "DACE-Qwen2.5-1.5B-2K-8K-16resp",      # v1
    "DACE-v2-Qwen2.5-1.5B-alpha0.1-beta0.5" # v2
]

# 重点对比：
1. dace/intrinsic_reward_contribution
2. dace/hard_task_ratio
3. critic/rewards/mean (总奖励)
4. val-core/*/acc/* (最终性能)
```

---

## 🎯 实验假设

### H1: 更强的内在奖励（α=0.1）能加速学习

**预期**：v2在训练早期（前3个epoch）应该显示更快的提升

**验证指标**：
- `critic/rewards/mean` 上升速度
- `val-core/*/acc/*` 提升曲线

### H2: 更严格的阈值（β=0.5）能避免过早exploitation

**预期**：v2保持exploration更久，最终可能达到更高的性能上限

**验证指标**：
- `dace/hard_task_ratio` 下降速度（v2应该下降更慢）
- 最终的验证集性能

### H3: v2可能出现的问题

**潜在风险**：
1. 过度惩罚 → 模型变得过于保守
2. 训练不稳定 → loss曲线波动更大
3. 收敛变慢 → 需要更多epochs

**监控指标**：
- `actor/pg_loss` 稳定性
- `actor/pg_clipfrac` 是否过高
- `response_length/mean` 是否异常变短

---

## 🚀 运行实验

### v1 (已运行)
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

**路径**: `./output/ArcherCodeR-DACE/DACE-Qwen2.5-1.5B-2K-8K-16resp`

### v2 (新实验)
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer-v2.sh
```

**路径**: `./output/ArcherCodeR-DACE/DACE-v2-Qwen2.5-1.5B-alpha0.1-beta0.5`

---

## 📝 实验记录模板

### Step 1观察

| Metric | v1 (α=0.05, β=0.4) | v2 (α=0.1, β=0.5) | 差异 |
|--------|-------------------|------------------|------|
| Correct rate | 37.5% (264/704) | ? | ? |
| Difficulty mean | 0.625 | ? | ? |
| Hard task % | 77.3% | ? | ? |
| Intrinsic reward mean | -0.0295 | ? | ? |
| Intrinsic contribution | 8.5% | ? | ? |
| Certainty mean | 0.9551 | ? | ? |

### 最终性能（Epoch 10）

| Metric | v1 | v2 | Winner |
|--------|----|----|--------|
| Validation accuracy | ? | ? | ? |
| Training time/step | ? | ? | ? |
| Final difficulty | ? | ? | ? |
| Convergence stability | ? | ? | ? |

---

## 💡 结论（待填写）

训练完成后填写：

### 哪个配置更好？
- [ ] v1 (α=0.05, β=0.4) - 更稳定/更保守
- [ ] v2 (α=0.1, β=0.5) - 更激进/更高性能
- [ ] 不确定 - 需要更多实验

### 主要发现

1. 

2. 

3. 

### 未来实验建议

1. 

2. 

3. 

---

## 📚 参考

- DACE原始论文超参数：α=0.05, β=0.4
- 当前实验：探索α和β对性能的影响
- 目标：找到最优的DACE超参数组合



