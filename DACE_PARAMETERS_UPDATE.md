# ✅ DACE参数更新说明

## 🎯 重要更新

根据DACE论文的明确说明，已将所有默认参数更新为论文推荐值。

### 论文原文

> "For DACE, our default configuration uses a scaling factor of **αscale = 0.05** and a difficulty threshold of **βthreshold = 0.4**."

---

## 📝 更新内容

### 修改前 ❌

```python
# 旧的默认值（不正确）
alpha_scale: float = 0.1
beta_threshold: float = 0.5
```

### 修改后 ✅

```python
# 论文推荐的默认值（正确）
alpha_scale: float = 0.05
beta_threshold: float = 0.4
```

---

## 📁 更新的文件

### 1. 核心算法文件

**文件**: `verl/trainer/ppo/core_algos.py`  
**行数**: 317-318

```python
@register_adv_est(AdvantageEstimator.DACE)
def compute_dace_advantage(
    ...
    alpha_scale: float = 0.05,        # ✅ 更新为0.05
    beta_threshold: float = 0.4,      # ✅ 更新为0.4
    ...
):
```

### 2. Trainer文件

**文件**: `verl/trainer/ppo/ray_trainer.py`  
**行数**: 479-481

```python
# Get DACE hyperparameters from config (paper defaults: α=0.05, β=0.4)
alpha_scale = config.get("dace_alpha_scale", 0.05)        # ✅ 更新
beta_threshold = config.get("dace_beta_threshold", 0.4)    # ✅ 更新
```

### 3. ArcherCodeR训练脚本

**文件**: `scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh`  
**行数**: 31-32

```bash
# DACE-specific hyperparameters (paper defaults)
dace_alpha_scale=0.05          # ✅ 更新为0.05
dace_beta_threshold=0.4        # ✅ 更新为0.4
```

### 4. GSM8K/MATH训练脚本

**文件**: `scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-gsm8k-math.sh`  
**行数**: 25, 31

```bash
DACE_ALPHA_SCALE=0.05          # ✅ 更新为0.05
DACE_BETA_THRESHOLD=0.4        # ✅ 更新为0.4
```

### 5. 文档更新

**新增文档**:
- ✅ `docs/DACE_PAPER_DEFAULTS.md` - 详细说明论文默认值
- ✅ 更新 `scripts-iclr-server/train/DACE_vs_GRPO_comparison.md`

---

## 🔍 参数对比

| 参数 | 旧值 | 新值 | 来源 |
|------|------|------|------|
| `alpha_scale` | ~~0.1~~ | **0.05** | 论文 |
| `beta_threshold` | ~~0.5~~ | **0.4** | 论文 |

---

## 💡 参数含义

### α_scale = 0.05

**含义**: 内在奖励是外部奖励的5%

**效果**:
- 保守的内在奖励强度
- 确保外部奖励（正确性）仍然主导
- 提供足够的探索/利用引导

**示例**:
```python
external_reward = 1.0 (正确答案)
certainty = 0.5 (高确定性)
intrinsic_reward = 0.05 × 0.5 = 0.025 (2.5%的额外奖励)
```

### β_threshold = 0.4

**含义**: 当difficulty < 0.4时视为"简单"，对应success_rate > 60%

**效果**:
- 60%正确率是"掌握"的自然分界点
- 偏向让更多任务快速收敛（exploitation）
- 仍保留足够的困难任务进行探索

**分类**:
```
difficulty < 0.4 (success_rate > 60%) → 简单任务 → α = +0.05
difficulty ≥ 0.4 (success_rate ≤ 60%) → 困难任务 → α = -0.05
```

---

## 🎯 使用建议

### 首次使用（强烈推荐）

**直接使用论文默认值**：

```bash
algorithm.dace_alpha_scale=0.05
algorithm.dace_beta_threshold=0.4
```

**原因**:
- ✅ 论文作者经过充分实验验证
- ✅ 在多个数据集上表现稳定
- ✅ 平衡了效果和稳定性

### 需要调优时

只在以下情况才考虑调整：

#### 情况1: 训练不稳定
```bash
algorithm.dace_alpha_scale=0.03    # 减小内在奖励
```

#### 情况2: 数据集整体简单
```bash
algorithm.dace_beta_threshold=0.3   # 降低阈值
```

#### 情况3: 数据集整体困难
```bash
algorithm.dace_beta_threshold=0.5   # 提高阈值
```

#### 情况4: 需要更强探索信号
```bash
algorithm.dace_alpha_scale=0.1      # 增大内在奖励
```

---

## ⚠️ 重要提醒

### 为什么之前使用0.1和0.5？

之前的默认值（α=0.1, β=0.5）是基于：
- 常见的超参数取值范围
- 一般性的探索/利用平衡

但这**不是论文推荐的值**！

### 为什么要用论文值？

1. **权威性**: 论文作者经过大量实验确定
2. **稳定性**: 在多个任务上验证过
3. **可复现**: 与论文结果对齐
4. **基准线**: 作为调优的起点

### 如果想实验其他值？

完全可以！但建议：

1. **先测试论文默认值** (α=0.05, β=0.4)
2. **作为baseline对比**
3. **记录实验结果**
4. **理解改动原因**

---

## 📊 预期影响

### 从 (0.1, 0.5) 改为 (0.05, 0.4)

#### α_scale: 0.1 → 0.05

**影响**:
- ✅ 内在奖励影响减半
- ✅ 训练更稳定
- ⚠️ 探索/利用信号略弱

**适合**:
- 大多数标准数据集
- 追求稳定训练的场景

#### β_threshold: 0.5 → 0.4

**影响**:
- ✅ 更多任务被视为"简单"
- ✅ 更快收敛到exploitation
- ⚠️ 可能减少困难任务的探索

**适合**:
- 难度分布均衡的数据集
- 希望快速收敛的场景

---

## ✅ 检查清单

更新后请确认：

- [ ] 代码中默认值已更新 (core_algos.py)
- [ ] Trainer中默认值已更新 (ray_trainer.py)
- [ ] 训练脚本已更新 (run_DACE-*.sh)
- [ ] 文档已更新
- [ ] 理解新参数的含义
- [ ] 准备监控训练效果

---

## 📚 参考文档

- **详细说明**: `docs/DACE_PAPER_DEFAULTS.md`
- **参数对比**: `scripts-iclr-server/train/DACE_vs_GRPO_comparison.md`
- **算法解释**: `docs/DACE_ALGORITHM_EXPLANATION.md`

---

## 🎓 总结

| 方面 | 更新内容 |
|------|---------|
| **参数** | α: 0.1→0.05, β: 0.5→0.4 |
| **依据** | DACE论文明确说明 |
| **影响** | 更稳定，更接近论文结果 |
| **建议** | 首次使用直接采用 |

**关键点**:
1. ✅ 所有默认值已更新为论文推荐值
2. ✅ 使用 α=0.05, β=0.4 作为起点
3. ✅ 只在有明确理由时才调整
4. ✅ 始终以论文值作为baseline对比

