# DACE算法简明教程

## 🎯 核心思想（一句话）

**根据任务难度，自适应地调整"鼓励高确定性"还是"鼓励低确定性（探索）"**

---

## 📍 Difficulty在哪里计算？

### 位置
**文件**: `verl/trainer/ppo/core_algos.py`  
**函数**: `compute_dace_advantage()`  
**行数**: 第392-397行

### 代码
```python
for prompt_id in id2rewards:
    rewards = torch.stack(id2rewards[prompt_id])
    # 💡 Difficulty计算公式
    success_rate = torch.mean(rewards)
    difficulty = 1.0 - success_rate
    id2difficulty[prompt_id] = difficulty
```

### 计算逻辑

1. **输入**: 对同一个prompt生成的多个响应（比如16个）及其奖励
   ```python
   prompt_0的8个响应的奖励: [1, 0, 1, 1, 0, 1, 0, 1]
   ```

2. **计算成功率**: 
   ```python
   success_rate = mean([1, 0, 1, 1, 0, 1, 0, 1]) = 5/8 = 0.625
   ```

3. **计算difficulty**: 
   ```python
   difficulty = 1 - 0.625 = 0.375
   ```

4. **解释**:
   - `difficulty = 0.0`: 所有响应都对 → **超级简单**
   - `difficulty = 0.375`: 62.5%正确 → **比较简单**
   - `difficulty = 0.5`: 一半对一半错 → **中等难度**
   - `difficulty = 0.8`: 只有20%正确 → **很困难**
   - `difficulty = 1.0`: 全错 → **超级困难**

---

## 🔄 完整数据流

```
Step 1: 生成阶段
prompt_0 → [resp_0, resp_1, ..., resp_7]  (8个响应)
prompt_1 → [resp_8, resp_9, ..., resp_15] (8个响应)
   ↓
Step 2: 计算外部奖励
resp_0 → R_ext = 1 (正确)
resp_1 → R_ext = 0 (错误)
...
   ↓
Step 3: 计算difficulty (按prompt分组)
prompt_0: rewards = [1,0,1,1,0,1,0,1]
         success_rate = 5/8 = 0.625
         difficulty = 1 - 0.625 = 0.375  ← 💡 在这里！
   ↓
Step 4: 计算adaptive coefficient
difficulty = 0.375 < β_threshold (0.5)
→ 这是简单任务
→ α = +0.1 (正系数)
   ↓
Step 5: 计算内在奖励
对prompt_0的每个响应:
  certainty = -mean(log_prob)
  R_int = α × certainty = 0.1 × certainty
  R_total = R_ext + R_int
```

---

## 🎲 具体例子

### 场景设置
- `n_responses_per_prompt = 8`
- `α_scale = 0.1`  
- `β_threshold = 0.5`

### Prompt A: 简单数学题 "1+1=?"

**Step 1: 生成8个响应**
```python
responses = [
    "2", "2", "2", "2",  # 4个正确
    "2", "2", "3", "1"   # 2个正确，2个错误
]
rewards = [1, 1, 1, 1, 1, 1, 0, 0]
```

**Step 2: 计算difficulty**
```python
success_rate = 6/8 = 0.75
difficulty = 1 - 0.75 = 0.25  # 简单任务！
```

**Step 3: 计算α**
```python
sign = sign(0.5 - 0.25) = sign(0.25) = +1
α = 0.1 × (+1) = +0.1  # 正系数
```

**Step 4: 对每个响应计算总奖励**
```python
# Response 0: "2" (正确)
R_ext = 1.0
certainty = 0.3  (高确定性，因为"2"的概率很高)
R_int = 0.1 × 0.3 = 0.03
R_total = 1.0 + 0.03 = 1.03  ✅ 奖励正确+高确定性

# Response 6: "3" (错误)
R_ext = 0.0
certainty = 2.5  (低确定性，因为"3"的概率较低)
R_int = 0.1 × 2.5 = 0.25
R_total = 0.0 + 0.25 = 0.25  ❌ 仍然是正的，但远小于正确答案
```

**效果**: 模型学到在简单题上要用高确定性给出正确答案

---

### Prompt B: 困难奥数题

**Step 1: 生成8个响应**
```python
responses = [复杂数学推导...]
rewards = [0, 0, 0, 1, 0, 0, 0, 0]  # 只有1个正确
```

**Step 2: 计算difficulty**
```python
success_rate = 1/8 = 0.125
difficulty = 1 - 0.125 = 0.875  # 困难任务！
```

**Step 3: 计算α**
```python
sign = sign(0.5 - 0.875) = sign(-0.375) = -1
α = 0.1 × (-1) = -0.1  # 负系数！
```

**Step 4: 对每个响应计算总奖励**
```python
# Response 3: 正确答案
R_ext = 1.0
certainty = 1.2  (中等确定性)
R_int = -0.1 × 1.2 = -0.12
R_total = 1.0 + (-0.12) = 0.88  ✅ 正确答案仍然最高

# Response 0: 错误但在探索 (低确定性)
R_ext = 0.0
certainty = 3.5  (很低的确定性，在尝试不同方法)
R_int = -0.1 × 3.5 = -0.35
R_total = 0.0 + (-0.35) = -0.35  
相对优势 = -0.35 - (-0.5) = +0.15  ✅ 比其他错误答案好

# Response 1: 错误且确定 (高确定性)
R_ext = 0.0
certainty = 0.8  (高确定性，但错了)
R_int = -0.1 × 0.8 = -0.08
R_total = 0.0 + (-0.08) = -0.08
相对优势 = -0.08 - (-0.5) = +0.42  ❌ 给予较少的相对优势
```

**效果**: 模型学到在困难题上要持续探索（低确定性），不要过早收敛到错误答案

---

## 📊 Difficulty对训练的影响

| Difficulty范围 | 任务类型 | α符号 | 训练策略 | 例子 |
|---------------|---------|------|---------|------|
| 0.0 - 0.3 | 很简单 | 正 (+) | 强化高确定性正确答案 | "1+1=?" |
| 0.3 - 0.5 | 比较简单 | 正 (+) | 鼓励exploitation | 简单算术 |
| 0.5 | 临界点 | 0 | 纯外部奖励 | - |
| 0.5 - 0.7 | 比较困难 | 负 (-) | 鼓励exploration | 中等数学题 |
| 0.7 - 1.0 | 很困难 | 负 (-) | 强烈鼓励探索 | 奥数、研究级题目 |

---

## 🔧 使用方法

### 1. 训练脚本
```bash
bash scripts-iclr-server/train/run_DACE-Qwen2.5-1.5B-Archer.sh
```

### 2. 关键配置
```bash
adv_estimator=dace                    # 使用DACE算法
dace_alpha_scale=0.1                  # 内在奖励强度
dace_beta_threshold=0.5               # 难度阈值
n_resp_per_prompt=16                  # 每个prompt生成16个响应（用于估计difficulty）
```

### 3. 调参建议

**α_scale (内在奖励强度)**:
- 从 `0.1` 开始
- 如果模型探索不足 → 增加到 `0.2` 或 `0.3`
- 如果训练不稳定 → 减少到 `0.05`

**β_threshold (难度阈值)**:
- 从 `0.5` 开始
- 如果数据集偏难 → 增加到 `0.6-0.7` (更多任务被视为困难)
- 如果数据集偏简单 → 减少到 `0.3-0.4` (更多任务被视为简单)

**n_responses**:
- 最少 `8` 个（统计意义）
- 推荐 `16` 个（平衡准确性和效率）
- 困难数据集可用 `32` 个（更准确的difficulty估计）

---

## ✅ 优势总结

1. **自动化**: 不需要手动区分简单/困难任务
2. **自适应**: 根据模型当前能力动态调整策略
3. **高效**: 简单任务快速收敛，困难任务持续探索
4. **统一框架**: 一套算法处理所有难度的任务

## 🆚 与其他方法对比

| 方法 | 外部奖励 | 内在奖励 | 自适应 | Difficulty计算 |
|-----|---------|---------|--------|----------------|
| GRPO | ✅ | ❌ | ❌ | 不计算 |
| INTUITOR | ❌ | ✅ (固定) | ❌ | 不计算 |
| **DACE** | ✅ | ✅ (自适应) | ✅ | **多次采样统计** |

---

## 📝 快速检查清单

训练DACE前确认：
- [ ] `n_resp_per_prompt >= 8` (确保difficulty估计准确)
- [ ] 数据集有不同难度的题目混合
- [ ] `α_scale` 在0.05-0.3之间
- [ ] `β_threshold` 在0.3-0.7之间
- [ ] 启用了 `norm_adv_by_std_in_grpo=True`

看到这些日志说明DACE正常工作：
```
-------------------------------- This is DACE --------------------------------
DACE Hyperparameters:
  α_scale (intrinsic reward scaling): 0.1
  β_threshold (difficulty threshold): 0.5
External reward range: [0.0000, 1.0000]
Certainty range: [0.5234, 3.8471]
Advantage range: [2.1543, -1.8765]
-------------------------------- End of DACE --------------------------------
```

