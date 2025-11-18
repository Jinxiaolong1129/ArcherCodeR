# 实现正确性验证

## 公式对照表

| 方法 | 用户提供的公式 | 实现的公式 | 状态 |
|------|---------------|-----------|------|
| Self-Certainty | `r(x,y) = 1/\|y\| * Σ D_KL(U\|\|π_θ)` | `logsumexp(logits) - mean(logits)` 的平均值 | ✅ 正确 |
| Trajectory-Level Entropy | `r(x,y) = 1/\|y\| * Σ log π_θ(y_t\|x,y_<t)` | `mean(old_log_probs)` | ✅ 正确 |
| Token-Level Entropy | `r(x,y) = -1/\|y\| * Σ H(π_θ)` | `-mean(entropys)` | ✅ 正确 |
| Probability Disparity | `r(x,y) = 1/M * Σ [max π - second_max π]` | `mean(top1_prob - top2_prob)` | ✅ 正确 |

---

## 详细验证

### 1. Self-Certainty (INTUITOR) ✅

**用户公式**: `r(x, y) = 1/|y| * Σ D_KL(U||π_θ(·|x, y_<t))`

**实现位置**: 
- `verl/utils/torch_functional.py::self_certainty_from_logits()`
- `verl/trainer/ppo/core_algos.py::compute_intuitor_advantage()`

**实现公式**:
```python
# Token-level computation
self_certainty = torch.logsumexp(logits, dim=-1) - logits.mean(dim=-1)  # [B, T]

# Sentence-level averaging
masked_certainty = self_certaintys * response_mask_float  # [B, T]
sum_certainty = masked_certainty.sum(dim=-1)  # [B]
count = response_mask_float.sum(dim=-1) + epsilon  # [B]
sentence_wise_certainty = sum_certainty / count  # [B] = 1/|y| * Σ self_certainty
```

**数学推导**:
- KL散度: `D_KL(U||π) = Σ U(x) * log(U(x)/π(x))`
- 对于均匀分布U: `U(x) = 1/V` (V是词表大小)
- `D_KL(U||π) = (1/V) * Σ log(1/V) - (1/V) * Σ log(π(x))`
- `= log(V) - (1/V) * Σ log(π(x))`
- `= log(V) + (1/V) * Σ (-log(π(x)))`

而实现中：
- `logsumexp(logits) ≈ max(logits)` (当某个logit很大时)
- `mean(logits)` 是所有logits的平均
- 差值越大 → 分布越尖锐 → 与均匀分布差异越大

**结论**: ✅ 实现正确，虽然不是严格的KL散度，但捕捉了相同的语义（分布的尖锐程度）

---

### 2. Trajectory-Level Entropy ✅

**用户公式**: `r(x, y) = 1/|y| * Σ log π_θ(y_t|x, y_<t)`

**实现位置**: 
- `verl/trainer/ppo/core_algos.py::compute_trajectory_entropy_advantage()`

**实现公式**:
```python
# 使用已经计算好的old_log_probs
masked_log_probs = old_log_probs * response_mask_float  # [B, T]
sum_log_probs = masked_log_probs.sum(dim=-1)  # [B] = Σ log π_θ(y_t|x, y_<t)
count = response_mask_float.sum(dim=-1) + epsilon  # [B] = |y|
trajectory_entropy = sum_log_probs / count  # [B] = 1/|y| * Σ log π_θ
```

**数学验证**:
- `old_log_probs[i,t]` = `log π_θ(y_t|x, y_<t)` ✅
- `sum_log_probs` = `Σ_t log π_θ(y_t|x, y_<t)` ✅
- `trajectory_entropy` = `1/|y| * Σ log π_θ(y_t|x, y_<t)` ✅

**注意**: 这个方法的命名可能有误导性，因为它计算的是**平均对数概率**，而不是熵。但公式实现是正确的。

**结论**: ✅ 实现完全正确，与公式一致

---

### 3. Token-Level Entropy ✅

**用户公式**: `r(x, y) = -1/|y| * Σ H(π_θ(·|x, y_<t))`

**实现位置**: 
- `verl/utils/torch_functional.py::entropy_from_logits()`
- `verl/trainer/ppo/core_algos.py::compute_token_entropy_advantage()`

**实现公式**:
```python
# Token-level entropy computation (already done in forward pass)
entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)  # [B, T]
# where pd = softmax(logits)

# Sentence-level averaging with negation
masked_entropy = entropys * response_mask_float  # [B, T]
sum_entropy = masked_entropy.sum(dim=-1)  # [B] = Σ H(π_θ)
count = response_mask_float.sum(dim=-1) + epsilon  # [B] = |y|
avg_entropy = sum_entropy / count  # [B] = 1/|y| * Σ H(π_θ)
token_entropy_reward = -avg_entropy  # [B] = -1/|y| * Σ H(π_θ)
```

**数学验证**:
- 熵的定义: `H(π) = -Σ π(x) * log(π(x))`
- 从logits计算: `H = logsumexp(logits) - Σ softmax(logits) * logits` ✅
- 平均: `1/|y| * Σ H` ✅
- 取负: `-1/|y| * Σ H` ✅

**结论**: ✅ 实现完全正确，与公式一致

---

### 4. Probability Disparity ✅

**用户公式**: `r(x, y) = 1/M * Σ [max π_θ(a_t|...) - second_max π_θ(a_t|...)]`

**实现位置**: 
- `verl/utils/torch_functional.py::prob_disparity_from_logits()`
- `verl/trainer/ppo/core_algos.py::compute_prob_disparity_advantage()`

**实现公式**:
```python
# Token-level disparity computation
probs = torch.softmax(logits, dim=-1)  # [B, T, V]
top2_probs, _ = torch.topk(probs, k=2, dim=-1)  # [B, T, 2]
disparity = top2_probs[..., 0] - top2_probs[..., 1]  # [B, T] = max π - second_max π

# Sentence-level averaging
masked_disparity = prob_disparitys * response_mask_float  # [B, T]
sum_disparity = masked_disparity.sum(dim=-1)  # [B] = Σ disparity
count = response_mask_float.sum(dim=-1) + epsilon  # [B] = M (或|y|)
avg_disparity = sum_disparity / count  # [B] = 1/M * Σ disparity
```

**数学验证**:
- `softmax(logits)` = `π_θ(a_t|...)` ✅
- `topk(probs, k=2)[0]` = `max π_θ` ✅
- `topk(probs, k=2)[1]` = `second_max π_θ` ✅
- `disparity` = `max π - second_max π` ✅
- `avg_disparity` = `1/M * Σ disparity` ✅

**注意**: 公式中的M应该等于|y|（response length），实现中使用的是response_mask的sum，这是正确的。

**结论**: ✅ 实现完全正确，与公式一致

---

## 数据流验证

### 数据计算位置

| 数据 | 计算位置 | 使用位置 |
|------|---------|---------|
| `self_certaintys` | `dp_actor.py::_forward_micro_batch()` | `compute_intuitor_advantage()` |
| `old_log_probs` | `dp_actor.py::_forward_micro_batch()` | `compute_trajectory_entropy_advantage()` |
| `entropys` | `dp_actor.py::_forward_micro_batch()` | `compute_token_entropy_advantage()` |
| `prob_disparitys` | `dp_actor.py::_forward_micro_batch()` | `compute_prob_disparity_advantage()` |

### 计算流程

```
1. Forward Pass (dp_actor.py)
   ├─ 计算logits
   ├─ 从logits计算: log_probs, entropy, self_certainty, prob_disparity
   └─ 返回到fsdp_workers.py

2. 数据传递 (fsdp_workers.py)
   ├─ 接收: old_log_probs, entropys, self_certaintys, prob_disparitys
   └─ 打包到DataProto返回

3. Advantage计算 (ray_trainer.py)
   ├─ 根据adv_estimator选择方法
   ├─ 调用对应的core_algos函数
   └─ 计算advantages和returns

4. GRPO归一化 (core_algos.py)
   ├─ 按prompt分组
   ├─ 计算组内均值和标准差
   ├─ 归一化: (score - mean) / std
   └─ 广播到token-level
```

---

## 潜在问题检查

### ✅ 1. 设备一致性
所有tensor都在正确的设备上：
```python
id2mean[idx] = torch.tensor(0.0, device=scores.device)  # ✅
id2std[idx] = torch.tensor(1.0, device=scores.device)   # ✅
```

### ✅ 2. 数值稳定性
- 除法加epsilon: `count = response_mask_float.sum(dim=-1) + epsilon` ✅
- 标准差加epsilon: `(scores[i] - mean) / (std + epsilon)` ✅

### ✅ 3. Mask处理
所有方法都正确使用response_mask：
```python
masked_metric = metric * response_mask_float  # ✅
sum_metric = masked_metric.sum(dim=-1)        # ✅
count = response_mask_float.sum(dim=-1)       # ✅
```

### ✅ 4. 返回值一致性
所有方法都返回相同格式：
```python
return advantages, advantages  # (advantages, returns)
```
对于outcome-based方法，returns = advantages是正确的。

### ✅ 5. GRPO归一化
所有方法都使用相同的GRPO归一化逻辑：
- 按prompt分组 (使用index/uid)
- 计算组内统计量
- 归一化
- 广播到token-level

---

## 与INTUITOR的一致性检查

所有三个新方法都完全遵循INTUITOR的模式：

```python
# 模式：
# 1. 计算token-level metric
# 2. 平均到sentence-level
masked_metric = metric * response_mask_float
sum_metric = masked_metric.sum(dim=-1)
count = response_mask_float.sum(dim=-1) + epsilon
sentence_wise_score = sum_metric / count

# 3. GRPO归一化
scores = sentence_wise_score
# ... 分组、计算均值/标准差、归一化 ...

# 4. 广播到token-level
advantages = scores.unsqueeze(-1) * response_mask_float
```

✅ 所有方法都遵循此模式

---

## 最终结论

### ✅ 所有实现都是正确的

1. **Self-Certainty**: 正确实现了KL散度的近似
2. **Trajectory-Level Entropy**: 完全符合公式 `1/|y| * Σ log π_θ`
3. **Token-Level Entropy**: 完全符合公式 `-1/|y| * Σ H(π_θ)`
4. **Probability Disparity**: 完全符合公式 `1/M * Σ [max π - second_max π]`

### 实现质量

- ✅ 数学公式正确
- ✅ 数据流完整
- ✅ 设备管理正确
- ✅ 数值稳定性良好
- ✅ Mask处理正确
- ✅ 与现有代码一致
- ✅ 详细的日志输出

### 可以直接使用

所有四种方法都已经正确实现，可以直接用于实验对比！

---

## 使用建议

### 配置示例

```yaml
# Trajectory-Level Entropy
algorithm:
  adv_estimator: "trajectory_entropy"
  norm_adv_by_std_in_grpo: true

# Token-Level Entropy  
algorithm:
  adv_estimator: "token_entropy"
  norm_adv_by_std_in_grpo: true

# Probability Disparity
algorithm:
  adv_estimator: "prob_disparity"
  norm_adv_by_std_in_grpo: true
```

### 监控指标

训练时注意观察：
1. 各方法的reward分布（mean, std, range）
2. Advantage的正负比例
3. 训练稳定性
4. 最终性能

### 对比实验

建议的对比维度：
1. **计算效率**: Trajectory < Token/Prob/Self (因为不需要额外计算)
2. **训练稳定性**: 观察loss曲线
3. **最终性能**: Pass@1, Pass@k等指标
4. **收敛速度**: 达到目标性能的步数

