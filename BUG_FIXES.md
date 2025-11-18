# Bug Fixes for Self-RL Methods

## 问题总结

在运行新实现的三个Self-RL方法时，发现了两个关键bug：

### Bug 1: Token Entropy 方法缺少 `entropys` 键

**错误信息**:
```
KeyError: 'key "entropys" not found in TensorDict with keys [..., 'prob_disparitys', ..., 'self_certaintys', ...]'
```

**根本原因**:
在 `verl/trainer/ppo/ray_trainer.py` 的第1673行，代码在计算完entropy metrics后删除了 `entropys` 键：
```python
old_log_prob.batch.pop("entropys")
```

这导致 `TOKEN_ENTROPY` advantage estimator 无法访问所需的 `entropys` 数据。

**修复方案**:
修改 `ray_trainer.py` 第1673-1676行，只在不需要 `entropys` 的情况下才删除它：
```python
# Only remove entropys if not needed by advantage estimator
if self.config.algorithm.adv_estimator != AdvantageEstimator.TOKEN_ENTROPY:
    old_log_prob.batch.pop("entropys")
batch = batch.union(old_log_prob)
```

### Bug 2: `_forward_micro_batch` 返回值数量不匹配

**错误信息**:
```
ValueError: too many values to unpack (expected 3)
File "verl/workers/actor/dp_actor.py", line 472, in update_policy
    entropy, log_prob, _ = self._forward_micro_batch(...)
```

**根本原因**:
在 `verl/workers/actor/dp_actor.py` 的 `update_policy` 方法中（第472行），调用 `_forward_micro_batch` 时：
- 期望返回3个值：`entropy, log_prob, _`
- 但实际返回4个值：`entropy, log_prob, self_certainty, prob_disparity`

这是因为在之前的修改中，我们更新了 `_forward_micro_batch` 的签名以支持 `prob_disparity` 计算，但忘记更新 `update_policy` 中的调用。

**修复方案**:
修改 `dp_actor.py` 第472-477行，正确接收4个返回值并传递所有必需参数：
```python
entropy, log_prob, _, _ = self._forward_micro_batch(
    micro_batch=data, 
    temperature=temperature, 
    calculate_entropy=calculate_entropy,
    calculate_self_certainty=False,
    calculate_prob_disparity=False
)
```

## 修复文件列表

1. **`verl/workers/actor/dp_actor.py`** (第472-477行)
   - 更新 `update_policy` 方法中的 `_forward_micro_batch` 调用
   - 正确接收4个返回值
   - 传递 `calculate_prob_disparity=False` 参数

2. **`verl/trainer/ppo/ray_trainer.py`** (第1673-1676行)
   - 添加条件判断，只在不需要时删除 `entropys`
   - 确保 `TOKEN_ENTROPY` 方法可以访问 `entropys` 数据

## 验证

修复后，所有四种Self-RL方法应该都能正常运行：
- ✅ **Self-Certainty (INTUITOR)**: 使用 `self_certaintys`
- ✅ **Trajectory-Level Entropy**: 使用 `old_log_probs`
- ✅ **Token-Level Entropy**: 使用 `entropys`
- ✅ **Probability Disparity**: 使用 `prob_disparitys`

## 测试建议

重新运行失败的训练脚本：

```bash
# Token Entropy
sbatch scripts-iclr-server/token_entropy/slurm-TokenEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh

# Trajectory Entropy
sbatch scripts-iclr-server/trajectory_entropy/slurm-TrajectoryEntropy-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh

# Probability Disparity
sbatch scripts-iclr-server/prob_disparity/slurm-ProbDisparity-Qwen2.5-1.5B-2K-8K-8k-batch64-no-kl-simple_ray.sh
```

## 技术细节

### 数据流

1. **生成阶段** (`fsdp_workers.py`):
   ```python
   output, entropys, self_certaintys, prob_disparitys = self.actor.compute_log_prob(
       data=data, 
       calculate_entropy=True, 
       calculate_self_certainty=True,
       calculate_prob_disparity=True
   )
   ```

2. **数据传播** (`fsdp_workers.py`):
   ```python
   output = DataProto.from_dict(
       tensors={"old_log_probs": output, "entropys": entropys, 
                "self_certaintys": self_certaintys, "prob_disparitys": prob_disparitys}
   )
   ```

3. **Advantage计算** (`ray_trainer.py`):
   ```python
   # 现在 entropys 会保留在 batch 中，供 TOKEN_ENTROPY 使用
   batch = batch.union(old_log_prob)
   batch = compute_advantage(batch, adv_estimator=...)
   ```

4. **训练更新** (`dp_actor.py`):
   ```python
   # update_policy 中不需要计算 prob_disparity
   entropy, log_prob, _, _ = self._forward_micro_batch(
       ..., calculate_prob_disparity=False
   )
   ```

## 日期

修复日期: 2025-11-18

