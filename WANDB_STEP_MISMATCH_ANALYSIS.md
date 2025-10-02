# WandB Step 与 Checkpoint Step 不匹配分析

## 问题现象

**观察到的情况**:
- Checkpoint 文件名: `global_step_80`
- WandB 显示的步骤: step 130+

**差异**: 50 步

---

## 可能原因分析

### 原因 1: 之前从 step 80 恢复后继续训练 ⭐ (最可能)

**时间线**:
```
第一次训练:
  Step 1 → 10 → 20 → ... → 80 (保存 checkpoint)
  ↓ 继续训练
  Step 81 → 82 → ... → 130 (WandB 记录到这里)
  ↓ 训练中断/完成
  
第二次恢复训练:
  加载 global_step_80
  WandB resume="allow" → 发现之前已到 step 130
  ↓
  尝试从 step 80 继续记录
  ⚠️ 冲突！step 80 < 130
```

**验证方法**:
```bash
# 检查是否有更高步骤的 checkpoint 曾经存在
ls -la output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/

# 查看 wandb 历史
ls -la wandb/
```

---

### 原因 2: Save Frequency 配置

**代码逻辑**:
```python
# 训练循环
for each_batch:
    # Step 81: 训练
    tracklogger.log(data=metrics, step=81)  # WandB 记录 step 81
    
    if step % save_freq == 0:
        _save_checkpoint()  # 可能不保存
    
    self.global_steps += 1  # step → 82
```

**检查你的配置**:
```bash
# 在你的脚本中
save_freq=10  # 每 10 步保存一次
```

如果 `save_freq=10`:
- Step 80: 保存 checkpoint ✅
- Step 81-89: 不保存，但 WandB 持续记录
- Step 90: 保存下一个 checkpoint

如果训练在 step 130 中断，但最后保存的 checkpoint 是 step 80，这是正常的。

---

### 原因 3: Checkpoint 清理策略

**检查这个配置**:
```bash
# 你的脚本第 146-147 行
+trainer.max_actor_ckpt_to_keep=20
+trainer.max_critic_ckpt_to_keep=20
```

可能的情况：
- 训练实际到了 step 130
- 中间的 checkpoint (90, 100, 110, 120, 130) 被自动删除了
- 只保留了 step 80 (可能是某个特定原因)

---

### 原因 4: 不同的训练运行共享同一个 WandB ID

**WandB 初始化**:
```python
wandb.init(
    project=project_name,
    name=experiment_name,
    id=experiment_name,  # ⚠️ 关键：使用 experiment_name 作为 ID
    resume="allow"       # 允许恢复
)
```

**可能的场景**:
```
运行 1 (GRPO):
  实验名: Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl
  WandB ID: Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl
  训练到 step 130
  保存 checkpoint: global_step_80 (中途保存)

运行 2 (Intuitor from GRPO):
  实验名: Archer-Intuitor-Selective-... (不同)
  但从 global_step_80 恢复
  WandB 尝试恢复上一次的运行 → 发现 step 130
```

---

## 如何确认真实原因

### 检查 1: 查看 checkpoint 目录

```bash
cd /mnt/people/zhuoterq/xiaolong-swebench/ArcherCodeR
ls -lht output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/

# 查看是否有其他步骤的 checkpoint
find output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/ -name "global_step_*" -type d | sort
```

### 检查 2: 查看训练日志

```bash
# 查看原始训练的日志
grep "Saving checkpoint" output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/*.log | tail -20

# 查看最后记录的步骤
grep "Step.*completed" output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/*.log | tail -5
```

### 检查 3: 查看 WandB 历史

```bash
# 查看 wandb 本地缓存
ls -la wandb/
cat wandb/latest-run/run-*.wandb | grep -a step | tail -20
```

### 检查 4: 查看 dataloader 状态

```bash
# dataloader 会保存当前的迭代状态
# 如果存在，可以看到实际训练到哪一步
python3 << EOF
import torch
path = "output/ArcherCodeR/Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/global_step_80/data.pt"
try:
    state = torch.load(path, weights_only=False)
    print("Dataloader state:", state)
except Exception as e:
    print(f"Cannot load: {e}")
EOF
```

---

## 正常的训练逻辑（供对比）

### 场景 A: 正常训练流程

```
Step 79:
  ├─ 训练 batch
  ├─ tracklogger.log(step=79)
  └─ global_steps += 1 → 80

Step 80:
  ├─ 训练 batch
  ├─ tracklogger.log(step=80)
  ├─ 检查: 80 % 10 == 0 ✅
  ├─ _save_checkpoint() → 保存 global_step_80
  └─ global_steps += 1 → 81

Step 81:
  ├─ 训练 batch
  ├─ tracklogger.log(step=81)
  └─ global_steps += 1 → 82
```

**关键点**:
- Checkpoint 名称 = 保存时的 `global_steps`
- WandB 记录的 step = 当时的 `global_steps`
- 两者应该一致（除非继续训练了）

---

## 最可能的真相

基于你的脚本配置 (`save_freq=10`, `test_freq=10`):

```
原始 GRPO 训练:
  Step 1 → 10 (保存) → 20 (保存) → ... → 80 (保存) → 90 (保存)
  ↓ 继续训练
  Step 100 (保存) → 110 (保存) → 120 (保存) → 130 (保存)
  ↓ 达到 max_ckpt_to_keep=20
  自动删除: global_step_90, 100, 110, 120, 130
  保留: global_step_80 (实际上应该还保留了更多)
```

**但如果只有 global_step_80 存在**，更可能是：

```
训练到 step 80 → 保存 checkpoint
继续训练 → step 81, 82, ..., 130 (WandB 记录)
中途崩溃/中断 → 没有保存后续 checkpoint
或者手动选择使用 step 80 的 checkpoint
```

---

## 对当前问题的影响

**好消息**: 这不影响你当前的修复方案 ✅

修复后的行为:
```
从 global_step_80 恢复:
  ├─ 加载模型权重: ✅ 正确的 step 80 模型
  ├─ global_steps = 80
  ├─ WandB 创建新运行（避免冲突）
  └─ 从 step 80 继续训练和记录 ✅
```

无论之前的 WandB 是 step 130 还是其他值，新的运行都会从 step 80 正确开始。

---

## 建议

1. **确认 checkpoint 实际是 step 80 的模型** ✅
   ```bash
   # 如果有疑虑，可以检查模型文件的时间戳
   ls -lh output/.../global_step_80/actor/
   ```

2. **记录完整的训练日志** 📝
   ```bash
   # 在脚本中已经有了
   tee ${CKPTS_DIR}/${project_name}_${exp_name}_intuitor_selective.log
   ```

3. **清理 WandB 缓存（可选）** 🧹
   ```bash
   # 如果想完全重新开始
   rm -rf wandb/
   ```

4. **使用唯一的实验名** ✨
   ```bash
   # 已经在用不同的名称，很好！
   exp_name='Archer-Intuitor-Selective-...'  # 不同于 GRPO
   ```

---

## 总结

**为什么 checkpoint 是 step 80 但 WandB 是 step 130?**

最可能的原因：
1. ✅ 之前训练从 step 80 继续到了 step 130
2. ✅ WandB 持续记录到 step 130
3. ✅ 但你选择使用 step 80 的 checkpoint 来恢复（可能因为 step 130 的 checkpoint 被删除或不存在）

**这正常吗？**

✅ 完全正常！特别是在：
- 训练中断后恢复
- 使用中间 checkpoint 而不是最新的
- 比较不同训练阶段的效果

**现在的修复是否解决问题？**

✅ 是的！新的 WandB 运行会从 step 80 正确开始，不会有冲突。

