# WandB 步骤号警告问题解决方案

## 问题描述

运行训练脚本时出现警告：
```
wandb: WARNING Tried to log to step 132 that is less than the current step 137. 
Steps must be monotonically increasing, so this data will be ignored.
```

## 问题原因

这个问题出现在**从中间 checkpoint 回退训练**的场景：

1. **训练历史**: 
   - 原始训练：Step 1 → 80 (保存) → 130 (保存)
   - WandB 已记录到 Step 130 ✅

2. **你的选择**: 
   - ❌ 不使用最新的 global_step_130
   - ✅ 选择使用中间的 global_step_80
   - 原因：可能 step 80 效果更好，或想从这个点尝试新算法

3. **跨实验恢复**: 
   - 从 `Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/global_step_80` 
   - 恢复到新实验 `Archer-Intuitor-Selective-Qwen2.5-1.5B-2k-8k-batch64-no-kl-from-grpo`

4. **WandB Resume 模式**: 
   - WandB 使用 `resume="allow"` 和 `id=experiment_name`
   - 尝试恢复之前的运行

5. **步骤号冲突**: 
   - 新实验从步骤 80 开始（回退了！）
   - 但 WandB 认为之前已经到达步骤 130
   - 当尝试记录步骤 80-132 时，WandB 发现 < 130，拒绝记录 ❌

6. **WandB 限制**: WandB 要求步骤号必须严格单调递增（不允许回退）

## 解决方案

### ✅ 方案 1：自动检测跨实验恢复（已实现）

**优点**: 自动处理，无需手动修改脚本

我已经修改了 `verl/utils/tracking.py`，添加了智能检测：
- 当检测到从不同实验的 checkpoint 恢复时，自动启动新的 WandB 运行
- 避免步骤号冲突

**代码逻辑**:
```python
# 检测是否是跨实验恢复
if resume_mode == "resume_path" and resume_from_path:
    if experiment_name not in resume_from_path:
        # 从不同实验恢复，启动新的 WandB 运行
        resume_mode = False
```

**使用方法**: 直接运行你的脚本即可，无需额外修改

---

### 方案 2：修改实验名称（最简单）

**适用场景**: 想要保持训练历史分开

在脚本 `intuitor_selective_Qwen-1.5B-2k-8k-8k-batch64-no-kl-from-grpo.sh` 中修改：

```bash
# 第 18 行，添加版本号或日期
exp_name='Archer-Intuitor-Selective-Qwen2.5-1.5B-2k-8k-batch64-no-kl-from-grpo-v2'
# 或者
exp_name='Archer-Intuitor-Selective-Qwen2.5-1.5B-2k-8k-batch64-no-kl-from-grpo-20251002'
```

---

### 方案 3：禁用 WandB Resume（适用于全新开始）

在脚本中添加环境变量：

```bash
# 在脚本开头添加
export WANDB_RESUME="never"  # 强制 WandB 总是开始新运行
```

---

### 方案 4：手动管理步骤偏移（适用于需要连续步骤号）

如果你想让新实验的步骤号从之前的最大值继续：

修改 `verl/trainer/ppo/ray_trainer.py` 中的 `_load_checkpoint` 方法：

```python
def _load_checkpoint(self):
    # ... 现有代码 ...
    
    # 从 checkpoint 路径提取步骤号
    self.global_steps = int(global_step_folder.split("global_step_")[-1])
    
    # 如果 WandB 已有更大的步骤号，使用它
    if hasattr(wandb, 'run') and wandb.run is not None:
        wandb_step = wandb.run.step
        if wandb_step > self.global_steps:
            print(f"⚠️  WandB step ({wandb_step}) > checkpoint step ({self.global_steps})")
            print(f"   Continuing from WandB step {wandb_step}")
            self.global_steps = wandb_step
```

---

## 推荐方案

### 对于你的情况：

1. **已自动修复** ✅: 方案 1 已经实现，你的脚本现在应该可以正常运行而不会出现警告

2. **验证修复**: 重新运行脚本，查看是否有这样的输出：
   ```
   🔄 Detected cross-experiment resume: from 'Archer-Qwen2.5-1.5B-2K-8K-16resp-no-kl/global_step_80' to 'Archer-Intuitor-Selective-...'
      Starting new WandB run to avoid step conflicts
   ```

3. **如果仍有问题**: 使用方案 2，修改实验名称为唯一值

---

## 技术细节

### WandB 步骤管理机制

WandB 使用以下参数管理运行的唯一性和恢复：

```python
wandb.init(
    project=project_name,      # 项目名称
    name=experiment_name,      # 运行显示名称
    id=experiment_name,        # 运行唯一 ID（用于恢复）
    resume="allow"             # 允许恢复同 ID 的运行
)
```

- `id` 相同 + `resume="allow"` → 尝试恢复之前的运行
- `resume=False` → 总是开始新运行（即使 id 相同）

### 步骤号单调性要求

WandB 要求：`step(n+1) > step(n)`

违反此规则的数据会被**静默丢弃**（只显示警告），这可能导致：
- 训练指标缺失
- 图表不完整
- 难以追踪训练进度

---

## 预防措施

为避免将来出现类似问题：

1. **命名规范**: 跨实验恢复时，使用不同的实验名称
   ```bash
   # 原始训练
   exp_name='Model-Training-Phase1'
   
   # 从 Phase1 恢复继续训练其他算法
   exp_name='Model-Training-Phase2-Intuitor'
   ```

2. **清理 WandB 缓存**: 如果遇到持久性问题
   ```bash
   rm -rf wandb/
   ```

3. **使用唯一 ID**: 在配置中添加时间戳
   ```python
   import time
   experiment_name = f"{base_name}-{int(time.time())}"
   ```

---

## 验证修复

运行以下命令检查是否还有警告：

```bash
bash scripts-iclr-server/intuitor/intuitor_selective_Qwen-1.5B-2k-8k-8k-batch64-no-kl-from-grpo.sh 2>&1 | grep -i "WARNING Tried to log"
cong xia

如果没有输出，说明问题已解决 ✅

---

## 联系支持

如果问题仍然存在，请检查：
1. `verl/utils/tracking.py` 的修改是否生效
2. WandB 版本: `pip show wandb`
3. 完整的错误日志

---

生成时间: 2025-10-02


